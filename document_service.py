"""
Dalilak AI — Document Intelligence Service (Phases 4, 7, 9)

Provides:
- analyze_document()  : GPT-4o powered document analysis (type detection, field extraction, warnings)
- review_contract()   : Deep contract review with clause analysis and Arabic recommendations
- get_clause_checklist(): Standard clause lists for common Lebanese contract types
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("dalilak.document_service")

# ── Prompts ───────────────────────────────────────────────────────────────────

_ANALYZE_SYSTEM = """\
أنت دليلك AI — محرك تحليل الوثائق القانونية والإدارية.
تلقّيت نص وثيقة من المستخدم. حلّلها وأرجع JSON صالحاً وفق هذا المخطط بالضبط:

{
  "document_type": "lease_contract"|"sale_contract"|"power_of_attorney"|"civil_record"|"property_document"|"company_document"|"identity_document"|"invoice"|"certificate"|"correspondence"|"unknown",
  "detected_country": "lebanon"|"syria"|"unknown",
  "detected_language": "ar"|"en"|"fr"|"mixed",
  "document_date": string|null,
  "extracted_fields": [
    {"label": string, "value": string, "confidence": "high"|"medium"|"low"}
  ],
  "parties": [{"role": string, "name": string|null}],
  "key_facts": [string],
  "related_procedures": [string],
  "missing_documents": [
    {
      "id": string,
      "title": string,
      "required": boolean,
      "reason": string,
      "priority": "low"|"medium"|"high"|"critical",
      "status": "missing"
    }
  ],
  "warnings": [{"level": "info"|"warning"|"critical", "message": string}],
  "suggested_next_actions": [
    {"label": string, "action_type": "upload_document"|"request_human_review"|"verify_source"|"download_checklist"|"ask_followup"}
  ],
  "confidence": {"level": "high"|"medium"|"low"|"unknown", "reason": string|null},
  "summary": string
}

قواعد صارمة:
- لا تخترع أسماء أو أرقام أو تواريخ غير موجودة في الوثيقة
- إذا لم تجد معلومة، ضع null أو قائمة فارغة
- حدّد نوع الوثيقة بدقة
- استخرج الحقول المهمة فقط (الأطراف، التاريخ، المبلغ، العقار، الرقم الرسمي)
- اكتب الملخص بالعربية دائماً حتى لو الوثيقة بلغة أخرى
- أضف id فريداً لكل وثيقة ناقصة (uuid v4 بصيغة مبسطة: doc_1, doc_2 ...)
"""

_CONTRACT_REVIEW_SYSTEM = """\
أنت دليلك AI — محرك مراجعة العقود القانونية.
تلقّيت نص عقد. راجعه وفق القانون اللبناني وأرجع JSON صالحاً وكاملاً:

{
  "document_type": "lease_contract"|"sale_contract"|"power_of_attorney"|"service_contract"|"employment_contract"|"unknown",
  "summary": string,
  "extracted_facts": {
    "parties": [string],
    "subject": string|null,
    "property": string|null,
    "duration": string|null,
    "amount": string|null,
    "currency": string|null,
    "payment_terms": string|null,
    "start_date": string|null,
    "end_date": string|null
  },
  "key_clauses_found": [
    {"clause": string, "found": boolean, "strength": "strong"|"acceptable"|"weak"|"missing"|"unclear", "notes": string|null}
  ],
  "missing_or_weak_clauses": [
    {
      "clause": string,
      "risk_level": "low"|"medium"|"high"|"critical",
      "why_it_matters": string,
      "recommendation": string,
      "suggested_clause_draft": string|null
    }
  ],
  "party_risk_balance": {
    "favors": "party_one"|"party_two"|"balanced"|"unclear",
    "notes": string
  },
  "practical_recommendations": [string],
  "questions_for_lawyer": [string],
  "risk_score": {
    "level": "low"|"medium"|"high"|"critical",
    "score": number,
    "reasons": [string]
  },
  "confidence": {"level": "high"|"medium"|"low"|"unknown", "reason": string|null},
  "disclaimer": string
}

للعقود الإيجارية، تحقق من وجود هذه البنود:
الأطراف والأهلية | وصف العقار | المدة | بدل الإيجار | العملة | الدفع |
التأمين | الصيانة | الخدمات والضرائب | الاستخدام | التأجير من الباطن |
التجديد | الفسخ | التأخير | التسليم | المخزون | القوة القاهرة | حل النزاعات |
المحكمة المختصة | التسجيل | التوقيع

قواعد صارمة:
- لا تخترع حقائق غير موجودة في العقد
- اكتب كل البنود المقترحة بالعربية حتى لو كان العقد بلغة أخرى
- أضف disclaimer واضحاً دائماً
- score يجب أن يكون عدداً بين 0 و 100
"""

# ── Standard Clause Lists ─────────────────────────────────────────────────────

LEASE_CLAUSES = [
    "الأطراف والأهلية القانونية",
    "وصف العقار الكامل",
    "مدة العقد وتاريخ البداية والنهاية",
    "بدل الإيجار الشهري",
    "العملة وطريقة الدفع",
    "مواعيد وآلية الدفع",
    "التأمين أو الضمان",
    "بند الصيانة (مسؤولية من؟)",
    "الخدمات والضرائب (من يدفع؟)",
    "الاستخدام المسموح به",
    "التأجير من الباطن أو التنازل",
    "التجديد التلقائي أو الاتفاقي",
    "الفسخ والإنهاء المبكر",
    "التأخير في الدفع والغرامات",
    "التسليم والتسلم وحالة العقار",
    "قائمة المنقولات والتجهيزات",
    "القوة القاهرة",
    "حل النزاعات",
    "المحكمة المختصة",
    "التسجيل والتوثيق",
    "توقيع الأطراف والشهود",
]

SALE_CLAUSES = [
    "الأطراف وأهليتهم",
    "وصف المبيع الكامل",
    "الثمن الإجمالي",
    "آلية الدفع والأقساط",
    "تاريخ نقل الملكية",
    "تسليم المبيع",
    "ضمان استحقاق الملكية",
    "ضمان الخلو من الحقوق والرهون",
    "التسجيل العقاري",
    "الضرائب والرسوم (من يتحمل؟)",
    "حق الفسخ والشروط",
    "القوة القاهرة",
    "حل النزاعات والمحكمة المختصة",
    "توقيع الأطراف والشهود والتوثيق",
]

POA_CLAUSES = [
    "هوية الموكّل والوكيل",
    "نطاق الصلاحيات بدقة",
    "مدة الوكالة",
    "صلاحية التفويض من الباطن",
    "الإجراءات المسموح بها",
    "التوقيع والتوثيق الرسمي",
    "حق الموكل في العدول",
]

_CLAUSE_LISTS: dict[str, list[str]] = {
    "lease_contract": LEASE_CLAUSES,
    "sale_contract": SALE_CLAUSES,
    "power_of_attorney": POA_CLAUSES,
}

# ── Public API ────────────────────────────────────────────────────────────────

def get_clause_checklist(doc_type: str) -> list[str]:
    """Return the expected clause list for a given document type."""
    return _CLAUSE_LISTS.get(doc_type, [])


async def analyze_document(text: str, file_name: str, oai_client: Any) -> dict:
    """
    Analyze a document using GPT-4o.

    Args:
        text: Extracted plaintext of the document (max ~8000 chars used)
        file_name: Original filename (hints at doc type)
        oai_client: An async OpenAI client instance

    Returns:
        DocumentAnalysis dict (see _ANALYZE_SYSTEM for schema)
    """
    excerpt = text[:8000]
    try:
        resp = await oai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM},
                {"role": "user", "content": f"اسم الملف: {file_name}\n\nنص الوثيقة:\n{excerpt}"},
            ],
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        # Ensure required keys exist
        result.setdefault("document_type", "unknown")
        result.setdefault("summary", "")
        result.setdefault("extracted_fields", [])
        result.setdefault("warnings", [])
        result.setdefault("missing_documents", [])
        result.setdefault("suggested_next_actions", [])
        result.setdefault("confidence", {"level": "unknown", "reason": None})
        return result
    except Exception as exc:
        logger.error("Document analysis failed for %s: %s", file_name, exc)
        return {
            "document_type": "unknown",
            "detected_country": "unknown",
            "detected_language": "unknown",
            "document_date": None,
            "extracted_fields": [],
            "parties": [],
            "key_facts": [],
            "related_procedures": [],
            "missing_documents": [],
            "warnings": [{"level": "warning", "message": "تعذّر التحليل التلقائي — يرجى المحاولة لاحقاً"}],
            "suggested_next_actions": [
                {"label": "طلب مراجعة بشرية", "action_type": "request_human_review"}
            ],
            "confidence": {"level": "unknown", "reason": str(exc)},
            "summary": "تعذّر التحليل التلقائي لهذه الوثيقة.",
        }


async def review_contract(text: str, file_name: str, oai_client: Any) -> dict:
    """
    Deep contract clause review using GPT-4o.

    Args:
        text: Extracted plaintext of the contract (max ~10000 chars used)
        file_name: Original filename
        oai_client: An async OpenAI client instance

    Returns:
        ContractRiskReview dict (see _CONTRACT_REVIEW_SYSTEM for schema)
    """
    excerpt = text[:10000]
    try:
        resp = await oai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _CONTRACT_REVIEW_SYSTEM},
                {"role": "user", "content": f"اسم الملف: {file_name}\n\nنص العقد:\n{excerpt}"},
            ],
            max_tokens=3500,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        # Ensure critical keys exist
        result.setdefault("document_type", "unknown")
        result.setdefault("summary", "")
        result.setdefault("extracted_facts", {})
        result.setdefault("key_clauses_found", [])
        result.setdefault("missing_or_weak_clauses", [])
        result.setdefault("party_risk_balance", {"favors": "unclear", "notes": ""})
        result.setdefault("practical_recommendations", [])
        result.setdefault("questions_for_lawyer", [])
        result.setdefault("risk_score", {"level": "unknown", "score": 0, "reasons": []})
        result.setdefault("confidence", {"level": "unknown", "reason": None})
        if not result.get("disclaimer"):
            result["disclaimer"] = (
                "هذه المراجعة أولية وتعتمد على الذكاء الاصطناعي. "
                "لا تُعدّ استشارة قانونية ولا تُغني عن مراجعة محامٍ متخصص."
            )
        return result
    except Exception as exc:
        logger.error("Contract review failed for %s: %s", file_name, exc)
        return {
            "document_type": "unknown",
            "summary": "تعذّر مراجعة العقد تلقائياً",
            "extracted_facts": {},
            "key_clauses_found": [],
            "missing_or_weak_clauses": [],
            "party_risk_balance": {"favors": "unclear", "notes": ""},
            "practical_recommendations": [],
            "questions_for_lawyer": [],
            "risk_score": {"level": "unknown", "score": 0, "reasons": []},
            "confidence": {"level": "unknown", "reason": str(exc)},
            "disclaimer": (
                "هذه المراجعة أولية — فشل التحليل التلقائي. "
                "يُرجى مراجعة محامٍ متخصص."
            ),
        }
