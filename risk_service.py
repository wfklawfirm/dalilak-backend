"""
Dalilak AI — Risk Scoring Engine (Phase 6)

Pure rule-based scoring — no AI calls, deterministic, fast.
Returns a risk score 0-100 with level (low/medium/high/critical) and Arabic reasons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class _RiskFactor:
    key: str
    contribution: int   # 0-100 added to total score
    reason_ar: str


_FACTORS: dict[str, _RiskFactor] = {
    "low_confidence": _RiskFactor(
        "low_confidence", 20,
        "ثقة الاسترجاع منخفضة — المعلومات قد لا تكون شاملة",
    ),
    "no_sources": _RiskFactor(
        "no_sources", 15,
        "لا توجد مصادر رسمية موثّقة لهذه الإجابة",
    ),
    "missing_documents": _RiskFactor(
        "missing_documents", 25,
        "توجد مستندات ناقصة مطلوبة للمعاملة",
    ),
    "multiple_missing_critical": _RiskFactor(
        "multiple_missing_critical", 35,
        "أكثر من وثيقتين مفقودتين بشكل حرج",
    ),
    "missing_fees": _RiskFactor(
        "missing_fees", 8,
        "الرسوم غير موثّقة — قد تختلف عن المبلغ الفعلي",
    ),
    "unverified_fees": _RiskFactor(
        "unverified_fees", 5,
        "بعض الرسوم المذكورة غير موثّقة رسمياً",
    ),
    "unclear_authority": _RiskFactor(
        "unclear_authority", 20,
        "الجهة المختصة غير محددة أو متعددة",
    ),
    "user_abroad": _RiskFactor(
        "user_abroad", 15,
        "المستخدم خارج لبنان — قد تتطلب توكيلاً رسمياً أو إجراءات إضافية",
    ),
    "property_transaction": _RiskFactor(
        "property_transaction", 20,
        "المعاملة تتعلق بعقار — ينصح بمراجعة محامٍ",
    ),
    "company_transaction": _RiskFactor(
        "company_transaction", 15,
        "المعاملة تتعلق بشركة — ينصح بمراجعة خبير قانوني",
    ),
    "high_value": _RiskFactor(
        "high_value", 20,
        "معاملة ذات قيمة مالية مرتفعة",
    ),
    "inheritance": _RiskFactor(
        "inheritance", 25,
        "معاملة ارث — غالباً تتطلب محامياً ووثائق قضائية",
    ),
    "power_of_attorney": _RiskFactor(
        "power_of_attorney", 15,
        "وكالة قانونية — تحتاج تحقق دقيق من صلاحيات الوكيل",
    ),
    "contract_missing_clauses": _RiskFactor(
        "contract_missing_clauses", 30,
        "العقد يفتقر لبنود جوهرية",
    ),
}

# Procedure slug sets for domain-specific risk bumps
_PROPERTY_SLUGS   = frozenset({"property-transfer", "property-sale", "real-estate-statement"})
_COMPANY_SLUGS    = frozenset({"company-registration", "commercial-registry", "ngo-registration"})
_INHERITANCE_SLUGS = frozenset({"inheritance-certificate"})
_POA_SLUGS        = frozenset({"power-of-attorney"})
_HIGH_VALUE_SLUGS = _PROPERTY_SLUGS | _COMPANY_SLUGS | _INHERITANCE_SLUGS

_ABROAD_KEYWORDS = frozenset({"abroad", "خارج", "outside", "سفارة", "embassy", "diaspora"})


def compute_risk(
    *,
    confidence_level: str = "unknown",
    missing_documents_count: int = 0,
    has_sources: bool = True,
    has_fees: bool = True,
    has_authority: bool = True,
    fees_verified: bool = True,
    procedure_slug: Optional[str] = None,
    user_answers: Optional[dict] = None,
    contract_missing_clauses_count: int = 0,
) -> dict:
    """
    Compute a risk profile for a transaction or document.

    Returns:
        {
            level: "low" | "medium" | "high" | "critical",
            score: int (0-100),
            reasons: list[str]  (Arabic, deduplicated),
            recommendedAction: "continue" | "verify" | "lawyer_review" | "human_support",
            factors: list[str]  (factor keys)
        }
    """
    active: list[_RiskFactor] = []
    score = 0

    def _add(key: str) -> None:
        nonlocal score
        f = _FACTORS[key]
        active.append(f)
        score += f.contribution

    # ── Confidence ────────────────────────────────────────────
    if confidence_level in ("low", "unknown"):
        _add("low_confidence")

    # ── Sources ───────────────────────────────────────────────
    if not has_sources:
        _add("no_sources")

    # ── Missing documents ──────────────────────────────────────
    if missing_documents_count >= 3:
        _add("multiple_missing_critical")
    elif missing_documents_count > 0:
        _add("missing_documents")

    # ── Fees ──────────────────────────────────────────────────
    if not has_fees:
        _add("missing_fees")
    elif not fees_verified:
        _add("unverified_fees")

    # ── Authority ──────────────────────────────────────────────
    if not has_authority:
        _add("unclear_authority")

    # ── Procedure-specific ─────────────────────────────────────
    slug = (procedure_slug or "").lower()
    if slug in _HIGH_VALUE_SLUGS:
        _add("high_value")
    if slug in _PROPERTY_SLUGS:
        _add("property_transaction")
    if slug in _COMPANY_SLUGS:
        _add("company_transaction")
    if slug in _INHERITANCE_SLUGS:
        _add("inheritance")
    if slug in _POA_SLUGS:
        _add("power_of_attorney")

    # ── User answers (GuidedFlow context) ─────────────────────
    if user_answers:
        loc = str(user_answers.get("location") or "").lower()
        if any(kw in loc for kw in _ABROAD_KEYWORDS):
            _add("user_abroad")

    # ── Contract clauses ───────────────────────────────────────
    if contract_missing_clauses_count >= 3:
        _add("contract_missing_clauses")

    # ── Clamp and level ────────────────────────────────────────
    score = min(100, score)

    if score >= 65:
        level = "critical"
        action = "lawyer_review"
    elif score >= 40:
        level = "high"
        action = "lawyer_review"
    elif score >= 20:
        level = "medium"
        action = "verify"
    else:
        level = "low"
        action = "continue"

    # Deduplicate reasons while preserving order
    seen: set[str] = set()
    reasons: list[str] = []
    for f in active:
        if f.reason_ar not in seen:
            seen.add(f.reason_ar)
            reasons.append(f.reason_ar)

    return {
        "level": level,
        "score": score,
        "reasons": reasons,
        "recommendedAction": action,
        "factors": [f.key for f in active],
    }
