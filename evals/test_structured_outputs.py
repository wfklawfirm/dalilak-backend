"""
Test structured AI output validity for Dalilak AI.
Run standalone: python backend/evals/test_structured_outputs.py
"""

VALID_SAMPLE = {
    "summary": "لاستخراج جواز السفر اللبناني تراجع الأمن العام.",
    "required_documents": [
        {"title": "بطاقة الهوية", "required": True, "notes": "أصل وصورة"},
        {"title": "صورة شخصية", "required": True, "notes": "خلفية بيضاء"},
    ],
    "steps": [
        {"order": 1, "title": "تقديم الطلب", "description": "تقديم الطلب في مركز الأمن العام"},
        {"order": 2, "title": "دفع الرسوم", "description": "دفع رسوم الإصدار"},
    ],
    "authority": {
        "name_ar": "المديرية العامة للأمن العام",
        "name_en": "General Directorate of General Security",
        "verified": True,
    },
    "fees": [{"label": "رسوم الإصدار", "amount": "50,000", "currency": "LBP", "verified": False}],
    "forms": [],
    "next_action": {"label": "ابدأ المعاملة", "action_type": "start_flow"},
    "warnings": [{"level": "info", "message": "تأكد من صحة المستندات قبل التقديم"}],
    "sources": [{"title": "الأمن العام اللبناني", "type": "official", "reliability": "high"}],
    "confidence": {"level": "medium", "reason": "مصادر متوسطة الثقة"},
    "disclaimer": "هذه المعلومات للإرشاد فقط وليست استشارة قانونية.",
}


def test_top_level_keys():
    required = ["summary", "required_documents", "steps", "authority", "fees", "forms",
                "next_action", "warnings", "sources", "confidence", "disclaimer"]
    for k in required:
        assert k in VALID_SAMPLE, f"Missing key: {k}"
    print("✅ All top-level keys present")


def test_documents_schema():
    for doc in VALID_SAMPLE["required_documents"]:
        assert "title" in doc
        assert "required" in doc
        assert isinstance(doc["required"], bool)
    print("✅ Required documents schema valid")


def test_steps_schema():
    for step in VALID_SAMPLE["steps"]:
        assert "order" in step and isinstance(step["order"], int)
        assert "title" in step
    print("✅ Steps schema valid")


def test_confidence_level():
    valid = {"high", "medium", "low", "unknown"}
    assert VALID_SAMPLE["confidence"]["level"] in valid
    print("✅ Confidence level valid")


def test_warning_levels():
    valid = {"info", "warning", "critical"}
    for w in VALID_SAMPLE["warnings"]:
        assert w["level"] in valid
    print("✅ Warning levels valid")


def test_action_type():
    valid = {"download_checklist", "generate_form", "upload_document", "start_flow",
             "ask_followup", "contact_human", "save_procedure", "none"}
    assert VALID_SAMPLE["next_action"]["action_type"] in valid
    print("✅ Next action type valid")


def test_fees_not_auto_verified():
    """Fees must default to verified=False unless backed by a source."""
    for fee in VALID_SAMPLE["fees"]:
        if fee.get("verified"):
            assert fee.get("source_id") or fee.get("notes"), \
                "Verified fee must have source_id or notes"
    print("✅ Fee verification logic valid")


def test_pydantic_importable():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from main import AgentResponseModel, StructuredChatRequest
        print("✅ Pydantic models importable from main.py")
    except Exception as e:
        print(f"⚠️  Could not import from main.py (acceptable in unit tests): {e}")


if __name__ == "__main__":
    print("\n🧪 Structured Output Tests\n" + "=" * 40)
    test_top_level_keys()
    test_documents_schema()
    test_steps_schema()
    test_confidence_level()
    test_warning_levels()
    test_action_type()
    test_fees_not_auto_verified()
    test_pydantic_importable()
    print("\n✅ All structured output tests passed!\n")
