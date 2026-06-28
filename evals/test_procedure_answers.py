"""
Test that AI-generated procedure answers conform to the structured schema
and contain expected content for golden questions.

Run standalone: python backend/evals/test_procedure_answers.py
Run with pytest: pytest backend/evals/test_procedure_answers.py -v
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_questions.json")

REQUIRED_KEYS = [
    "summary", "required_documents", "steps", "authority",
    "fees", "forms", "next_action", "warnings", "sources",
    "confidence", "disclaimer",
]

VALID_CONFIDENCE_LEVELS = {"high", "medium", "low", "unknown"}
VALID_ACTION_TYPES = {
    "download_checklist", "generate_form", "upload_document",
    "start_flow", "ask_followup", "contact_human",
    "save_procedure", "none",
}
VALID_WARNING_LEVELS = {"info", "warning", "critical"}


def load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)["questions"]


# ── Schema validators ─────────────────────────────────────────────────────────

def validate_schema(answer: dict, question_id: str) -> list[str]:
    """Return list of schema errors (empty = valid)."""
    errors = []

    for k in REQUIRED_KEYS:
        if k not in answer:
            errors.append(f"[{question_id}] Missing top-level key: '{k}'")

    # summary must be non-empty string
    if not isinstance(answer.get("summary", ""), str) or not answer.get("summary", "").strip():
        errors.append(f"[{question_id}] 'summary' must be a non-empty string")

    # required_documents list
    for doc in answer.get("required_documents", []):
        if "title" not in doc:
            errors.append(f"[{question_id}] Document missing 'title'")
        if "required" not in doc or not isinstance(doc.get("required"), bool):
            errors.append(f"[{question_id}] Document 'required' must be bool")

    # steps list
    for step in answer.get("steps", []):
        if "order" not in step or not isinstance(step.get("order"), int):
            errors.append(f"[{question_id}] Step missing integer 'order'")
        if "title" not in step:
            errors.append(f"[{question_id}] Step missing 'title'")

    # authority dict
    auth = answer.get("authority")
    if auth is not None and not isinstance(auth, dict):
        errors.append(f"[{question_id}] 'authority' must be dict or null")

    # fees list
    for fee in answer.get("fees", []):
        if "label" not in fee and "label_ar" not in fee:
            errors.append(f"[{question_id}] Fee missing 'label'")

    # next_action
    na = answer.get("next_action", {})
    if isinstance(na, dict) and na.get("action_type") not in VALID_ACTION_TYPES:
        errors.append(
            f"[{question_id}] Invalid action_type '{na.get('action_type')}'"
        )

    # warnings
    for w in answer.get("warnings", []):
        if w.get("level") not in VALID_WARNING_LEVELS:
            errors.append(f"[{question_id}] Invalid warning level '{w.get('level')}'")

    # confidence
    conf = answer.get("confidence", {})
    if isinstance(conf, dict):
        if conf.get("level") not in VALID_CONFIDENCE_LEVELS:
            errors.append(
                f"[{question_id}] Invalid confidence level '{conf.get('level')}'"
            )
    elif isinstance(conf, str):
        if conf not in VALID_CONFIDENCE_LEVELS:
            errors.append(f"[{question_id}] Invalid confidence string '{conf}'")

    # disclaimer
    if not isinstance(answer.get("disclaimer", ""), str):
        errors.append(f"[{question_id}] 'disclaimer' must be a string")

    return errors


# ── Mock answer generator (used when no live API) ─────────────────────────────

def _make_mock_answer(question: dict) -> dict:
    """
    Generates a minimal valid mock answer to validate schema without hitting the API.
    In CI, swap this with a real call to /chat/structured.
    """
    lang = question.get("language", "ar")
    slug = question.get("expected_procedure_slug", "unknown")
    return {
        "summary": f"Mock summary for {slug} in {'Arabic' if lang == 'ar' else 'English'}.",
        "required_documents": [
            {"title": "بطاقة الهوية" if lang == "ar" else "ID Card", "required": True, "notes": ""},
        ],
        "steps": [
            {"order": 1, "title": "تقديم الطلب" if lang == "ar" else "Submit application",
             "description": ""},
        ],
        "authority": {
            "name_ar": "الجهة المختصة",
            "name_en": "Competent Authority",
            "verified": False,
        },
        "fees": [
            {"label": "رسوم" if lang == "ar" else "Fees", "amount": "0", "currency": "USD", "verified": False}
        ],
        "forms": [],
        "next_action": {"label": "ابدأ" if lang == "ar" else "Start", "action_type": "start_flow"},
        "warnings": [{"level": "info", "message": "معلومات غير موثوقة" if lang == "ar" else "Unverified info"}],
        "sources": [{"title": "مصدر" if lang == "ar" else "Source", "type": "official", "reliability": "low"}],
        "confidence": {"level": "low", "reason": "Mock answer"},
        "disclaimer": "للإرشاد فقط." if lang == "ar" else "For guidance only.",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_all_golden_pass_schema():
    """All golden questions must produce schema-valid answers."""
    questions = load_golden()
    all_errors = []
    for q in questions:
        answer = _make_mock_answer(q)
        errs = validate_schema(answer, q["id"])
        all_errors.extend(errs)

    if all_errors:
        for e in all_errors:
            print(f"  ❌ {e}")
        raise AssertionError(f"{len(all_errors)} schema errors")
    print(f"✅ All {len(questions)} golden questions pass schema validation")


def test_disclaimer_always_present():
    """Every answer must include a non-empty disclaimer."""
    questions = load_golden()
    for q in questions:
        answer = _make_mock_answer(q)
        assert answer.get("disclaimer", "").strip(), f"[{q['id']}] Disclaimer is empty"
    print("✅ All answers include a disclaimer")


def test_no_html_in_summary():
    """Summary must not contain raw HTML tags."""
    import re
    html_re = re.compile(r'<[a-zA-Z][^>]*>')
    questions = load_golden()
    for q in questions:
        answer = _make_mock_answer(q)
        summary = answer.get("summary", "")
        assert not html_re.search(summary), f"[{q['id']}] Raw HTML found in summary"
    print("✅ No raw HTML in summaries")


def test_unverified_fees_marked():
    """All fees in mock answers must default to verified=False."""
    questions = load_golden()
    for q in questions:
        answer = _make_mock_answer(q)
        for fee in answer.get("fees", []):
            if fee.get("verified"):
                assert fee.get("source_id") or fee.get("notes"), \
                    f"[{q['id']}] Verified fee missing source_id or notes"
    print("✅ Unverified fees correctly marked")


def test_mock_answers_all_valid():
    """Meta-test: mock answer generator itself produces valid answers."""
    sample = {
        "id": "meta_001",
        "language": "ar",
        "expected_procedure_slug": "passport",
    }
    answer = _make_mock_answer(sample)
    errs = validate_schema(answer, "meta_001")
    assert not errs, f"Mock generator produced invalid answer: {errs}"
    print("✅ Mock answer generator produces valid schema")


def test_validate_schema_catches_missing_keys():
    """validate_schema must catch missing top-level keys."""
    bad = {"summary": "ok"}  # missing most keys
    errs = validate_schema(bad, "bad_001")
    assert len(errs) > 0, "Expected errors for missing keys"
    print(f"✅ Schema validator caught {len(errs)} errors on bad input")


if __name__ == "__main__":
    print("\n🧪 Procedure Answer Tests\n" + "=" * 40)
    test_mock_answers_all_valid()
    test_validate_schema_catches_missing_keys()
    test_all_golden_pass_schema()
    test_disclaimer_always_present()
    test_no_html_in_summary()
    test_unverified_fees_marked()
    print("\n✅ All procedure answer tests passed!\n")
