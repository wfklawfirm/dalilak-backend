"""
Test retrieval quality for Dalilak AI.
Run standalone: python backend/evals/test_retrieval.py
Run with pytest: pytest backend/evals/test_retrieval.py -v
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_questions.json")


def load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)["questions"]


def test_golden_questions_loaded():
    qs = load_golden()
    assert len(qs) > 0
    print(f"✅ Loaded {len(qs)} golden questions")


def test_golden_schema():
    required = ["id", "category", "language", "country", "question", "expected_procedure_slug"]
    for q in load_golden():
        for field in required:
            assert field in q, f"Missing '{field}' in {q.get('id')}"
    print("✅ All golden questions have required fields")


def test_retrieval_service_import():
    from retrieval_service import RetrievalService
    print("✅ RetrievalService imported")


def test_language_detection():
    from retrieval_service import _detect_language
    assert _detect_language("كيف أستخرج جواز سفر لبناني؟") == "ar"
    assert _detect_language("How do I get a Lebanese passport?") == "en"
    print("✅ Language detection works")


def test_country_detection():
    from retrieval_service import _detect_country
    assert _detect_country("معاملة في لبنان") == "lebanon"
    assert _detect_country("procedure in lebanon") == "lebanon"
    assert _detect_country("something in syria") == "syria"
    assert _detect_country("random text") is None
    print("✅ Country detection works")


def test_procedure_detection():
    from retrieval_service import _detect_procedure
    assert _detect_procedure("استخراج جواز سفر") == "passport"
    assert _detect_procedure("إخراج قيد فردي") == "civil-registry-extract"
    assert _detect_procedure("تأسيس شركة SARL") == "company-registration"
    assert _detect_procedure("حصر الإرث") == "inheritance-certificate"
    print("✅ Procedure detection works")


def test_normalize_arabic():
    from retrieval_service import _normalize_arabic
    assert _normalize_arabic("إخراج") == "اخراج"
    assert _normalize_arabic("أحمد") == "احمد"
    assert _normalize_arabic("آل") == "ال"
    print("✅ Arabic normalization works")


def test_confidence_scoring():
    from retrieval_service import RetrievalService

    class MockSvc(RetrievalService):
        def __init__(self): pass

    svc = MockSvc()
    assert svc.calculate_confidence([]) == "low"
    assert svc.calculate_confidence([{"score": 0.55}, {"score": 0.45}]) == "high"
    assert svc.calculate_confidence([{"score": 0.38}]) == "medium"
    assert svc.calculate_confidence([{"score": 0.20}]) == "low"
    print("✅ Confidence scoring thresholds correct")


def test_source_extraction():
    from retrieval_service import RetrievalService

    class MockSvc(RetrievalService):
        def __init__(self): pass

    svc = MockSvc()
    chunks = [
        {"title": "الأمن العام", "ministry": "", "score": 0.6, "country": "lebanon", "url": "", "last_reviewed": ""},
        {"title": "الأمن العام", "ministry": "", "score": 0.5, "country": "lebanon", "url": "", "last_reviewed": ""},
        {"title": "وزارة الداخلية", "ministry": "", "score": 0.4, "country": "lebanon", "url": "", "last_reviewed": ""},
    ]
    sources = svc.extract_sources(chunks)
    assert len(sources) == 2, f"Expected 2 unique sources, got {len(sources)}"
    print("✅ Source extraction deduplicates correctly")


if __name__ == "__main__":
    print("\n🧪 Dalilak Retrieval Tests\n" + "=" * 40)
    test_golden_questions_loaded()
    test_golden_schema()
    test_retrieval_service_import()
    test_language_detection()
    test_country_detection()
    test_procedure_detection()
    test_normalize_arabic()
    test_confidence_scoring()
    test_source_extraction()
    print("\n✅ All tests passed!\n")
