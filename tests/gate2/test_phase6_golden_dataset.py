"""Phase 6 — Golden dataset schema & coverage tests (Tier A: offline, no network)."""
from __future__ import annotations
import json, os
import pytest

_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "golden_v1.json"
)

@pytest.fixture(scope="module")
def dataset():
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def items(dataset):
    return dataset["items"]

# ── Schema tests ───────────────────────────────────────────────────────────────

@pytest.mark.tier_a
def test_p6_a1_file_exists():
    assert os.path.exists(_DATASET_PATH), f"Dataset not found: {_DATASET_PATH}"

@pytest.mark.tier_a
def test_p6_a2_version_field(dataset):
    assert "version" in dataset
    assert dataset["version"] >= 1

@pytest.mark.tier_a
def test_p6_a3_minimum_item_count(items):
    assert len(items) >= 150, f"Expected >=150 items, got {len(items)}"

@pytest.mark.tier_a
def test_p6_a4_required_fields(items):
    required = {"id", "query", "should_pass_gate", "expected_keywords", "category"}
    for item in items:
        missing = required - set(item.keys())
        assert not missing, f"Item {item.get('id')} missing fields: {missing}"

@pytest.mark.tier_a
def test_p6_a5_unique_ids(items):
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "Duplicate IDs in dataset"

@pytest.mark.tier_a
def test_p6_a6_queries_non_empty(items):
    for item in items:
        assert item["query"].strip(), f"Item {item['id']} has empty query"

@pytest.mark.tier_a
def test_p6_a7_bool_gate_field(items):
    for item in items:
        assert isinstance(item["should_pass_gate"], bool), \
            f"Item {item['id']} should_pass_gate is not bool"

@pytest.mark.tier_a
def test_p6_a8_keywords_are_list(items):
    for item in items:
        assert isinstance(item["expected_keywords"], list), \
            f"Item {item['id']} expected_keywords is not a list"

# ── Coverage tests ─────────────────────────────────────────────────────────────

@pytest.mark.tier_a
def test_p6_a9_positive_items_majority(items):
    positive = sum(1 for i in items if i["should_pass_gate"])
    assert positive >= 100, f"Expected >=100 positive items, got {positive}"

@pytest.mark.tier_a
def test_p6_a10_negative_items_present(items):
    negative = sum(1 for i in items if not i["should_pass_gate"])
    assert negative >= 10, f"Expected >=10 negative (out-of-scope) items, got {negative}"

@pytest.mark.tier_a
def test_p6_a11_category_diversity(items):
    categories = {i["category"] for i in items}
    assert len(categories) >= 10, f"Expected >=10 categories, got {len(categories)}"

@pytest.mark.tier_a
def test_p6_a12_positive_items_have_keywords(items):
    for item in items:
        if item["should_pass_gate"]:
            assert len(item["expected_keywords"]) >= 1, \
                f"Positive item {item['id']} has no expected_keywords"

@pytest.mark.tier_a
def test_p6_a13_arabic_queries(items):
    # At least 80% of positive queries should contain Arabic characters
    arabic_range = range(0x0600, 0x06FF)
    def has_arabic(s):
        return any(ord(c) in arabic_range for c in s)
    pos = [i for i in items if i["should_pass_gate"]]
    arabic_count = sum(1 for i in pos if has_arabic(i["query"]))
    assert arabic_count / len(pos) >= 0.8, \
        f"Less than 80% of positive queries are Arabic ({arabic_count}/{len(pos)})"

@pytest.mark.tier_a
def test_p6_a14_no_duplicate_queries(items):
    queries = [i["query"].strip() for i in items]
    assert len(queries) == len(set(queries)), "Duplicate queries found in dataset"

@pytest.mark.tier_a
def test_p6_a15_personal_status_coverage(items):
    cats = {i["category"] for i in items}
    assert "أحوال_شخصية" in cats, "Missing أحوال_شخصية category"

@pytest.mark.tier_a
def test_p6_a16_legal_coverage(items):
    legal_cats = {i["category"] for i in items if "قضاء" in i["category"] or "قانون" in i["category"]}
    assert len(legal_cats) >= 3, f"Insufficient legal categories: {legal_cats}"

@pytest.mark.tier_a
def test_p6_a17_labor_coverage(items):
    cats = {i["category"] for i in items}
    assert "عمل" in cats, "Missing عمل (labor) category"

@pytest.mark.tier_a
def test_p6_a18_banking_coverage(items):
    cats = {i["category"] for i in items}
    assert "مصارف" in cats, "Missing مصارف (banking) category"

@pytest.mark.tier_a
def test_p6_a19_cnss_coverage(items):
    cnss = [i for i in items if "CNSS" in " ".join(i["expected_keywords"]) or "ضمان" in i["query"]]
    assert len(cnss) >= 3, "Insufficient CNSS/social-security coverage"

@pytest.mark.tier_a
def test_p6_a20_out_of_scope_categories_marked(items):
    out = [i for i in items if "OUT_OF_SCOPE" in i["category"]]
    assert len(out) >= 10
    for item in out:
        assert item["should_pass_gate"] is False, \
            f"OUT_OF_SCOPE item {item['id']} incorrectly marked should_pass_gate=True"
