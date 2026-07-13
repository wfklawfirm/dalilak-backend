"""Phase 6 — Golden dataset schema & coverage tests (Tier A: offline, no network)."""
from __future__ import annotations
import json, os
import pytest

_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_v1.json")

@pytest.fixture(scope="module")
def dataset():
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def items(dataset):
    return dataset["items"]

@pytest.mark.tier_a
def test_p6_a1_file_exists():
    assert os.path.exists(_DATASET_PATH)

@pytest.mark.tier_a
def test_p6_a2_version_field(dataset):
    assert dataset.get("version", 0) >= 1

@pytest.mark.tier_a
def test_p6_a3_minimum_item_count(items):
    assert len(items) >= 150

@pytest.mark.tier_a
def test_p6_a4_required_fields(items):
    required = {"id", "query", "should_pass_gate", "expected_keywords", "category"}
    for item in items:
        assert not (required - set(item.keys())), f"Item {item.get('id')} missing fields"

@pytest.mark.tier_a
def test_p6_a5_unique_ids(items):
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))

@pytest.mark.tier_a
def test_p6_a6_queries_non_empty(items):
    for item in items:
        assert item["query"].strip(), f"Empty query: {item['id']}"

@pytest.mark.tier_a
def test_p6_a7_bool_gate_field(items):
    for item in items:
        assert isinstance(item["should_pass_gate"], bool), f"{item['id']} gate not bool"

@pytest.mark.tier_a
def test_p6_a8_keywords_are_list(items):
    for item in items:
        assert isinstance(item["expected_keywords"], list), f"{item['id']} keywords not list"

@pytest.mark.tier_a
def test_p6_a9_positive_items_majority(items):
    assert sum(1 for i in items if i["should_pass_gate"]) >= 100

@pytest.mark.tier_a
def test_p6_a10_negative_items_present(items):
    assert sum(1 for i in items if not i["should_pass_gate"]) >= 10

@pytest.mark.tier_a
def test_p6_a11_category_diversity(items):
    assert len({i["category"] for i in items}) >= 10

@pytest.mark.tier_a
def test_p6_a12_positive_items_have_keywords(items):
    for item in items:
        if item["should_pass_gate"]:
            assert len(item["expected_keywords"]) >= 1, f"Positive {item['id']} has no keywords"

@pytest.mark.tier_a
def test_p6_a13_arabic_queries(items):
    def has_arabic(s): return any(0x0600 <= ord(c) <= 0x06FF for c in s)
    pos = [i for i in items if i["should_pass_gate"]]
    arabic = sum(1 for i in pos if has_arabic(i["query"]))
    assert arabic / len(pos) >= 0.8

@pytest.mark.tier_a
def test_p6_a14_no_duplicate_queries(items):
    queries = [i["query"].strip() for i in items]
    assert len(queries) == len(set(queries))

@pytest.mark.tier_a
def test_p6_a15_personal_status_coverage(items):
    assert "أحوال_شخصية" in {i["category"] for i in items}

@pytest.mark.tier_a
def test_p6_a16_legal_coverage(items):
    legal = {i["category"] for i in items if "قضاء" in i["category"] or "قانون" in i["category"]}
    assert len(legal) >= 3

@pytest.mark.tier_a
def test_p6_a17_labor_coverage(items):
    assert "عمل" in {i["category"] for i in items}

@pytest.mark.tier_a
def test_p6_a18_banking_coverage(items):
    assert "مصارف" in {i["category"] for i in items}

@pytest.mark.tier_a
def test_p6_a19_cnss_coverage(items):
    cnss = [i for i in items if "CNSS" in " ".join(i["expected_keywords"]) or "ضمان" in i["query"]]
    assert len(cnss) >= 3

@pytest.mark.tier_a
def test_p6_a20_out_of_scope_marked_false(items):
    for item in items:
        if "OUT_OF_SCOPE" in item["category"]:
            assert item["should_pass_gate"] is False, f"{item['id']} out-of-scope but gate=True"
