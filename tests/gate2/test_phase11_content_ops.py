"""
Phase 11 -- Content Operations -- Tier A (offline, source-inspection) tests
8 tests, zero network calls, zero env vars required.
"""
import json
import pathlib
import re

BACKEND     = pathlib.Path(__file__).parent.parent.parent
SRC_PATH    = BACKEND / "main.py"
GOLDEN_PATH = BACKEND / "tests" / "gate2" / "golden_v1.json"


def _src() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


# -- Golden dataset structure -------------------------------------------------

def test_p11_a01_golden_dataset_exists():
    assert GOLDEN_PATH.exists(), "golden_v1.json not found"


def test_p11_a02_golden_has_minimum_170_entries():
    d = _golden()
    items = d.get("items", [])
    assert len(items) >= 170, f"Expected >= 170 items, got {len(items)}"


def test_p11_a03_golden_has_out_of_scope_entries():
    """Dataset must contain negative examples (should_pass_gate=False)."""
    d = _golden()
    no_pass = [i for i in d["items"] if i.get("should_pass_gate") is False]
    assert len(no_pass) >= 20, (
        f"Expected >= 20 should_pass_gate=False entries, got {len(no_pass)}"
    )


def test_p11_a04_golden_has_cross_jurisdiction_entries():
    """At least 3 entries flagged as out-of-jurisdiction."""
    d = _golden()
    cross = [
        i for i in d["items"]
        if "OUT_OF_SCOPE" in i.get("category", "") and "OTHER" in i.get("category", "").upper()
        or i.get("category", "").endswith("_OTHER")
        or "SCOPE" in i.get("category", "").upper() and "JURISDICTION" in i.get("category", "").upper()
    ]
    # Accept either English or Arabic category naming convention
    cross_ar = [
        i for i in d["items"]
        if "دولة" in i.get("category", "")  # arabic "dolah" (country)
    ]
    total = len(cross) + len(cross_ar)
    assert total >= 3, (
        f"Expected >= 3 cross-jurisdiction entries, got {total}"
    )


def test_p11_a05_golden_has_prompt_injection_entries():
    """Dataset must contain prompt-injection attempts as negative examples."""
    d = _golden()
    injection = [
        i for i in d["items"]
        if (
            "حقن" in i.get("category", "")   # "haqn" (injection in Arabic)
            or "injection" in i.get("category", "").lower()
            or "prompt" in i.get("category", "").lower()
        )
    ]
    assert len(injection) >= 2, (
        f"Expected >= 2 prompt-injection entries, got {len(injection)}"
    )


def test_p11_a06_every_item_has_required_schema_keys():
    """All items must have id, query, should_pass_gate, category."""
    d = _golden()
    required = {"id", "query", "should_pass_gate", "category"}
    for item in d["items"]:
        missing = required - set(item.keys())
        assert not missing, f"Item {item.get('id', '?')} missing keys: {missing}"


def test_p11_a07_no_duplicate_ids():
    d = _golden()
    ids = [i["id"] for i in d["items"]]
    assert len(ids) == len(set(ids)), "Duplicate IDs found in golden dataset"


# -- Evidence gate constants in source ----------------------------------------

def test_p11_a08_evidence_sufficiency_constants_in_source():
    """SUFFICIENCY_MSG and SUFFICIENCY_TOP_SCORE must exist in main.py."""
    src = _src()
    assert "SUFFICIENCY_MSG" in src, "SUFFICIENCY_MSG constant not found in main.py"
    assert "SUFFICIENCY_TOP_SCORE" in src, "SUFFICIENCY_TOP_SCORE threshold missing"
    # Verify the threshold is a valid float between 0 and 1
    m = re.search(r"SUFFICIENCY_TOP_SCORE\s*=\s*([0-9.]+)", src)
    assert m, "SUFFICIENCY_TOP_SCORE not assigned a numeric value"
    val = float(m.group(1))
    assert 0.0 < val < 1.0, f"SUFFICIENCY_TOP_SCORE must be in (0,1), got {val}"
