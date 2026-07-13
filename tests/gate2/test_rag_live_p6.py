"""
Phase 6 — RAG live evaluation (Tier B: requires DALILAK_TEST_TOKEN + network).

Run manually (Windows):
    set DALILAK_TEST_TOKEN=eyJ...your_jwt...
    python -m pytest tests\gate2\test_rag_live_p6.py -v -m tier_b

This file is SAFE to commit — it does nothing unless DALILAK_TEST_TOKEN is set.
SECURITY: No real credentials in this file. Token must come from env only.
"""
from __future__ import annotations
import json, os, statistics, time
from pathlib import Path
import pytest

_TOKEN = os.environ.get("DALILAK_TEST_TOKEN", "")
_BASE  = os.environ.get("DALILAK_API_URL", "https://dalilak-backend-bvb9.onrender.com")
_DATASET_PATH = Path(__file__).parent / "golden_v1.json"

pytestmark = pytest.mark.tier_b

# Skip entire module if no token
if not _TOKEN:
    pytest.skip("DALILAK_TEST_TOKEN not set — skipping Tier B live tests", allow_module_level=True)

try:
    import httpx as _httpx
except ImportError:
    pytest.skip("httpx not installed — pip install httpx", allow_module_level=True)

@pytest.fixture(scope="module")
def items():
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["items"]

@pytest.fixture(scope="module")
def headers():
    return {"Authorization": f"Bearer {_TOKEN}"}

def _chat(q, domain, hdrs):
    r = _httpx.post(f"{_BASE}/chat", json={"message": q, "domain": domain, "history": []},
                    headers=hdrs, timeout=30)
    r.raise_for_status()
    return r.json()

@pytest.mark.tier_b
def test_p6_b1_api_reachable(headers):
    r = _httpx.get(f"{_BASE}/", headers=headers, timeout=15)
    assert r.status_code == 200

@pytest.mark.tier_b
def test_p6_b2_positive_gate_pass_rate(items, headers):
    pos = [i for i in items if i["should_pass_gate"]][:20]
    passed = sum(1 for i in pos if _chat(i["query"], i.get("domain"), headers).get("model") != "gate")
    assert passed / len(pos) >= 0.80, f"Positive pass rate {passed}/{len(pos)} < 80%"

@pytest.mark.tier_b
def test_p6_b3_negative_gate_block_rate(items, headers):
    neg = [i for i in items if not i["should_pass_gate"]]
    blocked = sum(1 for i in neg if _chat(i["query"], i.get("domain"), headers).get("model") == "gate")
    assert blocked / len(neg) >= 0.90, f"Negative block rate {blocked}/{len(neg)} < 90%"

@pytest.mark.tier_b
def test_p6_b4_no_secrets_in_answers(items, headers):
    import re
    pat = re.compile(r"(?:sk-|re_)[A-Za-z0-9]{15,}")
    for item in items[:5]:
        resp = _chat(item["query"], item.get("domain"), headers)
        assert not pat.search(resp.get("answer", "")), f"Possible secret in answer for {item['id']}"
        time.sleep(0.3)
