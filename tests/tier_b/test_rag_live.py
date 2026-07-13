"""
Phase 6 — RAG live evaluation (Tier B: requires DALILAK_TEST_TOKEN + network).

SECURITY: Never use real user credentials in this file.
          Set DALILAK_TEST_TOKEN to a dedicated test-account JWT issued by the owner.
          This file is EXCLUDED from CI by default (only tier_a runs in CI).

Run manually:
    DALILAK_TEST_TOKEN=<jwt> pytest tests/tier_b/ -v -m tier_b --timeout=60

Metrics produced per run:
    gate_pass_rate   — fraction of positive queries that cleared the 0.35 gate
    gate_block_rate  — fraction of negative queries correctly blocked
    mean_top_score   — average top-chunk score for positive queries that passed
"""
from __future__ import annotations

import json, os, statistics, time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.tier_b

_BASE = os.environ.get("DALILAK_API_URL", "https://dalilak-backend-bvb9.onrender.com")
_TOKEN = os.environ.get("DALILAK_TEST_TOKEN", "")
_DATASET_PATH = Path(__file__).parent.parent / "datasets" / "golden_v1.json"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def items():
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["items"]

@pytest.fixture(scope="module")
def auth_headers():
    if not _TOKEN:
        pytest.skip("DALILAK_TEST_TOKEN not set — skipping Tier B live tests")
    return {"Authorization": f"Bearer {_TOKEN}"}

# ── Helper ────────────────────────────────────────────────────────────────────

def _chat(query: str, domain: str | None, headers: dict) -> dict:
    """POST /chat and return the parsed response."""
    payload = {"message": query, "domain": domain, "history": []}
    r = httpx.post(f"{_BASE}/chat", json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.tier_b
def test_p6_b1_api_reachable(auth_headers):
    r = httpx.get(f"{_BASE}/", headers=auth_headers, timeout=15)
    assert r.status_code == 200

@pytest.mark.tier_b
def test_p6_b2_positive_gate_pass_rate(items, auth_headers):
    """>=80% of positive queries must clear the evidence gate (model!='gate')."""
    positives = [i for i in items if i["should_pass_gate"]][:30]  # sample first 30
    passed = 0
    for item in positives:
        resp = _chat(item["query"], item.get("domain"), auth_headers)
        if resp.get("model") != "gate":
            passed += 1
        time.sleep(0.5)  # gentle throttle
    rate = passed / len(positives)
    assert rate >= 0.80, f"Positive gate pass rate {rate:.0%} < 80% threshold"

@pytest.mark.tier_b
def test_p6_b3_negative_gate_block_rate(items, auth_headers):
    """>=90% of OUT_OF_SCOPE queries must be blocked by the gate (model=='gate')."""
    negatives = [i for i in items if not i["should_pass_gate"]]
    blocked = 0
    for item in negatives:
        resp = _chat(item["query"], item.get("domain"), auth_headers)
        if resp.get("model") == "gate":
            blocked += 1
        time.sleep(0.5)
    rate = blocked / len(negatives)
    assert rate >= 0.90, f"Negative gate block rate {rate:.0%} < 90% threshold"

@pytest.mark.tier_b
def test_p6_b4_mean_top_score_positive(items, auth_headers):
    """Mean top-chunk score for passing positive queries should be >= 0.40."""
    positives = [i for i in items if i["should_pass_gate"]][:20]
    scores = []
    for item in positives:
        resp = _chat(item["query"], item.get("domain"), auth_headers)
        sources = resp.get("sources", [])
        if sources:
            scores.append(sources[0].get("score", 0))
        time.sleep(0.5)
    if scores:
        mean = statistics.mean(scores)
        assert mean >= 0.40, f"Mean top score {mean:.3f} < 0.40 threshold"

@pytest.mark.tier_b
def test_p6_b5_no_secrets_in_response(items, auth_headers):
    """Responses must never contain JWT secrets or API key patterns."""
    import re
    sample = items[:5]
    key_pattern = re.compile(r"(?:sk-|re_)[A-Za-z0-9]{15,}")
    for item in sample:
        resp = _chat(item["query"], item.get("domain"), auth_headers)
        answer = resp.get("answer", "")
        assert not key_pattern.search(answer), \
            f"Possible secret key found in answer for query: {item['id']}"
        time.sleep(0.5)
