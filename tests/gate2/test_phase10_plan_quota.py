"""
Phase 10 -- Server-Side Plan Quota Enforcement -- Tier A (offline, source-inspection) tests
10 tests, zero network calls, zero env vars required.
"""
import importlib.util
import pathlib
import re
import sys

BACKEND    = pathlib.Path(__file__).parent.parent.parent
SRC_PATH   = BACKEND / "main.py"
QUOTA_PATH = BACKEND / "plan_quota.py"


def _src() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


def _quota() -> str:
    return QUOTA_PATH.read_text(encoding="utf-8")


# ── plan_quota.py structure ───────────────────────────────────────────────────

def test_p10_a01_plan_quota_module_exists():
    assert QUOTA_PATH.exists(), "plan_quota.py not found"


def test_p10_a02_plan_daily_limits_dict_exists():
    src = _quota()
    assert "PLAN_DAILY_LIMITS" in src, "PLAN_DAILY_LIMITS dict missing from plan_quota.py"
    # All four plan tiers must be present
    for plan in ("trial", "paid", "admin", "guest"):
        assert f'"{plan}"' in src or f"\'{plan}\'" in src, f"Plan tier \'{plan}\' missing"


def test_p10_a03_trial_limit_less_than_paid():
    src = _quota()
    m_trial = re.search(r'"trial"\s*:\s*(\d+)', src)
    m_paid  = re.search(r'"paid"\s*:\s*(\d+)', src)
    assert m_trial, "trial limit not found as integer"
    assert m_paid,  "paid limit not found as integer"
    assert int(m_trial.group(1)) < int(m_paid.group(1)), (
        f"trial limit ({m_trial.group(1)}) must be < paid limit ({m_paid.group(1)})"
    )


def test_p10_a04_admin_limit_is_none():
    src = _quota()
    # admin entry must map to None (unlimited)
    m = re.search(r'"admin"\s*:\s*(\S+)', src)
    assert m, "admin entry not found in PLAN_DAILY_LIMITS"
    assert m.group(1).rstrip(",") == "None", (
        "admin plan must be unlimited (None), got: " + m.group(1)
    )


def test_p10_a05_guest_has_positive_limit():
    src = _quota()
    m = re.search(r'"guest"\s*:\s*(\d+)', src)
    assert m, "guest limit not found as integer"
    assert int(m.group(1)) > 0, "guest limit must be positive"


def test_p10_a06_quota_key_uses_sha256_hash():
    src = _quota()
    assert "sha256" in src, "_quota_key should hash the username with SHA-256"
    assert "hexdigest" in src, "_quota_key should call hexdigest()"


def test_p10_a07_quota_key_includes_date():
    src = _quota()
    assert "strftime" in src or "%Y-%m-%d" in src, (
        "_quota_key must include a date component for daily reset"
    )


def test_p10_a08_check_and_increment_raises_429():
    src = _quota()
    assert "check_and_increment" in src, "check_and_increment function missing"
    assert "429" in src, "check_and_increment must raise HTTPException(429)"
    # Verify the Arabic quota-exceeded message is present
    assert "\u0627\u0644\u062d\u0635\u0629" in src or "استنفذت" in src, (
        "Arabic quota-exceeded message missing"
    )


# ── main.py wiring ────────────────────────────────────────────────────────────

def test_p10_a09_main_imports_check_quota():
    src = _src()
    assert "from plan_quota" in src, "main.py does not import from plan_quota"
    assert "_check_quota" in src or "check_and_increment" in src, (
        "check_and_increment not imported/aliased in main.py"
    )


def test_p10_a10_both_chat_endpoints_call_check_quota():
    src = _src()
    # /chat
    chat_idx = src.find('@app.post("/chat")')
    assert chat_idx >= 0, "/chat endpoint not found"
    chat_snippet = src[chat_idx: chat_idx + 300]
    assert "_check_quota" in chat_snippet, (
        "_check_quota not called in /chat endpoint"
    )
    # /chat/stream
    stream_idx = src.find('@app.post("/chat/stream")')
    assert stream_idx >= 0, "/chat/stream endpoint not found"
    stream_snippet = src[stream_idx: stream_idx + 300]
    assert "_check_quota" in stream_snippet, (
        "_check_quota not called in /chat/stream endpoint"
    )
