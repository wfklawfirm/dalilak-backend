"""
Phase 13 -- Final Validation -- Tier A (offline, source-inspection) tests
8 tests: verify all phase constants, module imports, and test file coverage.
Zero network calls, zero env vars required.
"""
import pathlib

BACKEND   = pathlib.Path(__file__).parent.parent.parent
SRC_PATH  = BACKEND / "main.py"
TEST_DIR  = BACKEND / "tests" / "gate2"


def _src() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


# -- Test coverage completeness -----------------------------------------------

def test_p13_a01_all_phase_test_files_exist():
    """Every implemented phase must have a Tier A test file."""
    expected = [
        "test_jwt_validation.py",
        "test_phase4_password_reset.py",
        "test_phase5_evidence_gate.py",
        "test_phase6_golden_dataset.py",
        "test_phase8_auth_hardening.py",
        "test_phase9_observability.py",
        "test_phase10_plan_quota.py",
        "test_phase11_content_ops.py",
        "test_phase12_perf_hardening.py",
    ]
    for name in expected:
        path = TEST_DIR / name
        assert path.exists(), f"Missing test file: {name}"


def test_p13_a02_golden_dataset_present():
    golden = TEST_DIR / "golden_v1.json"
    assert golden.exists(), "golden_v1.json missing"


# -- All security constants present in main.py --------------------------------

def test_p13_a03_jwt_constants_present():
    src = _src()
    for const in ("JWT_SECRET", "JWT_ALGO", "JWT_EXPIRY_DAYS", "create_token", "decode_token"):
        assert const in src, f"JWT constant/function missing: {const}"


def test_p13_a04_auth_hardening_constants_present():
    src = _src()
    for symbol in ("_revoked_tokens", "_blocklist_prune", "jti", "iss"):
        assert symbol in src, f"Phase 8 symbol missing: {symbol}"


def test_p13_a05_observability_present():
    src = _src()
    for symbol in ("_req_id_var", "_request_id_middleware", "_global_exception_handler", "_log"):
        assert symbol in src, f"Phase 9 symbol missing: {symbol}"


def test_p13_a06_quota_and_limits_present():
    src = _src()
    # Symbols that must be in main.py (imported or defined there)
    for symbol in ("_check_quota", "MAX_MESSAGE_LEN",
                   "QDRANT_TIMEOUT_SEC", "OPENAI_TIMEOUT_SEC"):
        assert symbol in src, f"Phase 10/12 symbol missing from main.py: {symbol}"
    # PLAN_DAILY_LIMITS is defined in plan_quota.py, not main.py
    quota_src = (BACKEND / "plan_quota.py").read_text(encoding="utf-8")
    assert "PLAN_DAILY_LIMITS" in quota_src, "PLAN_DAILY_LIMITS missing from plan_quota.py"


# -- Key security behaviours present ------------------------------------------

def test_p13_a07_cors_not_wildcard():
    src = _src()
    import re
    mw_idx = src.find("app.add_middleware")
    assert mw_idx >= 0
    snippet = src[mw_idx: mw_idx + 400]
    m = re.search(r"allow_origins\s*=\s*(\[[^\]]+\])", snippet)
    assert m, "allow_origins not found"
    assert m.group(1) != '["*"]', "CORS still uses wildcard"


def test_p13_a08_password_minimum_is_8():
    src = _src()
    register_idx = src.find('@app.post("/auth/register")')
    assert register_idx >= 0
    snippet = src[register_idx: register_idx + 500]
    assert "< 8" in snippet, "Register does not enforce 8-char minimum"
    assert "< 6" not in snippet, "Register still has old 6-char minimum"
