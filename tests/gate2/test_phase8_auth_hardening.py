"""
Phase 8 -- Auth/Session Hardening -- Tier A (offline, source-inspection) tests
12 tests, zero network calls, zero env vars required.
"""
import pathlib
import re

BACKEND = pathlib.Path(__file__).parent.parent.parent
SRC_PATH = BACKEND / "main.py"


def _src() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


def test_p8_a01_jwt_expiry_days_constant():
    src = _src()
    assert "JWT_EXPIRY_DAYS" in src, "JWT_EXPIRY_DAYS constant not found"
    m = re.search(r"JWT_EXPIRY_DAYS\s*=\s*(\d+)", src)
    assert m, "JWT_EXPIRY_DAYS not assigned"
    assert int(m.group(1)) == 7, "Expected 7, got " + m.group(1)


def test_p8_a02_create_token_uses_expiry_constant():
    src = _src()
    assert "days=30" not in src, "Old 30-day expiry still present in source"
    assert "JWT_EXPIRY_DAYS" in src


def test_p8_a03_create_token_has_iss_claim():
    src = _src()
    assert ('"iss"' in src or "'iss'" in src), "'iss' claim missing from token payload"
    assert "dalilak-ai" in src, "iss value 'dalilak-ai' not found"


def test_p8_a04_create_token_has_jti_claim():
    src = _src()
    assert ('"jti"' in src or "'jti'" in src), "'jti' claim missing from token payload"
    assert "token_hex" in src, "secrets.token_hex not used for jti"


def test_p8_a05_revoked_tokens_dict_exists():
    src = _src()
    assert "_revoked_tokens" in src, "_revoked_tokens blocklist not found"
    assert "dict" in src, "_revoked_tokens should be a dict"


def test_p8_a06_blocklist_prune_function_exists():
    src = _src()
    assert "def _blocklist_prune" in src, "_blocklist_prune function missing"
    assert "_revoked_tokens.pop" in src, "_blocklist_prune should pop stale entries"


def test_p8_a07_get_current_user_checks_blocklist():
    src = _src()
    gcu_idx = src.find("async def get_current_user")
    assert gcu_idx >= 0, "get_current_user not found"
    check_idx = src.find("in _revoked_tokens", gcu_idx)
    assert check_idx > gcu_idx, (
        "'in _revoked_tokens' membership check not found in get_current_user"
    )


def test_p8_a08_logout_endpoint_exists():
    src = _src()
    has_logout = '"/auth/logout"' in src or "'/auth/logout'" in src
    assert has_logout, "/auth/logout endpoint not registered"
    assert "async def logout" in src, "logout handler function missing"


def test_p8_a09_logout_adds_to_blocklist():
    src = _src()
    logout_idx = src.find('"/auth/logout"')
    assert logout_idx >= 0
    revoke_idx = src.find("_revoked_tokens[jti]", logout_idx)
    assert revoke_idx > logout_idx, "logout does not add jti to _revoked_tokens"


def test_p8_a10_cors_no_wildcard():
    src = _src()
    mw_idx = src.find("app.add_middleware")
    assert mw_idx >= 0, "app.add_middleware not found"
    snippet = src[mw_idx: mw_idx + 400]
    # allow_origins specifically must not be the wildcard list
    m = re.search(r"allow_origins\s*=\s*(\[[^\]]+\])", snippet)
    assert m, "allow_origins not found in middleware block"
    origins_val = m.group(1)
    assert origins_val != '["*"]', "CORS allow_origins is still wildcard -- must be restricted"
    assert "APP_BASE_URL" in origins_val, "allow_origins should include APP_BASE_URL"


def test_p8_a11_register_password_min_8():
    src = _src()
    register_idx = src.find("@app.post(\"/auth/register\")")
    assert register_idx >= 0, "/auth/register not found"
    snippet = src[register_idx: register_idx + 500]
    assert "< 8" in snippet, "Register endpoint does not enforce min 8-char password"
    assert "< 6" not in snippet, "Register endpoint still has old 6-char minimum"


def test_p8_a12_reset_password_min_8():
    src = _src()
    reset_idx = src.find("@app.post(\"/auth/reset-password\")")
    assert reset_idx >= 0, "/auth/reset-password not found"
    snippet = src[reset_idx: reset_idx + 500]
    assert "< 8" in snippet, "reset-password endpoint does not enforce min 8-char password"
    assert "< 6" not in snippet, "reset-password endpoint still has old 6-char minimum"
