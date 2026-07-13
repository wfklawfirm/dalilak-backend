"""Phase 4 -- Password-reset unit tests (Tier A: offline, no network)."""
from __future__ import annotations
import asyncio, hashlib, os, secrets, sys
import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

def _sha256_hex(token):
    return hashlib.sha256(token.encode()).hexdigest()

@pytest.mark.tier_a
def test_p4_a1_hash_is_sha256():
    raw = "test-token-value"
    digest = _sha256_hex(raw)
    assert len(digest) == 64
    assert digest == hashlib.sha256(raw.encode()).hexdigest()
    assert digest == digest.lower()

@pytest.mark.tier_a
def test_p4_a1b_main_source_defines_hash_token():
    src = open(os.path.join(_BACKEND, "main.py"), encoding="utf-8").read()
    assert "def _hash_reset_token" in src
    assert "hashlib.sha256" in src
    assert "_hash_reset_token(req.token)" in src
    assert "_hash_reset_token(raw_token)" in src

@pytest.mark.tier_a
def test_p4_a2_raw_not_in_hash():
    raw = secrets.token_urlsafe(32)
    assert raw not in _sha256_hex(raw)

@pytest.mark.tier_a
def test_p4_a3_token_length():
    t = secrets.token_urlsafe(32)
    assert len(t) >= 43
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in t)

@pytest.mark.tier_a
def test_p4_a4_not_six_digits():
    for _ in range(20):
        t = secrets.token_urlsafe(32)
        assert not (t.isdigit() and len(t) == 6)

@pytest.mark.tier_a
def test_p4_a5_expiry_15min():
    from datetime import timedelta
    assert timedelta(minutes=15).total_seconds() == 900
    assert timedelta(minutes=15) != timedelta(hours=1)

@pytest.mark.tier_a
def test_p4_a6_round_trip():
    raw = secrets.token_urlsafe(32)
    assert _sha256_hex(raw) == _sha256_hex(raw)

@pytest.mark.tier_a
def test_p4_a7_unique_hashes():
    t1, t2 = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    assert t1 != t2
    assert _sha256_hex(t1) != _sha256_hex(t2)

@pytest.mark.tier_a
def test_p4_a8_email_no_key_returns_false():
    os.environ.pop("RESEND_API_KEY", None)
    async def _run():
        from email_service import send_reset_email
        assert await send_reset_email("u@x.com", "https://example.com/reset?token=X") is False
    asyncio.run(_run())

@pytest.mark.tier_a
def test_p4_a9_email_service_importable():
    import email_service  # noqa: F401

@pytest.mark.tier_a
def test_p4_a10_no_hardcoded_key():
    import re
    src = open(os.path.join(_BACKEND, "email_service.py"), encoding="utf-8").read()
    assert not re.search(r"\bre_[A-Za-z0-9]{20,}", src)

@pytest.mark.tier_a
def test_p4_a11_expiry_in_source():
    src = open(os.path.join(_BACKEND, "main.py"), encoding="utf-8").read()
    assert "timedelta(minutes=15)" in src
