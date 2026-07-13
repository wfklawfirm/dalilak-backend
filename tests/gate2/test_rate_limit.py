"""
Phase 3 — Rate-limit unit tests (Tier A: no network, no Redis).

All tests run against the in-memory fallback path so they are completely
offline and do not require a running Redis instance.

SECURITY: no real user credentials, no real IPs, no paid-service calls.
"""
from __future__ import annotations

import asyncio
import os
import pytest

# Force in-memory fallback — unset REDIS_URL before rate_limit is imported.
os.environ.pop("REDIS_URL", None)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_request(ip: str = "1.2.3.4") -> object:
    """Minimal stub that satisfies _get_ip()."""
    class _Headers:
        def get(self, name, default=""):
            return default   # no X-Forwarded-For
    class _Client:
        host = ip
    class _Request:
        headers = _Headers()
        client = _Client()
    return _Request()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.tier_a
def test_p3_a1_policy_keys_defined():
    """All expected endpoint keys exist in _POLICIES."""
    from rate_limit import _POLICIES
    for ep in ("login", "register", "forgot", "chat"):
        assert ep in _POLICIES, f"Missing policy for endpoint '{ep}'"
    # chat must have user layers
    assert "user" in _POLICIES["chat"]
    assert "user_ep" in _POLICIES["chat"]


@pytest.mark.tier_a
def test_p3_a2_enforce_allows_under_limit():
    """First request must be allowed (no prior history)."""
    from rate_limit import _mem

    _mem.clear()   # fresh state

    async def _run():
        from rate_limit import enforce
        req = _mock_request("10.0.0.1")
        # Should not raise
        await enforce(req, "login")

    asyncio.run(_run())


@pytest.mark.tier_a
def test_p3_a3_in_memory_blocks_after_limit():
    """After exhausting ip_ep limit for 'forgot' (3/15 min), next call must raise 429."""
    from rate_limit import _mem, _check_memory
    from fastapi import HTTPException

    _mem.clear()

    async def _run():
        from rate_limit import enforce

        req = _mock_request("192.168.0.5")
        # 'forgot' ip_ep limit = 3 per 900s
        # Override _check_memory won't work cleanly, so call enforce 3 times
        # to exhaust the ip_ep bucket, then confirm 4th raises 429.
        for _ in range(3):
            await enforce(req, "forgot")
        # 4th must be blocked
        with pytest.raises(HTTPException) as exc_info:
            await enforce(req, "forgot")
        assert exc_info.value.status_code == 429

    asyncio.run(_run())


@pytest.mark.tier_a
def test_p3_a4_different_ips_do_not_share_bucket():
    """Two distinct IPs each get their own bucket — neither blocks the other."""
    from rate_limit import _mem
    from fastapi import HTTPException

    _mem.clear()

    async def _run():
        from rate_limit import enforce

        req_a = _mock_request("10.10.10.1")
        req_b = _mock_request("10.10.10.2")

        # Exhaust ip_ep for IP A on 'forgot' (limit = 3)
        for _ in range(3):
            await enforce(req_a, "forgot")

        # IP B should still be allowed
        try:
            await enforce(req_b, "forgot")
        except HTTPException:
            pytest.fail("IP B was incorrectly blocked by IP A's bucket")

    asyncio.run(_run())


@pytest.mark.tier_a
def test_p3_a5_chat_user_layer_checked():
    """chat endpoint with user_id exercises layers 4+5 without raising (under limit)."""
    from rate_limit import _mem

    _mem.clear()

    async def _run():
        from rate_limit import enforce

        req = _mock_request("172.16.0.1")
        # Should not raise — first call per user
        await enforce(req, "chat", user_id="test_user_alpha")

    asyncio.run(_run())


@pytest.mark.tier_a
def test_p3_a6_retry_after_header_present():
    """429 response must include Retry-After header."""
    from rate_limit import _mem
    from fastapi import HTTPException

    _mem.clear()

    async def _run():
        from rate_limit import enforce

        req = _mock_request("203.0.113.1")
        for _ in range(3):
            await enforce(req, "forgot")
        with pytest.raises(HTTPException) as exc_info:
            await enforce(req, "forgot")
        exc = exc_info.value
        assert exc.status_code == 429
        headers = exc.headers or {}
        assert "Retry-After" in headers, "429 must include Retry-After header"
        assert int(headers["Retry-After"]) > 0

    asyncio.run(_run())


@pytest.mark.tier_a
def test_p3_a7_get_ip_xff_respected():
    """_get_ip must prefer leftmost X-Forwarded-For value."""
    from rate_limit import _get_ip

    class _Headers:
        def get(self, name, default=""):
            if name == "x-forwarded-for":
                return "5.6.7.8, 10.0.0.1, 192.168.1.1"
            return default

    class _Req:
        headers = _Headers()
        client = None

    ip = _get_ip(_Req())
    assert ip == "5.6.7.8"


@pytest.mark.tier_a
def test_p3_a8_no_redis_url_uses_memory_fallback():
    """With REDIS_URL unset, _get_redis() must return None (in-memory path)."""
    os.environ.pop("REDIS_URL", None)

    async def _run():
        import importlib
        import rate_limit as rl

        # Reset init state so _get_redis() runs fresh
        rl._redis_init_done = False
        rl._redis = None
        rl._redis_lock = None

        result = await rl._get_redis()
        assert result is None, "_get_redis() must return None when REDIS_URL is unset"

    asyncio.run(_run())
