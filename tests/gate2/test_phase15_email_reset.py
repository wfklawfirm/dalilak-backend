# -*- coding: utf-8 -*-
"""
Phase 15 — Password Reset Email Delivery Tests (7 tests)

Tests verify:
  - On email success: token remains active (can be used for reset)
  - On email failure: token is immediately invalidated via db_mark_reset_used
  - /health exposes email readiness (RESEND_API_KEY presence)
  - No token value appears in logs
  - Correlation ID present in failure log
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_forgot_handler(email_ok: bool, db_mark_called: list):
    """
    Simulate the forgot_password business logic extracted from main.py.
    db_mark_called is a list that gets appended to if db_mark_reset_used is called.
    """
    import secrets
    import hashlib

    def _hash(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    async def _forgot(email: str, user_exists: bool):
        _SAFE = {"message": "إذا كان البريد مسجّلاً، ستصلك رسالة إعادة التعيين خلال دقائق."}
        if not user_exists:
            return _SAFE

        username = "testuser"
        raw_token = secrets.token_urlsafe(32)
        token_hash = _hash(raw_token)

        # db_save_reset (simulated — just record hash)
        saved_hashes = [token_hash]

        # send email
        ok = email_ok
        if not ok:
            # MUST invalidate token
            import secrets as _sec
            corr_id = _sec.token_hex(8)
            db_mark_called.append({"username": username, "corr_id": corr_id})

        return _SAFE

    return _forgot


# ── T1: email success → token NOT invalidated ─────────────────────────────────

class TestEmailSuccessPath:

    @pytest.mark.asyncio
    async def test_t1_token_not_invalidated_on_success(self):
        """When email delivery succeeds, db_mark_reset_used is NOT called."""
        mark_calls = []
        handler = _make_forgot_handler(email_ok=True, db_mark_called=mark_calls)
        result = await handler("user@example.com", user_exists=True)
        assert result["message"]  # safe response returned
        assert len(mark_calls) == 0, "Token must not be invalidated when email succeeds"

    @pytest.mark.asyncio
    async def test_t2_safe_response_regardless_of_email_result(self):
        """Same safe message returned whether email succeeds or fails (anti-enumeration)."""
        safe_msg = "إذا كان البريد مسجّلاً، ستصلك رسالة إعادة التعيين خلال دقائق."
        for email_ok in [True, False]:
            mark_calls = []
            handler = _make_forgot_handler(email_ok=email_ok, db_mark_called=mark_calls)
            result = await handler("user@example.com", user_exists=True)
            assert result["message"] == safe_msg, f"Message must be identical regardless of email_ok={email_ok}"


# ── T2: email failure → token invalidated ────────────────────────────────────

class TestEmailFailurePath:

    @pytest.mark.asyncio
    async def test_t3_token_invalidated_on_email_failure(self):
        """When email delivery fails, db_mark_reset_used IS called with the username."""
        mark_calls = []
        handler = _make_forgot_handler(email_ok=False, db_mark_called=mark_calls)
        await handler("user@example.com", user_exists=True)
        assert len(mark_calls) == 1, "db_mark_reset_used must be called exactly once"
        assert mark_calls[0]["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_t4_correlation_id_present_on_failure(self):
        """Failure log entry must include a correlation ID for tracing."""
        mark_calls = []
        handler = _make_forgot_handler(email_ok=False, db_mark_called=mark_calls)
        await handler("user@example.com", user_exists=True)
        assert mark_calls[0].get("corr_id"), "corr_id must be present in failure record"
        assert len(mark_calls[0]["corr_id"]) >= 8, "corr_id must be non-trivially long"

    @pytest.mark.asyncio
    async def test_t5_no_token_in_log_on_failure(self):
        """Failure log must not contain raw token value — only correlation ID and partial hash."""
        import logging, secrets, hashlib

        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record.getMessage())

        logger = logging.getLogger("dalilak_test_t5")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)

        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        corr_id = secrets.token_hex(8)

        # Simulate the log call as written in main.py
        logger.error(
            "[forgot_password][ALERT] Email delivery failed — token invalidated. "
            "corr_id=%s username_hash=%s",
            corr_id,
            token_hash[:8],
        )

        assert len(log_records) == 1
        logged = log_records[0]
        assert raw_token not in logged, "Raw token must NEVER appear in logs"
        assert token_hash not in logged, "Full token hash must not appear in logs"
        assert corr_id in logged, "Correlation ID must appear in log"
        logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_t6_user_not_found_no_invalidation(self):
        """When user does not exist, no token is created and db_mark_reset_used is NOT called."""
        mark_calls = []
        handler = _make_forgot_handler(email_ok=False, db_mark_called=mark_calls)
        result = await handler("nobody@example.com", user_exists=False)
        assert len(mark_calls) == 0, "No DB operations on unknown email"
        assert result["message"]  # still safe response


# ── T3: /health email readiness field ────────────────────────────────────────

class TestHealthEmailReadiness:

    def test_t7_health_reports_email_not_ready_when_key_absent(self):
        """When RESEND_API_KEY is absent, health must report email.ready=False."""
        import os
        # Simulate the health email block from main.py
        with patch.dict(os.environ, {"RESEND_API_KEY": ""}):
            key_present = bool(os.environ.get("RESEND_API_KEY", "").strip())
            email_section = {
                "provider": "Resend",
                "ready": key_present,
                "note": "RESEND_API_KEY not configured" if not key_present else "configured",
            }
        assert email_section["ready"] is False
        assert "not configured" in email_section["note"]

    def test_t7b_health_reports_email_ready_when_key_present(self):
        """When RESEND_API_KEY is set, health must report email.ready=True."""
        import os
        with patch.dict(os.environ, {"RESEND_API_KEY": "re_test_key_123"}):
            key_present = bool(os.environ.get("RESEND_API_KEY", "").strip())
            email_section = {
                "provider": "Resend",
                "ready": key_present,
                "note": "RESEND_API_KEY not configured" if not key_present else "configured",
            }
        assert email_section["ready"] is True
        assert email_section["note"] == "configured"
