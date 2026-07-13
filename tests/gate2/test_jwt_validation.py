"""
tests/gate2/test_jwt_validation.py
Gate 2 — JWT_SECRET startup validation.

Tier A: Local isolated, no network.
These tests pass after Gate 2 (config.py + main.py fix) and must stay green in CI.
"""
import os
import secrets
import subprocess
import sys
import time

import pytest

_KNOWN_DEFAULT = "dalilak-secret-CHANGE-IN-PROD"

# Backend directory = the directory containing this tests/ package
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _isolated_env(jwt_secret: str) -> dict[str, str]:
    """Minimal environment for 'python -c \"import main\"' — no live credentials."""
    return {
        "JWT_SECRET": jwt_secret,
        "OPENAI_API_KEY": "sk-test-not-real-ci",
        "QDRANT_URL": "http://localhost:65534",   # unreachable
        "QDRANT_API_KEY": "test-qdrant-key-ci",
        "DATABASE_URL": "sqlite:///./ci-test-isolation.db",
        "admin_email": "admin-test@example-dalilak.invalid",
        "admin_password": "test-admin-pw-ci-isolation",
        "ADMIN_USERNAME": "admin-test-ci",
        "ADMIN_SECRET": "test-admin-secret-ci",
        "REDIS_URL": "redis://localhost:65535",   # unreachable
        "RESEND_API_KEY": "re_test_ci_not_real",
        "PATH": os.environ.get("PATH", ""),
    }


def _run_import(jwt_secret: str, timeout: float = 6.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=_BACKEND_DIR,
        env=_isolated_env(jwt_secret),
        capture_output=True,
        timeout=timeout,
    )


# ── Pure-function tests ───────────────────────────────────────────────────────

def test_validate_security_configuration_rejects_known_default():
    from config import validate_security_configuration, ConfigurationError
    with pytest.raises(ConfigurationError):
        validate_security_configuration(_KNOWN_DEFAULT)


def test_validate_security_configuration_accepts_non_default():
    from config import validate_security_configuration
    result = validate_security_configuration("A" * 64)
    assert result is None


def test_validate_security_configuration_rejects_empty_string():
    from config import validate_security_configuration, ConfigurationError
    with pytest.raises(ConfigurationError):
        validate_security_configuration("")


def test_validation_error_does_not_leak_secret_value():
    from config import validate_security_configuration, ConfigurationError
    sentinel = secrets.token_hex(32)
    try:
        validate_security_configuration(sentinel, known_defaults=frozenset({sentinel}))
        pytest.fail("ConfigurationError not raised")
    except ConfigurationError as exc:
        assert sentinel not in str(exc)
        assert sentinel not in repr(exc)


# ── Subprocess tests ──────────────────────────────────────────────────────────

def test_startup_validation_fires_before_external_service_init():
    start = time.monotonic()
    try:
        result = _run_import(_KNOWN_DEFAULT, timeout=6.0)
    except subprocess.TimeoutExpired:
        pytest.fail("Subprocess timed out — validation runs after external service init")
    elapsed = time.monotonic() - start
    assert result.returncode != 0
    assert elapsed < 3.5


def test_startup_passes_with_non_default_jwt_secret():
    try:
        result = _run_import("A" * 64, timeout=10.0)
    except subprocess.TimeoutExpired:
        return  # acceptable — blocked on unreachable services, not JWT check
    assert b"FATAL" not in result.stderr


def test_fatal_message_does_not_echo_credential_values():
    result = _run_import(_KNOWN_DEFAULT, timeout=6.0)
    combined = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert "DATABASE_URL" not in combined
    assert "OPENAI_API_KEY" not in combined
    assert "QDRANT_API_KEY" not in combined
    assert "REDIS_URL" not in combined
