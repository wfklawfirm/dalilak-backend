"""
config.py — Dalilak AI startup security configuration.

Pure-Python, no external imports.  Called before any network client is
initialised so that a misconfigured secret is caught at process start,
not at request time.
"""
from __future__ import annotations

__all__ = ["ConfigurationError", "validate_security_configuration"]


class ConfigurationError(RuntimeError):
    """Raised when a required security configuration is invalid at startup."""


_KNOWN_JWT_DEFAULTS: frozenset[str] = frozenset({
    "dalilak-secret-CHANGE-IN-PROD",
})


def validate_security_configuration(
    jwt_secret: str,
    *,
    known_defaults: frozenset[str] = _KNOWN_JWT_DEFAULTS,
) -> None:
    """
    Verify that jwt_secret is not a known insecure default and is not empty.

    This function is pure — no I/O, no network, no external imports.
    It must be called before any external service client is initialised.

    Security: the error message MUST NOT include the value of jwt_secret.

    Returns None on success.  Raises ConfigurationError on failure.
    """
    if not jwt_secret or jwt_secret in known_defaults:
        raise ConfigurationError(
            "[FATAL] JWT_SECRET is set to a known insecure default or is empty. "
            "Set JWT_SECRET to a randomly generated secret of at least 32 characters "
            "in the Render environment variables. The service will not start."
        )
