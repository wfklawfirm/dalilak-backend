"""
config.py — Dalilak AI startup security configuration.

Pure-Python, no external imports.  Called before any network client is
initialised so that a misconfigured secret is caught at process start,
not at request time.
"""
from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "validate_security_configuration",
    "validate_admin_secret_configuration",
]


class ConfigurationError(RuntimeError):
    """Raised when a required security configuration is invalid at startup."""


_MIN_SECRET_LENGTH = 32

_KNOWN_JWT_DEFAULTS: frozenset[str] = frozenset({
    "dalilak-secret-CHANGE-IN-PROD",
})

_KNOWN_ADMIN_DEFAULTS: frozenset[str] = frozenset({
    "dalilak-admin-CHANGE-IN-PROD",
})


def validate_security_configuration(
    jwt_secret: str,
    *,
    known_defaults: frozenset[str] = _KNOWN_JWT_DEFAULTS,
    min_length: int = _MIN_SECRET_LENGTH,
) -> None:
    """
    Verify that jwt_secret is not a known insecure default, is not empty,
    and meets the minimum length requirement.

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
    if len(jwt_secret) < min_length:
        raise ConfigurationError(
            f"[FATAL] JWT_SECRET is shorter than the required minimum of "
            f"{min_length} characters. Set JWT_SECRET to a randomly generated "
            f"secret of at least {min_length} characters in the Render "
            f"environment variables. The service will not start."
        )


def validate_admin_secret_configuration(
    admin_secret: str,
    *,
    known_defaults: frozenset[str] = _KNOWN_ADMIN_DEFAULTS,
    min_length: int = _MIN_SECRET_LENGTH,
) -> None:
    """
    Verify that admin_secret is not a known insecure default, is not empty,
    and meets the minimum length requirement.

    Mirrors validate_security_configuration() for JWT_SECRET. Added because
    ADMIN_SECRET previously had no startup validation at all, unlike
    JWT_SECRET — inconsistent even though verify_admin_secret() is currently
    unwired/dead code. Validating now prevents the gap from becoming
    exploitable if/when that code path is wired up.

    Security: the error message MUST NOT include the value of admin_secret.

    Returns None on success.  Raises ConfigurationError on failure.
    """
    if not admin_secret or admin_secret in known_defaults:
        raise ConfigurationError(
            "[FATAL] ADMIN_SECRET is set to a known insecure default or is "
            "empty. Set ADMIN_SECRET to a randomly generated secret of at "
            "least 32 characters in the Render environment variables. "
            "The service will not start."
        )
    if len(admin_secret) < min_length:
        raise ConfigurationError(
            f"[FATAL] ADMIN_SECRET is shorter than the required minimum of "
            f"{min_length} characters. Set ADMIN_SECRET to a randomly "
            f"generated secret of at least {min_length} characters in the "
            f"Render environment variables. The service will not start."
        )
