"""
plan_quota.py  --  Per-plan daily message quota enforcement for Dalilak AI.

Key pattern : dalilak:quota:{sha256(username)[:16]}:{date_utc}
TTL         : 25 hours (survives UTC day rollover)
Redis       : reuses the same async client as rate_limit.py.
Fallback    : in-process counter (correct for single-replica; does not sync
              across multiple Render instances).

Usage
-----
    from plan_quota import check_and_increment as _check_quota

    @app.post("/chat")
    async def chat(req, request, user = Depends(get_current_user)):
        await _check_quota(user["username"], user.get("plan", "trial"))
        ...

Phase 10 addition.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

_log = logging.getLogger("dalilak")

# ── Per-plan daily message ceilings (UTC calendar day). None = unlimited. ────
PLAN_DAILY_LIMITS: dict[str, Optional[int]] = {
    "trial": 20,    # free trial users
    "paid":  200,   # paying subscribers
    "admin": None,  # unlimited
    "guest": 5,     # unauthenticated visitors
}

_QUOTA_TTL_SEC: int = 25 * 3600   # 25 h -- survives UTC midnight rollover


# ── Key construction ──────────────────────────────────────────────────────────

def _quota_key(username: str) -> str:
    """
    Returns a privacy-preserving Redis key.
    The username is hashed (SHA-256, first 16 hex chars) so that logs and
    key dumps never expose PII.  The key resets each UTC calendar day.
    """
    uid_hash = hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"dalilak:quota:{uid_hash}:{date_utc}"


# ── Shared Redis client (from rate_limit module) ──────────────────────────────

async def _redis():
    """
    Lazily import _get_redis from rate_limit to reuse the same async client.
    Returns the Redis client, or None when Redis is unavailable.
    """
    from rate_limit import _get_redis  # noqa: PLC0415
    return await _get_redis()


# ── In-memory fallback ────────────────────────────────────────────────────────
_mem_counts: dict[str, int] = {}
_mem_lock = asyncio.Lock()


async def _mem_check_and_increment(key: str, limit: int) -> int:
    """Increment in-process counter; return remaining quota."""
    async with _mem_lock:
        count = _mem_counts.get(key, 0) + 1
        _mem_counts[key] = count
        if count > limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    "لقد استنفذت حصتك اليومية — "
                    "حاول مجدداً غداً أو قم بالترقية إلى الاشتراك المدفوع"
                ),
            )
        return limit - count


# ── Public API ────────────────────────────────────────────────────────────────

async def check_and_increment(username: str, plan: str) -> int:
    """
    Enforce the daily message quota for *username* on *plan*.

    Returns the number of messages remaining today, or -1 for unlimited.
    Raises HTTPException(429) when the quota is exceeded.

    This function MUST be awaited before any AI work is done for the request.
    """
    limit: Optional[int] = PLAN_DAILY_LIMITS.get(plan)

    if limit is None:
        return -1   # admin or unknown plan → unlimited, never block

    key = _quota_key(username)
    r = await _redis()

    if r is not None:
        try:
            # INCR is atomic; EXPIRE resets TTL on every request (safe).
            count = await r.incr(key)
            await r.expire(key, _QUOTA_TTL_SEC)
            if count > limit:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "لقد استنفذت حصتك اليومية — "
                        "حاول مجدداً غداً أو قم بالترقية إلى الاشتراك المدفوع"
                    ),
                )
            return limit - count
        except HTTPException:
            raise
        except Exception as exc:
            # Redis failure → fail open (never block a request due to infra error).
            _log.warning("plan_quota: Redis error — failing open: %s", exc)
            return limit   # assume no messages used today

    # No Redis: use in-process fallback.
    return await _mem_check_and_increment(key, limit)
