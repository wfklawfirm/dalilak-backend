"""
rate_limit.py — Atomic sliding-window rate limiter for Dalilak AI.

Uses Redis (async) when REDIS_URL is set.
Falls back to an in-process asyncio-safe store when Redis is unavailable.
The in-memory fallback is correct for a single-process deployment (Render);
it does not synchronise across multiple replicas.

5-layer key strategy
──────────────────────────────────────────────────────────────────────
Layer  Key pattern                    Purpose
─────  ────────────────────────────── ─────────────────────────────────
  1    rl:ip:{ip}                     Global per-IP ceiling (DDoS)
  2    rl:ip:{ip}:ep:{ep}             Per-IP per-endpoint (brute-force)
  3    rl:ep:{ep}                     Per-endpoint global ceiling
  4    rl:user:{uid}                  Per-user global (authenticated)
  5    rl:user:{uid}:ep:{ep}          Per-user per-endpoint (chat abuse)
──────────────────────────────────────────────────────────────────────

Usage
─────
    from rate_limit import enforce, Layer

    @app.post("/auth/login")
    async def login(req: LoginRequest, request: Request):
        await enforce(request, "login", user_id=None)
        ...

    @app.post("/chat")
    async def chat(req: ChatRequest, user=Depends(get_current_user), request: Request = ...):
        await enforce(request, "chat", user_id=user["username"])
        ...
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
from typing import Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# ── Lua sliding-window script ─────────────────────────────────────────────────
# Keys:  KEYS[1] = bucket key
# Args:  ARGV[1] = now_ms (int)   ARGV[2] = window_ms (int)   ARGV[3] = limit (int)
# Returns: 1 = allowed,  0 = denied
_LUA = """
local key      = KEYS[1]
local now_ms   = tonumber(ARGV[1])
local win_ms   = tonumber(ARGV[2])
local limit    = tonumber(ARGV[3])
local cutoff   = now_ms - win_ms
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local cnt = redis.call('ZCARD', key)
if cnt < limit then
    local member = now_ms .. ':' .. redis.call('INCR', key .. ':c')
    redis.call('ZADD', key, now_ms, member)
    redis.call('PEXPIRE', key, win_ms + 5000)
    return 1
end
return 0
"""

# ── Redis lazy init ───────────────────────────────────────────────────────────
_redis: object | None = None          # redis.asyncio.Redis or None
_lua_fn: object | None = None         # registered script callable or None
_redis_init_done = False
_redis_lock: asyncio.Lock | None = None


async def _get_redis():
    """Return async Redis client, or None if unavailable."""
    global _redis, _lua_fn, _redis_init_done, _redis_lock

    if _redis_init_done:
        return _redis

    # Create lock lazily (must be inside an event loop)
    if _redis_lock is None:
        _redis_lock = asyncio.Lock()

    async with _redis_lock:
        if _redis_init_done:
            return _redis

        url = os.getenv("REDIS_URL", "").strip()
        if not url:
            logger.warning(
                "[rate_limit] REDIS_URL not set — using in-memory fallback. "
                "Set REDIS_URL (e.g. Upstash) for distributed rate limiting."
            )
            _redis_init_done = True
            return None

        try:
            import redis.asyncio as aioredis  # type: ignore[import]
            client = aioredis.from_url(url, decode_responses=True)
            await client.ping()
            _lua_fn = client.register_script(_LUA)
            _redis = client
            logger.info("[rate_limit] Redis connected")
        except Exception as exc:
            logger.warning("[rate_limit] Redis unavailable (%s) — using in-memory fallback", exc)
            _redis = None

        _redis_init_done = True
        return _redis


# ── In-memory fallback ────────────────────────────────────────────────────────
_mem: dict[str, collections.deque] = {}
_mem_lock = asyncio.Lock()


async def _check_memory(key: str, window_ms: int, limit: int) -> bool:
    now_ms = int(time.monotonic() * 1000)
    cutoff = now_ms - window_ms
    async with _mem_lock:
        dq = _mem.setdefault(key, collections.deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) < limit:
            dq.append(now_ms)
            return True
        return False


# ── Core check ────────────────────────────────────────────────────────────────

async def _check(key: str, window_ms: int, limit: int) -> bool:
    """True = allowed, False = denied."""
    r = await _get_redis()
    if r is not None:
        try:
            now_ms = int(time.time() * 1000)
            result = await _lua_fn(keys=[key], args=[now_ms, window_ms, limit])
            return bool(result)
        except Exception as exc:
            logger.error("[rate_limit] Redis error — failing open: %s", exc)
            return True   # fail open: never block on infrastructure error
    return await _check_memory(key, window_ms, limit)


# ── IP extraction ─────────────────────────────────────────────────────────────

def _get_ip(request: Request) -> str:
    """
    Extract real client IP, respecting Render's reverse proxy.
    X-Forwarded-For may contain a comma-separated list; the leftmost is the client.
    Falls back to request.client.host.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# ── Rate-limit policy table ───────────────────────────────────────────────────
# (window_sec, limit)  —  all windows are in seconds for readability
_POLICIES: dict[str, dict[str, tuple[int, int]]] = {
    #            layer-1-ip      layer-2-ip+ep   layer-3-ep-global
    "login":     {
        "ip":          (60,   120),   # L1: 120 req/min per IP (all endpoints)
        "ip_ep":       (300,    5),   # L2:   5 attempts / 5 min per IP  (brute-force)
        "ep":          (60,   200),   # L3: 200 login/min globally
    },
    "register":  {
        "ip":          (60,   120),
        "ip_ep":       (3600,   5),   # L2:   5 registrations / hour per IP
        "ep":          (60,   100),
    },
    "forgot":    {
        "ip":          (60,   120),
        "ip_ep":       (900,    3),   # L2:   3 reset requests / 15 min per IP
        "ep":          (60,    60),
    },
    "chat":      {
        "ip":          (60,   120),   # L1
        "ip_ep":       (60,    30),   # L2: 30 chat/min per IP
        "ep":          (60,  1000),   # L3: 1000 chat/min globally
        "user":        (3600, 200),   # L4: 200 chat/hour per user
        "user_ep":     (3600, 100),   # L5: 100 chat/hour per user per endpoint
    },
}


async def enforce(
    request: Request,
    endpoint: str,
    user_id: Optional[str] = None,
) -> None:
    """
    Enforce all applicable rate-limit layers for the given endpoint.
    Raises HTTPException(429) if any layer is exceeded.

    endpoint: one of "login", "register", "forgot", "chat"
    user_id:  authenticated username, or None for unauthenticated endpoints
    """
    policy = _POLICIES.get(endpoint, {})
    ip = _get_ip(request)

    checks: list[tuple[str, int, int]] = []   # (key, window_ms, limit)

    if "ip" in policy:
        w, lim = policy["ip"]
        checks.append((f"rl:ip:{ip}", w * 1000, lim))

    if "ip_ep" in policy:
        w, lim = policy["ip_ep"]
        checks.append((f"rl:ip:{ip}:ep:{endpoint}", w * 1000, lim))

    if "ep" in policy:
        w, lim = policy["ep"]
        checks.append((f"rl:ep:{endpoint}", w * 1000, lim))

    if user_id and "user" in policy:
        w, lim = policy["user"]
        checks.append((f"rl:user:{user_id}", w * 1000, lim))

    if user_id and "user_ep" in policy:
        w, lim = policy["user_ep"]
        checks.append((f"rl:user:{user_id}:ep:{endpoint}", w * 1000, lim))

    for key, window_ms, limit in checks:
        allowed = await _check(key, window_ms, limit)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="لقد تجاوزت الحد المسموح به من الطلبات. حاول مجدداً لاحقاً.",
                headers={"Retry-After": str(window_ms // 1000)},
            )
