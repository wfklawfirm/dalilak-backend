# -*- coding: utf-8 -*-
"""
Phase 17 — My Procedures Workspace — Tier A (offline, TestClient + JWT mock)

Tests verify: auth guard, CRUD lifecycle, checklist completion tracking.
Uses a fake test JWT signed with the test secret and patches db_get_user
so that no Qdrant connection is made.

SECURITY: No real user accounts. JWT signed with test secret only.
Test secret must NOT match any production secret.
"""
from __future__ import annotations

import os
import sys
import secrets
import warnings
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, AsyncMock

# Must set env vars BEFORE importing main.
os.environ["JWT_SECRET"] = os.environ.get("JWT_SECRET", "test-secret-32-chars-minimum-ok!")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.pop("ADMIN_USERNAME", None)
os.environ.pop("REDIS_URL", None)  # keep Redis unconfigured → JTI check is in-memory only

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

warnings.filterwarnings("ignore", category=DeprecationWarning)

import jwt as _jwt  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# ── JWT helper ─────────────────────────────────────────────────────────────────

_JWT_SECRET = os.environ.get("JWT_SECRET", "test-secret-32-chars-minimum-ok!")
_JWT_ALGO = "HS256"

_TEST_USERNAME = "test_p17_user"
_TEST_USER = {
    "username": _TEST_USERNAME,
    "email": "p17test@test.local",
    "plan": "paid",
    "role": "user",
    "active": True,
    "trial_expires_at": None,
    "paid_until": None,
    "full_name": "Test P17",
    "phone": "",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "last_login": None,
}


def _make_token(username: str = _TEST_USERNAME, role: str = "user") -> str:
    """Create a signed HS256 JWT for testing. Never use the production secret."""
    now = datetime.now(timezone.utc)
    payload = {
        "iss": "dalilak-ai",
        "sub": username,
        "role": role,
        "jti": secrets.token_hex(8),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=7)).timestamp()),
    }
    return _jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGO)


def _auth_headers(token: str | None = None) -> dict:
    tok = token or _make_token()
    return {"Authorization": f"Bearer {tok}"}


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_client_with_mock_user() -> TestClient:
    """Return a TestClient with db_get_user patched to return the test user."""
    # Clear per-test state
    if _TEST_USERNAME in main._user_procedures:
        del main._user_procedures[_TEST_USERNAME]
    return TestClient(main.app, raise_server_exceptions=True)


# ── T17-01: GET /my-procedures without token returns 401 ─────────────────────

def test_p17_01_get_my_procedures_no_token_returns_401():
    client = TestClient(main.app, raise_server_exceptions=True)
    r = client.get("/my-procedures")
    # Without a token, user is "guest" which is blocked by _require_auth
    assert r.status_code == 401, f"Expected 401, got {r.status_code}: {r.text}"


# ── T17-02: POST /my-procedures without token returns 401 ────────────────────

def test_p17_02_post_my_procedures_no_token_returns_401():
    client = TestClient(main.app, raise_server_exceptions=True)
    r = client.post("/my-procedures", json={"procedure_id": "passport-renewal-lb"})
    assert r.status_code == 401, f"Expected 401, got {r.status_code}: {r.text}"


# ── T17-03: POST with valid token creates item with checklist ────────────────

def test_p17_03_create_my_procedure_with_valid_token():
    if _TEST_USERNAME in main._user_procedures:
        del main._user_procedures[_TEST_USERNAME]

    with patch("main.db_get_user", return_value=_TEST_USER):
        client = TestClient(main.app, raise_server_exceptions=True)
        r = client.post(
            "/my-procedures",
            json={"procedure_id": "passport-renewal-lb"},
            headers=_auth_headers(),
        )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["procedure_id"] == "passport-renewal-lb"
    assert data["status"] == "active"
    assert "checklist" in data
    assert len(data["checklist"]) >= 4, f"Expected >= 4 checklist steps, got {len(data['checklist'])}"
    assert data["completion_pct"] == 0
    # All checklist items should start as not done
    for step in data["checklist"]:
        assert step["done"] is False


# ── T17-04: GET /my-procedures with valid token returns list ─────────────────

def test_p17_04_list_includes_created_item():
    if _TEST_USERNAME in main._user_procedures:
        del main._user_procedures[_TEST_USERNAME]

    with patch("main.db_get_user", return_value=_TEST_USER):
        client = TestClient(main.app, raise_server_exceptions=True)
        headers = _auth_headers()

        # Create
        r_create = client.post(
            "/my-procedures",
            json={"procedure_id": "passport-renewal-lb", "notes": "test note"},
            headers=headers,
        )
        assert r_create.status_code == 200
        created_id = r_create.json()["id"]

        # List
        r_list = client.get("/my-procedures", headers=headers)

    assert r_list.status_code == 200
    data = r_list.json()
    assert "procedures" in data
    ids = [p["id"] for p in data["procedures"]]
    assert created_id in ids, f"Created item {created_id} not in list: {ids}"
    assert data["count"] >= 1


# ── T17-05: PUT updates completed_steps and recalculates completion_pct ──────

def test_p17_05_put_updates_checklist_and_pct():
    if _TEST_USERNAME in main._user_procedures:
        del main._user_procedures[_TEST_USERNAME]

    with patch("main.db_get_user", return_value=_TEST_USER):
        client = TestClient(main.app, raise_server_exceptions=True)
        headers = _auth_headers()

        # Create a passport procedure (4 steps)
        r_create = client.post(
            "/my-procedures",
            json={"procedure_id": "passport-renewal-lb"},
            headers=headers,
        )
        assert r_create.status_code == 200
        proc = r_create.json()
        proc_id = proc["id"]
        total_steps = len(proc["checklist"])

        # Mark step 1 and 2 as done
        r_update = client.put(
            f"/my-procedures/{proc_id}",
            json={"completed_steps": [1, 2]},
            headers=headers,
        )

    assert r_update.status_code == 200
    updated = r_update.json()
    expected_pct = round((2 / total_steps) * 100)
    assert updated["completion_pct"] == expected_pct, (
        f"Expected {expected_pct}%, got {updated['completion_pct']}%"
    )
    done_steps = [s for s in updated["checklist"] if s["done"]]
    assert len(done_steps) == 2


# ── T17-06: DELETE removes the item ──────────────────────────────────────────

def test_p17_06_delete_removes_item():
    if _TEST_USERNAME in main._user_procedures:
        del main._user_procedures[_TEST_USERNAME]

    with patch("main.db_get_user", return_value=_TEST_USER):
        client = TestClient(main.app, raise_server_exceptions=True)
        headers = _auth_headers()

        # Create
        r_create = client.post(
            "/my-procedures",
            json={"procedure_id": "birth-registration-lb"},
            headers=headers,
        )
        assert r_create.status_code == 200
        proc_id = r_create.json()["id"]

        # Delete
        r_del = client.delete(f"/my-procedures/{proc_id}", headers=headers)
        assert r_del.status_code == 200

        # Verify it's gone
        r_list = client.get("/my-procedures", headers=headers)

    assert r_list.status_code == 200
    ids = [p["id"] for p in r_list.json()["procedures"]]
    assert proc_id not in ids, f"Deleted item {proc_id} still in list: {ids}"
