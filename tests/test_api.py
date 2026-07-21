"""
Dalilak AI — Backend API Tests
Tests critical endpoints with mocked external services.
Run: pytest tests/ -v
"""
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set minimal env vars so the app can import
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-testing-only")
os.environ.setdefault("ADMIN_SECRET", "test-admin-secret")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")

# We need to mock Qdrant and OpenAI before importing the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Create test client with mocked external services."""
    # Mock QdrantClient to avoid real connections
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.return_value = MagicMock(points_count=100)
    mock_qdrant.scroll.return_value = ([], None)
    mock_qdrant.upsert.return_value = None
    mock_qdrant.search.return_value = []

    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant):
        with patch("database.create_engine"):
            with patch("database.Base.metadata.create_all"):
                from main import app
                return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token(client):
    """Register a test user and return auth token."""
    # Mock the Qdrant user store
    with patch("main.db_get_user", return_value=None), \
         patch("main.db_get_user_by_email", return_value=None), \
         patch("main.db_save_user", return_value=None):
        resp = client.post("/auth/register", json={
            "username": "testuser123",
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User",
            "phone": "+961 1 234567",
        })
        if resp.status_code == 200:
            return resp.json()["token"]
    return None


# ── Health Endpoints ────────────────────────────────────────────

class TestHealth:
    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "Dalilak" in data["name"]

    def test_ping_returns_pong(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["pong"] is True

    def test_ready_endpoint_exists(self, client):
        # /ready should return 200 or 503, not 404
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)


# ── Auth Endpoints ──────────────────────────────────────────────

class TestAuth:
    def test_register_valid_user(self, client):
        with patch("main.db_get_user", return_value=None), \
             patch("main.db_get_user_by_email", return_value=None), \
             patch("main.db_save_user", return_value=None):
            resp = client.post("/auth/register", json={
                "username": "newuser456",
                "email": "new@example.com",
                "password": "password123",
                "full_name": "New User",
                "phone": "+961 1 111111",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "user" in data
        # CRITICAL: password_hash must NOT be in the response
        assert "password_hash" not in str(data)

    def test_register_duplicate_username(self, client):
        existing_user = {
            "username": "existinguser",
            "email": "other@example.com",
            "plan": "trial",
            "active": True,
        }
        with patch("main.db_get_user", return_value=existing_user):
            resp = client.post("/auth/register", json={
                "username": "existinguser",
                "email": "new2@example.com",
                "password": "password123",
            })
        assert resp.status_code == 409

    def test_register_invalid_email(self, client):
        resp = client.post("/auth/register", json={
            "username": "validuser",
            "email": "not-an-email",
            "password": "password123",
        })
        assert resp.status_code in (400, 422)

    def test_register_short_password(self, client):
        resp = client.post("/auth/register", json={
            "username": "validuser2",
            "email": "valid@example.com",
            "password": "123",
        })
        assert resp.status_code in (400, 422)

    def test_login_wrong_password(self, client):
        from main import hash_pw
        user = {
            "username": "user1",
            "email": "user1@example.com",
            "password_hash": hash_pw("correctpassword"),
            "plan": "trial",
            "role": "user",
            "active": True,
            "trial_expires_at": "2099-01-01T00:00:00",
        }
        with patch("main.db_get_user", return_value=user), \
             patch("main.db_get_user_by_email", return_value=None), \
             patch("main.db_save_user", return_value=None):
            resp = client.post("/auth/login", json={
                "username": "user1",
                "password": "wrongpassword",
            })
        assert resp.status_code == 401

    def test_login_inactive_user(self, client):
        from main import hash_pw
        user = {
            "username": "inactive",
            "email": "inactive@example.com",
            "password_hash": hash_pw("password123"),
            "plan": "trial",
            "active": False,
        }
        with patch("main.db_get_user", return_value=user):
            resp = client.post("/auth/login", json={
                "username": "inactive",
                "password": "password123",
            })
        assert resp.status_code == 403

    def test_forgot_password_unknown_email(self, client):
        """Should return success even for unknown emails (prevent user enumeration)."""
        with patch("main.db_get_user_by_email", return_value=None):
            resp = client.post("/auth/forgot-password", json={"email": "nobody@example.com"})
        assert resp.status_code == 200
        # Must not reveal whether email exists
        assert "مسجّلاً" in resp.json()["message"] or "الدعم" in resp.json()["message"]

    def test_me_endpoint_requires_auth(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_returns_user_without_password(self, client):
        from main import create_token, hash_pw
        user = {
            "username": "meuser",
            "email": "me@example.com",
            "password_hash": hash_pw("secret"),
            "plan": "trial",
            "role": "user",
            "active": True,
            "trial_expires_at": "2099-01-01T00:00:00",
        }
        token = create_token("meuser")
        with patch("main.db_get_user", return_value=user):
            resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "password_hash" not in data
        assert data["username"] == "meuser"


# ── Protected Endpoints ─────────────────────────────────────────

class TestProtectedEndpoints:
    def test_chat_requires_auth(self, client):
        resp = client.post("/chat", json={"message": "كيف أستخرج جواز سفر"})
        assert resp.status_code == 401

    def test_admin_requires_admin_role(self, client):
        from main import create_token, hash_pw
        # Regular user token
        user = {
            "username": "regularuser",
            "email": "reg@example.com",
            "password_hash": hash_pw("pass"),
            "plan": "trial",
            "role": "user",
            "active": True,
            "trial_expires_at": "2099-01-01T00:00:00",
        }
        token = create_token("regularuser", "user")
        with patch("main.db_get_user", return_value=user):
            resp = client.get("/admin/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_expired_token_rejected(self, client):
        import jwt as _jwt
        from datetime import datetime, timedelta, timezone
        payload = {
            "sub": "user1",
            "role": "user",
            "iat": int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp()),
            "exp": int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp()),
        }
        from main import JWT_SECRET, JWT_ALGO
        expired_token = _jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
        assert resp.status_code == 401


# ── Request Validation ──────────────────────────────────────────

class TestValidation:
    def test_chat_message_too_long_rejected(self, client):
        from main import create_token, hash_pw
        user = {
            "username": "vuser",
            "email": "v@example.com",
            "password_hash": hash_pw("pass"),
            "plan": "trial",
            "role": "user",
            "active": True,
            "trial_expires_at": "2099-01-01T00:00:00",
        }
        token = create_token("vuser")
        long_msg = "أ" * 2001
        with patch("main.db_get_user", return_value=user):
            resp = client.post("/chat",
                json={"message": long_msg},
                headers={"Authorization": f"Bearer {token}"}
            )
        assert resp.status_code == 422

    def test_empty_message_rejected(self, client):
        from main import create_token, hash_pw
        user = {
            "username": "vuser2",
            "email": "v2@example.com",
            "password_hash": hash_pw("pass"),
            "plan": "paid",
            "role": "user",
            "active": True,
        }
        token = create_token("vuser2")
        with patch("main.db_get_user", return_value=user):
            resp = client.post("/chat",
                json={"message": ""},
                headers={"Authorization": f"Bearer {token}"}
            )
        assert resp.status_code == 422


# ── Security ────────────────────────────────────────────────────

class TestSecurity:
    def test_response_does_not_leak_stack_trace(self, client):
        """Error responses should not contain Python stack traces."""
        resp = client.post("/auth/login", json={"username": "x", "password": "y"})
        body = resp.text
        assert "Traceback" not in body
        assert "File \"" not in body

    def test_register_no_password_in_response(self, client):
        """Registration response must never contain the password."""
        with patch("main.db_get_user", return_value=None), \
             patch("main.db_get_user_by_email", return_value=None), \
             patch("main.db_save_user", return_value=None):
            resp = client.post("/auth/register", json={
                "username": "sectest123",
                "email": "sec@test.com",
                "password": "mysecretpassword",
                "full_name": "Sec Test",
                "phone": "",
            })
        if resp.status_code == 200:
            body = resp.text
            assert "mysecretpassword" not in body
            assert "password_hash" not in body

    def test_feedback_validation(self, client):
        from main import create_token, hash_pw
        user = {
            "username": "fbuser",
            "email": "fb@example.com",
            "password_hash": hash_pw("pass"),
            "plan": "trial",
            "role": "user",
            "active": True,
            "trial_expires_at": "2099-01-01T00:00:00",
        }
        token = create_token("fbuser")
        with patch("main.db_get_user", return_value=user):
            resp = client.post("/feedback",
                json={"question": "test", "answer": "test", "rating": "invalid"},
                headers={"Authorization": f"Bearer {token}"}
            )
        assert resp.status_code == 400
