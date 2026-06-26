#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dalilak AI — FastAPI Backend v4 (Auth + Subscriptions + Admin)"""

import base64
import hashlib
import io
import json
import os
import secrets
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional

import httpx
import jwt as _jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PayloadSchemaType, PointIdsList, PointStruct, VectorParams,
)

# ═══════════════════════════════════════════════════════════════
#  DOCUMENT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(b64: str) -> str:
    try:
        import pdfplumber
        raw = base64.b64decode(b64)
        parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages[:20]:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)[:15000]
    except Exception:
        try:
            import fitz
            raw = base64.b64decode(b64)
            doc = fitz.open(stream=raw, filetype="pdf")
            parts = [doc[i].get_text() for i in range(min(20, len(doc)))]
            doc.close()
            return "\n\n".join(parts)[:15000]
        except Exception as e:
            return f"[تعذّر استخراج نص PDF: {e}]"

def extract_text_from_docx(b64: str) -> str:
    try:
        from docx import Document
        raw = base64.b64decode(b64)
        doc = Document(io.BytesIO(raw))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
        return "\n".join(lines)[:15000]
    except Exception as e:
        return f"[تعذّر استخراج نص Word: {e}]"

def extract_text_from_excel(b64: str) -> str:
    try:
        import openpyxl
        raw = base64.b64decode(b64)
        wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
        lines = []
        for sheet in wb.worksheets:
            lines.append(f"--- ورقة: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                vals = [str(v) if v is not None else "" for v in row]
                if any(v.strip() for v in vals):
                    lines.append(" | ".join(vals))
        return "\n".join(lines)[:15000]
    except Exception as e:
        return f"[تعذّر استخراج Excel: {e}]"

def extract_text_from_pptx(b64: str) -> str:
    try:
        from pptx import Presentation
        raw = base64.b64decode(b64)
        prs = Presentation(io.BytesIO(raw))
        lines = []
        for i, slide in enumerate(prs.slides, 1):
            lines.append(f"--- شريحة {i} ---")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    lines.append(shape.text.strip())
        return "\n".join(lines)[:15000]
    except Exception as e:
        return f"[تعذّر استخراج PowerPoint: {e}]"

def extract_text_from_csv(b64: str) -> str:
    try:
        import csv
        raw = base64.b64decode(b64).decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(raw))
        lines = [" | ".join(row) for row in reader if any(c.strip() for c in row)]
        return "\n".join(lines[:500])[:15000]
    except Exception as e:
        return f"[تعذّر قراءة CSV: {e}]"

def extract_text_from_zip(b64: str) -> str:
    try:
        import zipfile
        raw = base64.b64decode(b64)
        text_exts = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".py", ".js", ".ts"}
        lines = []
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            lines.append(f"محتويات الأرشيف ({len(zf.namelist())} ملف):")
            for name in zf.namelist()[:50]:
                lines.append(f"  - {name}")
            lines.append("")
            for name in zf.namelist()[:10]:
                ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext in text_exts:
                    try:
                        content = zf.read(name).decode("utf-8", errors="replace")[:3000]
                        lines.append(f"=== {name} ===\n{content}")
                    except Exception:
                        pass
        return "\n".join(lines)[:15000]
    except Exception as e:
        return f"[تعذّر فتح ZIP: {e}]"

async def transcribe_audio(b64: str, file_type: str, file_name: str) -> str:
    try:
        raw = base64.b64decode(b64)
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "mp3"
        fname = f"audio.{ext}"
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                transcript = await oai().audio.transcriptions.create(
                    model="whisper-1", file=(fname, f, file_type or "audio/mpeg"),
                    response_format="text"
                )
            return str(transcript)[:15000]
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return f"[تعذّر تحويل الصوت: {e}]"

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

COLLECTION     = "dalilak_ai_v2"
EMBED_MODEL    = "text-embedding-3-large"
VECTOR_DIM     = 3072
MODEL_FAST     = "gpt-4o-mini"
MODEL_SMART    = "gpt-4o"
MIN_SCORE      = 0.28
MAX_CTX        = 12
MAX_TOKENS     = 2000
MAX_HISTORY    = 6
MAX_CHARS      = 12000
MAX_DOC_TOKENS = 3500

# Auth config
JWT_SECRET   = os.environ.get("JWT_SECRET", "dalilak-secret-CHANGE-IN-PROD")
JWT_ALGO     = "HS256"
TRIAL_DAYS   = 3
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "dalilak-admin-CHANGE-IN-PROD")

# Qdrant collections for users & logs
USERS_COL    = "dalilak_users"
LOGS_COL     = "dalilak_logs"
RESETS_COL   = "dalilak_resets"
_users_ready = _logs_ready = _resets_ready = False

# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except Exception:
    SYSTEM_PROMPT = "أنت دليلك AI، مساعد المواطن اللبناني في كل الشؤون الحكومية."

# ═══════════════════════════════════════════════════════════════
#  LAZY CLIENTS
# ═══════════════════════════════════════════════════════════════

_oai: Optional[AsyncOpenAI] = None
_qdrant: Optional[QdrantClient] = None

def oai() -> AsyncOpenAI:
    global _oai
    if _oai is None:
        _oai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _oai

def qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=os.environ.get("QDRANT_URL", "").rstrip("/"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=30,
        )
    return _qdrant

def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "").rstrip("/")

def qdrant_headers() -> dict:
    return {"api-key": os.environ.get("QDRANT_API_KEY", ""), "Content-Type": "application/json"}

# ═══════════════════════════════════════════════════════════════
#  LRU ANSWER CACHE
# ═══════════════════════════════════════════════════════════════

_CACHE_MAX = 200
_cache: OrderedDict[str, dict] = OrderedDict()

def _ck(q: str, d: Optional[str]) -> str:
    return hashlib.md5(f"{q.strip().lower()}||{d or ''}".encode()).hexdigest()

def _cget(key: str) -> Optional[dict]:
    v = _cache.get(key)
    if v:
        _cache.move_to_end(key)
    return v

def _cset(key: str, val: dict) -> None:
    _cache[key] = val
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)

# ═══════════════════════════════════════════════════════════════
#  PASSWORD HELPERS
# ═══════════════════════════════════════════════════════════════

def hash_pw(password: str) -> str:
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{key.hex()}"

def verify_pw(password: str, stored: str) -> bool:
    try:
        salt, key_hex = stored.split(":", 1)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return key.hex() == key_hex
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════════
#  JWT HELPERS
# ═══════════════════════════════════════════════════════════════

def create_token(username: str, role: str = "user") -> str:
    payload = {
        "sub": username,
        "role": role,
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
    }
    return _jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def decode_token(token: str) -> dict:
    return _jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])

# ═══════════════════════════════════════════════════════════════
#  QDRANT USER STORE
# ═══════════════════════════════════════════════════════════════

def _ensure_users() -> None:
    global _users_ready
    if _users_ready:
        return
    q = qdrant()
    try:
        q.get_collection(USERS_COL)
    except Exception:
        q.create_collection(USERS_COL, vectors_config=VectorParams(size=4, distance=Distance.DOT))
        for f in ("username", "email", "plan", "active"):
            try:
                q.create_payload_index(USERS_COL, f, PayloadSchemaType.KEYWORD)
            except Exception:
                pass
    _users_ready = True

def _ensure_logs() -> None:
    global _logs_ready
    if _logs_ready:
        return
    q = qdrant()
    try:
        q.get_collection(LOGS_COL)
    except Exception:
        q.create_collection(LOGS_COL, vectors_config=VectorParams(size=1, distance=Distance.DOT))
    _logs_ready = True

def _ensure_resets() -> None:
    global _resets_ready
    if _resets_ready:
        return
    q = qdrant()
    try:
        q.get_collection(RESETS_COL)
    except Exception:
        q.create_collection(RESETS_COL, vectors_config=VectorParams(size=4, distance=Distance.DOT))
        try:
            q.create_payload_index(RESETS_COL, "username", PayloadSchemaType.KEYWORD)
            q.create_payload_index(RESETS_COL, "token", PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    _resets_ready = True

def _uid(username: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dalilak_user_{username}"))

def db_get_user(username: str) -> Optional[dict]:
    _ensure_users()
    results, _ = qdrant().scroll(
        collection_name=USERS_COL,
        scroll_filter=Filter(must=[FieldCondition(key="username", match=MatchValue(value=username))]),
        limit=1, with_payload=True,
    )
    return results[0].payload if results else None

def db_get_user_by_email(email: str) -> Optional[dict]:
    _ensure_users()
    results, _ = qdrant().scroll(
        collection_name=USERS_COL,
        scroll_filter=Filter(must=[FieldCondition(key="email", match=MatchValue(value=email.lower()))]),
        limit=1, with_payload=True,
    )
    return results[0].payload if results else None

def db_save_user(data: dict) -> None:
    _ensure_users()
    qdrant().upsert(
        collection_name=USERS_COL,
        points=[PointStruct(id=_uid(data["username"]), vector=[0.0] * 4, payload=data)],
    )

def db_list_users() -> list[dict]:
    _ensure_users()
    results, _ = qdrant().scroll(collection_name=USERS_COL, limit=500, with_payload=True)
    return [r.payload for r in results if r.payload]

def db_save_reset(username: str, token: str, expires_at: str) -> None:
    _ensure_resets()
    rid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"reset_{username}"))
    qdrant().upsert(
        collection_name=RESETS_COL,
        points=[PointStruct(id=rid, vector=[0.0] * 4,
                            payload={"username": username, "token": token, "expires_at": expires_at, "used": False})],
    )

def db_get_reset(token: str) -> Optional[dict]:
    _ensure_resets()
    results, _ = qdrant().scroll(
        collection_name=RESETS_COL,
        scroll_filter=Filter(must=[FieldCondition(key="token", match=MatchValue(value=token))]),
        limit=1, with_payload=True,
    )
    return results[0].payload if results else None

def db_mark_reset_used(username: str) -> None:
    _ensure_resets()
    rid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"reset_{username}"))
    try:
        qdrant().set_payload(collection_name=RESETS_COL, payload={"used": True}, points=[rid])
    except Exception:
        pass

def log_query(username: str, query_type: str, elapsed_ms: int) -> None:
    try:
        _ensure_logs()
        qdrant().upsert(
            collection_name=LOGS_COL,
            points=[PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(uuid.uuid4()))),
                vector=[1.0],
                payload={
                    "username": username,
                    "type": query_type,
                    "elapsed_ms": elapsed_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )],
        )
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════
#  AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════

_bearer = HTTPBearer(auto_error=False)

def _check_subscription(user: dict) -> None:
    """Raise 403 if trial expired and not paid."""
    plan = user.get("plan", "trial")
    if plan in ("paid", "admin", "guest"):
        return
    # trial: check expiry
    expires = user.get("trial_expires_at", "")
    if expires:
        try:
            exp_dt = datetime.fromisoformat(expires).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > exp_dt:
                raise HTTPException(
                    status_code=402,
                    detail="انتهت الفترة التجريبية — يرجى الترقية إلى الاشتراك المدفوع",
                )
        except HTTPException:
            raise
        except Exception:
            pass

async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    # ── GUEST ACCESS: no token required ──────────────────────────
    if not creds:
        return {"username": "guest", "plan": "guest", "active": True, "trial_expires_at": None}
    # ── AUTHENTICATED USER ────────────────────────────────────────
    try:
        payload = decode_token(creds.credentials)
        username = payload["sub"]
    except Exception:
        raise HTTPException(401, detail="جلسة منتهية — سجّل الدخول مجدداً")
    user = db_get_user(username)
    if not user or not user.get("active", True):
        raise HTTPException(401, detail="الحساب غير موجود أو معطّل")
    _check_subscription(user)
    return user

async def get_admin_user(user: dict = Depends(get_current_user)) -> dict:
    if user.get("plan") != "admin" and user.get("role") != "admin":
        raise HTTPException(403, detail="صلاحية المشرف مطلوبة")
    return user

def verify_admin_secret(x_admin_secret: Optional[str] = Header(None)) -> None:
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(403, detail="ADMIN_SECRET غير صحيح")

# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    domain: Optional[str] = None

class AnalyzeRequest(BaseModel):
    file_base64: str
    file_type: str
    file_name: str
    message: str = "حلل هذه الوثيقة واقترح الإجراءات المناسبة"
    history: list[Message] = []

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""
    phone: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class UpdateUserRequest(BaseModel):
    plan: Optional[str] = None
    active: Optional[bool] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    paid_until: Optional[str] = None

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""
    phone: str = ""
    plan: str = "trial"

class FileExtractRequest(BaseModel):
    file_base64: str
    file_type: str
    file_name: str

# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="Dalilak AI API", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

# ═══════════════════════════════════════════════════════════════
#  RAG HELPERS
# ═══════════════════════════════════════════════════════════════

async def embed(text: str) -> list:
    r = await oai().embeddings.create(
        model=EMBED_MODEL, input=[text[:MAX_CHARS]], dimensions=VECTOR_DIM,
    )
    return r.data[0].embedding

async def search_qdrant(vec: list, domain: Optional[str] = None) -> list:
    body: dict = {
        "vector": vec, "limit": MAX_CTX,
        "score_threshold": MIN_SCORE, "with_payload": True,
    }
    if domain:
        body["filter"] = {"must": [{"key": "domain", "match": {"value": domain}}]}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            f"{qdrant_url()}/collections/{COLLECTION}/points/search",
            headers=qdrant_headers(), json=body,
        )
    items = r.json().get("result", [])
    return [
        {
            "score":    round(x.get("score", 0), 3),
            "title":    x["payload"].get("title", ""),
            "text":     x["payload"].get("text", ""),
            "domain":   x["payload"].get("domain", ""),
            "ministry": x["payload"].get("ministry", ""),
            "website":  x["payload"].get("website", ""),
            "phone":    x["payload"].get("phone", ""),
            "fees":     x["payload"].get("fees", ""),
        }
        for x in items
    ]

def context_str(chunks: list) -> str:
    if not chunks:
        return ""
    parts = ["=== المعلومات المتاحة ==="]
    for i, c in enumerate(chunks, 1):
        parts.append(f"\n[{i}] {c['title']}")
        if c["ministry"]: parts.append(f"الجهة: {c['ministry']}")
        parts.append(c["text"])
        if c["website"]:  parts.append(f"الموقع: {c['website']}")
        if c["phone"]:    parts.append(f"الهاتف: {c['phone']}")
        parts.append("---")
    return "\n".join(parts)

def pick_model(msg: str) -> str:
    keywords = ["نموذج", "وثيقة", "خطوات", "إجراءات", "اشرح", "مقارن", "form", "document"]
    return MODEL_SMART if any(k in msg for k in keywords) or len(msg) > 200 else MODEL_FAST

def build_messages(ctx: str, history: list, user_msg: str) -> list:
    system = SYSTEM_PROMPT + (f"\n\n{ctx}" if ctx else "")
    msgs = [{"role": "system", "content": system}]
    for m in history[-MAX_HISTORY:]:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

# ═══════════════════════════════════════════════════════════════
#  PUBLIC ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "ok", "name": "Dalilak AI", "version": "4.1.0"}

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{qdrant_url()}/collections/{COLLECTION}", headers=qdrant_headers())
        pts = r.json().get("result", {}).get("points_count", 0)
        return {"status": "ok", "collection": COLLECTION, "points": pts}
    except Exception as e:
        raise HTTPException(503, detail=str(e))

@app.get("/ping")
async def ping():
    return {"pong": True}

# ═══════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def register(req: RegisterRequest):
    if len(req.username) < 3:
        raise HTTPException(400, detail="اسم المستخدم يجب أن يكون 3 أحرف على الأقل")
    if len(req.password) < 6:
        raise HTTPException(400, detail="كلمة المرور يجب أن تكون 6 أحرف على الأقل")
    if "@" not in req.email:
        raise HTTPException(400, detail="البريد الإلكتروني غير صالح")
    if db_get_user(req.username.lower()):
        raise HTTPException(409, detail="اسم المستخدم محجوز — اختر اسماً آخر")
    if db_get_user_by_email(req.email.lower()):
        raise HTTPException(409, detail="البريد الإلكتروني مسجّل مسبقاً")

    now = datetime.now(timezone.utc)
    trial_expires = (now + timedelta(days=TRIAL_DAYS)).isoformat()
    user = {
        "username":         req.username.lower(),
        "email":            req.email.lower(),
        "password_hash":    hash_pw(req.password),
        "full_name":        req.full_name,
        "phone":            req.phone,
        "plan":             "trial",
        "role":             "user",
        "active":           True,
        "trial_expires_at": trial_expires,
        "paid_until":       None,
        "created_at":       now.isoformat(),
        "last_login":       None,
    }
    db_save_user(user)
    token = create_token(req.username.lower())
    return {
        "token": token,
        "user": {
            "username":         user["username"],
            "email":            user["email"],
            "full_name":        user["full_name"],
            "plan":             user["plan"],
            "trial_expires_at": trial_expires,
        },
        "message": f"مرحباً! لديك {TRIAL_DAYS} أيام تجريبية مجانية.",
    }

@app.post("/auth/login")
async def login(req: LoginRequest):
    user = db_get_user(req.username.lower())
    if not user:
        user = db_get_user_by_email(req.username.lower())
    if not user:
        raise HTTPException(401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")
    if not user.get("active", True):
        raise HTTPException(403, detail="الحساب معطّل — تواصل مع الدعم")
    if not verify_pw(req.password, user.get("password_hash", "")):
        raise HTTPException(401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")

    user["last_login"] = datetime.now(timezone.utc).isoformat()
    db_save_user(user)
    token = create_token(user["username"], user.get("role", "user"))

    plan = user.get("plan", "trial")
    subscription_status = plan
    days_left = None
    if plan == "trial":
        try:
            exp = datetime.fromisoformat(user.get("trial_expires_at", "")).replace(tzinfo=timezone.utc)
            delta = (exp - datetime.now(timezone.utc)).days
            days_left = max(0, delta)
            if days_left == 0:
                subscription_status = "expired"
        except Exception:
            pass

    return {
        "token": token,
        "user": {
            "username":            user["username"],
            "email":               user["email"],
            "full_name":           user.get("full_name", ""),
            "plan":                plan,
            "role":                user.get("role", "user"),
            "trial_expires_at":    user.get("trial_expires_at"),
            "paid_until":          user.get("paid_until"),
            "subscription_status": subscription_status,
            "days_left":           days_left,
        },
    }

@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    plan = user.get("plan", "trial")
    days_left = None
    if plan == "trial":
        try:
            exp = datetime.fromisoformat(user.get("trial_expires_at", "")).replace(tzinfo=timezone.utc)
            days_left = max(0, (exp - datetime.now(timezone.utc)).days)
        except Exception:
            pass
    return {
        "username":         user["username"],
        "email":            user.get("email", ""),
        "full_name":        user.get("full_name", ""),
        "plan":             plan,
        "role":             user.get("role", "user"),
        "trial_expires_at": user.get("trial_expires_at"),
        "paid_until":       user.get("paid_until"),
        "days_left":        days_left,
        "created_at":       user.get("created_at"),
    }

@app.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    user = db_get_user_by_email(req.email.lower())
    if not user:
        return {"message": "إذا كان البريد مسجّلاً، ستتلقى رمز الاستعادة من الدعم الفني."}
    token = str(secrets.randbelow(900000) + 100000)
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    db_save_reset(user["username"], token, expires)
    return {
        "message": "تم إنشاء رمز الاستعادة — تواصل مع الدعم الفني للحصول عليه.",
        "info": "سيتواصل معك فريق الدعم عبر البريد الإلكتروني.",
    }

@app.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    if len(req.new_password) < 6:
        raise HTTPException(400, detail="كلمة المرور يجب أن تكون 6 أحرف على الأقل")
    reset = db_get_reset(req.token)
    if not reset:
        raise HTTPException(400, detail="رمز الاستعادة غير صحيح")
    if reset.get("used"):
        raise HTTPException(400, detail="رمز الاستعادة مستخدم مسبقاً")
    try:
        exp = datetime.fromisoformat(reset["expires_at"]).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            raise HTTPException(400, detail="رمز الاستعادة منتهي الصلاحية")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, detail="رمز غير صالح")
    user = db_get_user(reset["username"])
    if not user:
        raise HTTPException(400, detail="الحساب غير موجود")
    user["password_hash"] = hash_pw(req.new_password)
    db_save_user(user)
    db_mark_reset_used(reset["username"])
    return {"message": "تم تغيير كلمة المرور بنجاح — يمكنك تسجيل الدخول الآن."}

# ═══════════════════════════════════════════════════════════════
#  ADMIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/admin/users")
async def admin_list_users(user: dict = Depends(get_admin_user)):
    users = db_list_users()
    now = datetime.now(timezone.utc)
    result = []
    for u in users:
        plan = u.get("plan", "trial")
        status = plan
        days_left = None
        if plan == "trial":
            try:
                exp = datetime.fromisoformat(u.get("trial_expires_at", "")).replace(tzinfo=timezone.utc)
                dl = (exp - now).days
                days_left = max(0, dl)
                if dl < 0:
                    status = "expired"
            except Exception:
                pass
        result.append({
            "username":   u.get("username"),
            "email":      u.get("email"),
            "full_name":  u.get("full_name", ""),
            "phone":      u.get("phone", ""),
            "plan":       plan,
            "status":     status,
            "days_left":  days_left,
            "active":     u.get("active", True),
            "created_at": u.get("created_at"),
            "last_login": u.get("last_login"),
            "paid_until": u.get("paid_until"),
        })
    result.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"users": result, "total": len(result)}

@app.post("/admin/users")
async def admin_create_user(req: CreateUserRequest, user: dict = Depends(get_admin_user)):
    if db_get_user(req.username.lower()):
        raise HTTPException(409, detail="اسم المستخدم محجوز")
    if db_get_user_by_email(req.email.lower()):
        raise HTTPException(409, detail="البريد مسجّل مسبقاً")
    now = datetime.now(timezone.utc)
    trial_expires = (now + timedelta(days=TRIAL_DAYS)).isoformat()
    new_user = {
        "username":         req.username.lower(),
        "email":            req.email.lower(),
        "password_hash":    hash_pw(req.password),
        "full_name":        req.full_name,
        "phone":            req.phone,
        "plan":             req.plan,
        "role":             "admin" if req.plan == "admin" else "user",
        "active":           True,
        "trial_expires_at": trial_expires,
        "paid_until":       None,
        "created_at":       now.isoformat(),
        "last_login":       None,
    }
    db_save_user(new_user)
    return {"message": f"تم إنشاء المستخدم {req.username}", "user": new_user}

@app.put("/admin/users/{username}")
async def admin_update_user(
    username: str,
    req: UpdateUserRequest,
    admin: dict = Depends(get_admin_user),
):
    user = db_get_user(username.lower())
    if not user:
        raise HTTPException(404, detail="المستخدم غير موجود")
    if req.plan is not None:
        user["plan"] = req.plan
        if req.plan == "admin":
            user["role"] = "admin"
    if req.active is not None:
        user["active"] = req.active
    if req.full_name is not None:
        user["full_name"] = req.full_name
    if req.phone is not None:
        user["phone"] = req.phone
    if req.paid_until is not None:
        user["paid_until"] = req.paid_until
        user["plan"] = "paid"
    db_save_user(user)
    return {"message": "تم التحديث بنجاح", "user": user}

@app.delete("/admin/users/{username}")
async def admin_deactivate_user(username: str, admin: dict = Depends(get_admin_user)):
    user = db_get_user(username.lower())
    if not user:
        raise HTTPException(404, detail="المستخدم غير موجود")
    user["active"] = False
    db_save_user(user)
    return {"message": f"تم تعطيل حساب {username}"}

@app.get("/admin/stats")
async def admin_stats(admin: dict = Depends(get_admin_user)):
    users = db_list_users()
    now = datetime.now(timezone.utc)
    total = len(users)
    paid = sum(1 for u in users if u.get("plan") == "paid")
    trial_active = trial_expired = suspended = 0
    for u in users:
        plan = u.get("plan", "trial")
        if not u.get("active", True):
            suspended += 1
        elif plan == "trial":
            try:
                exp = datetime.fromisoformat(u.get("trial_expires_at", "")).replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > exp:
                    trial_expired += 1
                else:
                    trial_active += 1
            except Exception:
                trial_active += 1
    return {
        "total": total, "paid": paid,
        "trial_active": trial_active, "trial_expired": trial_expired,
        "suspended": suspended,
        "conversion_rate": f"{round(paid / total * 100, 1)}%" if total else "0%",
    }

@app.get("/admin/resets")
async def admin_list_resets(admin: dict = Depends(get_admin_user)):
    _ensure_resets()
    results, _ = qdrant().scroll(collection_name=RESETS_COL, limit=100, with_payload=True)
    codes = [r.payload for r in results if r.payload and not r.payload.get("used")]
    now = datetime.now(timezone.utc)
    active = []
    for c in codes:
        try:
            exp = datetime.fromisoformat(c["expires_at"]).replace(tzinfo=timezone.utc)
            if now < exp:
                active.append({
                    "username":   c["username"],
                    "token":      c["token"],
                    "expires_at": c["expires_at"],
                })
        except Exception:
            pass
    return {"reset_codes": active}

# ── Extend trial ─────────────────────────────────────────────
class ExtendTrialRequest(BaseModel):
    days: int = 7

@app.post("/admin/users/{username}/extend-trial")
async def admin_extend_trial(username: str, req: ExtendTrialRequest, admin: dict = Depends(get_admin_user)):
    user = db_get_user(username.lower())
    if not user:
        raise HTTPException(404, detail="المستخدم غير موجود")
    now = datetime.now(timezone.utc)
    # Extend from now or from current expiry, whichever is later
    current_exp_str = user.get("trial_expires_at") or now.isoformat()
    try:
        current_exp = datetime.fromisoformat(current_exp_str).replace(tzinfo=timezone.utc)
        base = max(now, current_exp)
    except Exception:
        base = now
    new_exp = (base + timedelta(days=req.days)).isoformat()
    user["trial_expires_at"] = new_exp
    if user.get("plan") not in ("paid", "admin"):
        user["plan"] = "trial"
    user["active"] = True
    db_save_user(user)
    return {"message": f"تم تمديد المهلة {req.days} أيام", "trial_expires_at": new_exp}

# ── Per-user logs ────────────────────────────────────────────
@app.get("/admin/users/{username}/logs")
async def admin_user_logs(username: str, admin: dict = Depends(get_admin_user)):
    _ensure_logs()
    results, _ = qdrant().scroll(
        collection_name=LOGS_COL,
        scroll_filter=Filter(must=[FieldCondition(key="username", match=MatchValue(value=username.lower()))]),
        limit=200, with_payload=True,
    )
    logs = sorted(
        [r.payload for r in results if r.payload],
        key=lambda x: x.get("timestamp", ""), reverse=True,
    )
    total = len(logs)
    chat_count = sum(1 for l in logs if "chat" in l.get("type", ""))
    analyze_count = sum(1 for l in logs if "analyze" in l.get("type", ""))
    avg_ms = int(sum(l.get("elapsed_ms", 0) for l in logs) / total) if total else 0
    return {
        "username": username,
        "total_queries": total,
        "chat_count": chat_count,
        "analyze_count": analyze_count,
        "avg_response_ms": avg_ms,
        "logs": logs[:50],  # last 50
    }

# ── All logs (general report) ────────────────────────────────
@app.get("/admin/logs")
async def admin_all_logs(admin: dict = Depends(get_admin_user)):
    _ensure_logs()
    results, _ = qdrant().scroll(collection_name=LOGS_COL, limit=1000, with_payload=True)
    logs = [r.payload for r in results if r.payload]
    total = len(logs)
    chat_count = sum(1 for l in logs if "chat" in l.get("type", ""))
    analyze_count = sum(1 for l in logs if "analyze" in l.get("type", ""))
    avg_ms = int(sum(l.get("elapsed_ms", 0) for l in logs) / total) if total else 0
    # Daily activity (last 14 days)
    now = datetime.now(timezone.utc)
    daily: dict = {}
    for l in logs:
        try:
            d = datetime.fromisoformat(l.get("timestamp", "")).replace(tzinfo=timezone.utc)
            if (now - d).days <= 13:
                key = d.strftime("%Y-%m-%d")
                daily[key] = daily.get(key, 0) + 1
        except Exception:
            pass
    # Top users
    user_counts: dict = {}
    for l in logs:
        u = l.get("username", "guest")
        user_counts[u] = user_counts.get(u, 0) + 1
    top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "total_queries": total,
        "chat_count": chat_count,
        "analyze_count": analyze_count,
        "avg_response_ms": avg_ms,
        "daily_activity": daily,
        "top_users": [{"username": u, "count": c} for u, c in top_users],
    }

# ── Knowledge base management ────────────────────────────────
class KnowledgeAddRequest(BaseModel):
    title: str
    text: str
    domain: str = ""
    ministry: str = ""
    website: str = ""
    phone: str = ""
    fees: str = ""
    source: str = "admin"

@app.post("/admin/knowledge/add")
async def admin_knowledge_add(req: KnowledgeAddRequest, admin: dict = Depends(get_admin_user)):
    if not req.title.strip() or not req.text.strip():
        raise HTTPException(400, detail="العنوان والنص مطلوبان")
    full_text = f"{req.title}\n{req.text}"
    vec = await embed(full_text)
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"admin_{req.title}_{req.text[:50]}"))
    payload = {
        "title":    req.title.strip(),
        "text":     req.text.strip(),
        "domain":   req.domain.strip(),
        "ministry": req.ministry.strip(),
        "website":  req.website.strip(),
        "phone":    req.phone.strip(),
        "fees":     req.fees.strip(),
        "source":   req.source,
        "added_by": admin.get("username"),
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    qdrant().upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id, vector=vec, payload=payload)],
    )
    return {"message": "تمت إضافة المعلومة للقاعدة بنجاح", "id": point_id}

@app.get("/admin/knowledge/search")
async def admin_knowledge_search(q: str, admin: dict = Depends(get_admin_user)):
    if not q.strip():
        raise HTTPException(400, detail="أدخل نص البحث")
    vec = await embed(q)
    chunks = await search_qdrant(