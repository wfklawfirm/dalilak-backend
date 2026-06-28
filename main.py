#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dalilak AI — FastAPI Backend v4 (Auth + Subscriptions + Admin)"""

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import secrets
import time
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional

import httpx
import jwt as _jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PayloadSchemaType, PointIdsList, PointStruct, VectorParams,
)
from retrieval_service import RetrievalService
from database import init_db, db_session, repo
from risk_service import compute_risk
from document_service import analyze_document, review_contract

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

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

COLLECTION     = "dalilak_ai_v2"
EMBED_MODEL    = "text-embedding-3-large"
VECTOR_DIM     = 3072
MODEL_FAST     = "gpt-4o-mini"
MODEL_SMART    = "gpt-4o"
MIN_SCORE      = 0.26
MAX_CTX        = 15
MAX_TOKENS     = 3200
MAX_HISTORY    = 6
MAX_CHARS      = 16000
MAX_DOC_TOKENS = 4000

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
_retrieval: Optional[RetrievalService] = None

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

def retrieval() -> RetrievalService:
    """Lazy-initialised synchronous RetrievalService (wraps qdrant() + sync OpenAI)."""
    global _retrieval
    if _retrieval is None:
        from openai import OpenAI as SyncOpenAI
        _sync_oai = SyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        _retrieval = RetrievalService(
            qdrant_client=qdrant(),
            openai_client=_sync_oai,
            collection=COLLECTION,
            embed_model=EMBED_MODEL,
            embed_dim=VECTOR_DIM,
        )
    return _retrieval

def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "").rstrip("/")

def qdrant_headers() -> dict:
    return {"api-key": os.environ.get("QDRANT_API_KEY", ""), "Content-Type": "application/json"}

# ── Persistent async HTTP client for Qdrant (reuse connections) ──
_http: Optional[httpx.AsyncClient] = None

def http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=20,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
        )
    return _http

# ═══════════════════════════════════════════════════════════════
#  QUERY NORMALIZER  (strips diacritics + unifies whitespace)
# ═══════════════════════════════════════════════════════════════

_AR_DIACRITICS = re.compile(
    r'[ؐ-ًؚ-ٰٟ'
    r'ۖ-ۜ۟-۪ۤۧۨ-ۭ]'
)

def _normalize(text: str) -> str:
    """Strip Arabic diacritics, NFKC-normalize, collapse whitespace, lowercase."""
    t = _AR_DIACRITICS.sub('', text)
    t = unicodedata.normalize('NFKC', t)
    t = re.sub(r'\s+', ' ', t).strip().lower()
    return t

# ═══════════════════════════════════════════════════════════════
#  LRU ANSWER CACHE  (500 entries, normalize-keyed)
# ═══════════════════════════════════════════════════════════════

_CACHE_MAX = 500
_cache: OrderedDict[str, dict] = OrderedDict()

def _ck(q: str, d: Optional[str]) -> str:
    return hashlib.md5(f"{_normalize(q)}||{d or ''}".encode()).hexdigest()

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
#  EMBEDDING VECTOR CACHE  (avoid repeat OpenAI embed calls)
# ═══════════════════════════════════════════════════════════════

_EMCACHE_MAX = 500
_emcache: OrderedDict[str, list] = OrderedDict()

def _ek(text: str) -> str:
    return hashlib.md5(_normalize(text).encode()).hexdigest()

def _emget(key: str) -> Optional[list]:
    v = _emcache.get(key)
    if v is not None:
        _emcache.move_to_end(key)
    return v

def _emset(key: str, vec: list) -> None:
    _emcache[key] = vec
    _emcache.move_to_end(key)
    while len(_emcache) > _EMCACHE_MAX:
        _emcache.popitem(last=False)

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
    if plan == "paid":
        return
    if plan == "admin":
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
    if not creds:
        raise HTTPException(401, detail="يجب تسجيل الدخول أولاً")
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
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default=[], max_length=20)
    domain: Optional[str] = Field(default=None, max_length=100)
    transaction_id: Optional[str] = Field(default=None, max_length=36)
    document_ids: Optional[list[str]] = Field(default=None, max_length=10)
    document_context_mode: Optional[str] = Field(default=None, max_length=20)

class AnalyzeRequest(BaseModel):
    file_base64: str = Field(..., max_length=25_000_000)  # ~18 MB raw
    file_type: str = Field(..., max_length=200)           # OOXML types can be 70+ chars
    file_name: str = Field(..., max_length=255)
    message: str = Field(default="حلل هذه الوثيقة واقترح الإجراءات المناسبة", max_length=500)
    history: list[Message] = Field(default=[], max_length=20)

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_.-]+$')
    email: str = Field(..., max_length=200)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=150)
    phone: str = Field(default="", max_length=30)

    @field_validator('username')
    @classmethod
    def username_no_spaces(cls, v: str) -> str:
        return v.strip().lower()

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=200)   # username OR email
    password: str = Field(..., min_length=1, max_length=128)

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class UpdateUserRequest(BaseModel):
    plan: Optional[str] = None          # trial | paid | admin | suspended
    active: Optional[bool] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    paid_until: Optional[str] = None    # ISO date

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""
    phone: str = ""
    plan: str = "trial"

# ── Transaction File Models ───────────────────────────────────────────────────

class CreateTransactionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    procedure_slug: Optional[str] = Field(default=None, max_length=100)
    country: Optional[str] = Field(default="lebanon", max_length=20)
    user_type: Optional[str] = Field(default=None, max_length=50)
    summary: Optional[str] = Field(default=None, max_length=2000)
    notes: Optional[str] = Field(default=None, max_length=2000)
    required_documents: Optional[list] = None
    steps: Optional[list] = None
    sources: Optional[list] = None

class UpdateTransactionRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=500)
    status: Optional[str] = Field(default=None, max_length=20)
    summary: Optional[str] = Field(default=None, max_length=2000)
    notes: Optional[str] = Field(default=None, max_length=2000)
    procedure_slug: Optional[str] = Field(default=None, max_length=100)
    required_documents: Optional[list] = None
    missing_documents: Optional[list] = None
    steps: Optional[list] = None
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    risk_reasons: Optional[list] = None
    next_actions: Optional[list] = None
    sources: Optional[list] = None

# ── Document Upload Model (with transaction_id) ───────────────────────────────

class UploadDocumentRequest(BaseModel):
    file_base64: str = Field(..., max_length=25_000_000)
    file_type: str = Field(..., max_length=100)
    file_name: str = Field(..., max_length=255)
    transaction_id: Optional[str] = Field(default=None, max_length=36)

_ALLOWED_MIME_TYPES = frozenset({
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "image/jpeg",
    "image/png",
    "image/jpg",
})

# ═══════════════════════════════════════════════════════════════
#  STARTUP PRELOADER  (warm embedding + answer cache)
# ═══════════════════════════════════════════════════════════════

_PRELOAD_QUESTIONS = [
    "كيف أستخرج جواز سفر لبناني",
    "كيف أستخرج بطاقة هوية لبنانية",
    "كيف أسجل سيارة جديدة",
    "كيف أستخرج شهادة ميلاد",
    "كيف أسجل شركة في لبنان",
    "كيف أستخرج تصريح بناء",
    "كيف أجدد رخصة القيادة",
    "كيف أسجل الزواج الرسمي",
    "كيف أنقل ملكية عقار",
    "كيف أستخرج شهادة عدم محكومية",
    "كيف أسجل مولود",
    "كيف أحصل على إجازة مزاولة مهنة",
    "ما رسوم التسجيل في الضمان الاجتماعي",
    "كيف أطعن في قرار إداري",
    "كيف أجدد إقامة الأجانب في لبنان",
]

async def _preload() -> None:
    """Warm embedding + answer cache for the 15 most common questions at startup."""
    await asyncio.sleep(4)  # Let server fully start first
    for q in _PRELOAD_QUESTIONS:
        try:
            ck = _ck(q, None)
            if _cget(ck):
                continue  # Already cached
            qinfo  = classify_query(q)
            chunks = await retrieve_multi(q, qinfo, None)
            chunks = rerank_chunks(chunks, q)
            ctx    = context_str(chunks)
            model  = pick_model(q, qinfo)
            msgs   = build_messages(ctx, [], q, qinfo)
            resp   = await oai().chat.completions.create(
                model=model, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.1,
            )
            result = {
                "answer":      resp.choices[0].message.content,
                "model":       model,
                "chunks_used": len(chunks),
                "elapsed_s":   0.0,
                "query_type":  qinfo['type'],
                "sources":     [{"title": c["title"], "ministry": c["ministry"], "score": c["score"]} for c in chunks[:5]],
            }
            _cset(ck, result)
        except Exception:
            pass

@asynccontextmanager
async def lifespan(app_: FastAPI):
    asyncio.create_task(_preload())
    yield
    # Cleanup: close persistent HTTP client
    global _http
    if _http and not _http.is_closed:
        await _http.aclose()

# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="Dalilak AI API", version="4.0.0", lifespan=lifespan)

# ── CORS — restrict to known origins in production ────────────────────────────
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://dalilak-frontend.vercel.app,https://dalilak.ai,https://www.dalilak.ai"
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Secret"],
    allow_credentials=True,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
from collections import defaultdict

_rate_buckets: dict[str, list[float]] = defaultdict(list)

def _rate_limit(key: str, max_requests: int = 60, window_seconds: int = 60) -> None:
    """Sliding window rate limiter. Raises 429 when exceeded."""
    now = time.time()
    bucket = _rate_buckets[key]
    # Evict expired timestamps
    _rate_buckets[key] = [t for t in bucket if now - t < window_seconds]
    if len(_rate_buckets[key]) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail="طلبات كثيرة جداً — حاول مرة أخرى بعد قليل.",
            headers={"Retry-After": str(window_seconds)},
        )
    _rate_buckets[key].append(now)

async def _apply_rate_limit(request: Request) -> None:
    """FastAPI dependency: rate-limit by IP + path."""
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    _rate_limit(f"{client_ip}:{path}", max_requests=30, window_seconds=60)

# ═══════════════════════════════════════════════════════════════
#  RAG HELPERS
# ═══════════════════════════════════════════════════════════════

async def embed(text: str) -> list:
    ek = _ek(text)
    cached_vec = _emget(ek)
    if cached_vec is not None:
        return cached_vec
    r = await oai().embeddings.create(
        model=EMBED_MODEL, input=[text[:MAX_CHARS]], dimensions=VECTOR_DIM,
    )
    vec = r.data[0].embedding
    _emset(ek, vec)
    return vec

async def search_qdrant(vec: list, domain: Optional[str] = None) -> list:
    body: dict = {
        "vector": vec, "limit": MAX_CTX,
        "score_threshold": MIN_SCORE, "with_payload": True,
    }
    if domain:
        body["filter"] = {"must": [{"key": "domain", "match": {"value": domain}}]}
    r = await http().post(
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

# ═══════════════════════════════════════════════════════════════
#  QUERY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

# Patterns by question type (Arabic + English)
_PATTERNS = {
    'draft': re.compile(
        r'حضّر|حضر\s|أعد\s|اعد\s|اكتب\s|اكتبي\s|اصغ|صِغ|صغ\s|انشئ|أنشئ|ضع\s|ضعي\s|'
        r'اعطني\s+مسودة|اعطيني\s+مسودة|أعطني\s+مسودة|'
        r'مسودة\s|نموذج\s+طلب|نموذج\s+رسالة|'
        r'إنذار\s+بالإخلاء|انذار\s+بالاخلاء|طلب\s+إخلاء|طلب\s+اخلاء|'
        r'رسالة\s+رسمية|طلب\s+رسمي|وكالة\s+قانونية|'
        r'عقد\s+إيجار|عقد\s+بيع|محضر\s+تسليم|اتفاقية|تعهد\s+خطي|'
        r'إقرار\s+بالاستلام|براءة\s+ذمة|'
        r'draft\s|prepare a|write a|draw up|compose a', re.I),
    'comparative': re.compile(
        r'فرق|مقارنة|أفضل|أحسن|أسرع|أرخص|بدلاً|عوضاً|difference|compare|vs\b|versus|better|'
        r'أم\s|ام\s|أو\s.*\sأو', re.I),
    'legal': re.compile(
        r'مادة|قانون|مرسوم|نظام|قرار وزاري|شريعة|تشريع|نص قانوني|حق قانوني|'
        r'article|law|decree|legislation|legal right|statute|regulation', re.I),
    'eligibility': re.compile(
        r'هل أستطيع|هل يمكنني|هل أحق|هل يحق لي|هل أستحق|مؤهل|شروط الاستفادة|'
        r'can i|am i eligible|do i qualify|entitled to', re.I),
    'procedural': re.compile(
        r'كيف|خطوات|إجراءات|طريقة|آلية|مراحل|ماذا أفعل|ما الذي يجب|'
        r'how (to|do|can)|steps|procedure|process|what do i (need|have to)', re.I),
    'factual': re.compile(
        r'كم|رسوم|تكلفة|سعر|مدة|وقت|متى|أين|موقع|عنوان|هاتف|'
        r'how much|fee|cost|price|duration|how long|when|where|address|phone', re.I),
}

_MULTI_SPLIT = re.compile(r'[،,؛;]\s*|\bو(أيضاً|كذلك|أيضا)?\b|\bوكيف\b|\bوماذا\b|\bوهل\b|\bوما\b', re.I)


def classify_query(msg: str) -> dict:
    """
    Returns:
      type       : 'draft' | 'comparative' | 'legal' | 'eligibility' | 'procedural' | 'factual' | 'general'
      multipart  : bool — question has ≥2 distinct sub-questions
      complexity : 0-10 — drives model choice and retrieval depth
    """
    msg_n = _normalize(msg)

    qtype = 'general'
    for t, pat in _PATTERNS.items():
        if pat.search(msg):
            qtype = t
            break

    # Count distinct question words — proxy for sub-questions
    q_words = re.findall(r'\b(كيف|ما|هل|متى|أين|من|لماذا|كم|how|what|when|where|why|who|which)\b', msg, re.I)
    is_multipart = len(q_words) >= 2

    word_count = len(msg.split())
    complexity = min(10, word_count // 6 + len(q_words) + (3 if is_multipart else 0) +
                     (2 if qtype in ('legal', 'comparative') else 0))

    return {'type': qtype, 'multipart': is_multipart, 'complexity': complexity}


def split_subqueries(msg: str) -> list[str]:
    """Split a multi-part question into sub-queries for parallel retrieval."""
    parts = _MULTI_SPLIT.split(msg)
    # Keep parts that look like full questions (≥4 words)
    meaningful = [p.strip() for p in parts if len(p.split()) >= 4]
    return meaningful[:3] if meaningful else [msg]


async def retrieve_multi(msg: str, qinfo: dict, domain: Optional[str]) -> list[dict]:
    """
    Multi-query retrieval:
    - Always retrieves for the full question
    - If multi-part, also retrieves for each sub-question
    - Deduplicates by title, keeps top MAX_CTX by score
    """
    queries = [msg]
    if qinfo['multipart'] and qinfo['complexity'] >= 4:
        queries += split_subqueries(msg)

    seen: dict[str, dict] = {}
    tasks = [embed(q) for q in queries]
    vecs = await asyncio.gather(*tasks, return_exceptions=True)

    search_tasks = []
    for vec in vecs:
        if isinstance(vec, list):
            search_tasks.append(search_qdrant(vec, domain))

    results = await asyncio.gather(*search_tasks, return_exceptions=True)
    for chunks in results:
        if isinstance(chunks, list):
            for c in chunks:
                title = c['title']
                # Keep the highest-scoring version of each chunk
                if title not in seen or c['score'] > seen[title]['score']:
                    seen[title] = c

    merged = list(seen.values())
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged[:MAX_CTX]


def rerank_chunks(chunks: list[dict], query: str) -> list[dict]:
    """
    Keyword-overlap reranker on top of vector similarity.
    Pulls up chunks whose text/title contains more query keywords.
    """
    # Arabic words ≥3 chars + English words ≥4 chars
    keywords = set(re.findall(r'[؀-ۿ]{3,}|[a-zA-Z]{4,}', query.lower()))
    for c in chunks:
        text = (c.get('title', '') + ' ' + c.get('text', '')).lower()
        hits = sum(1 for kw in keywords if kw in text)
        c['_rr'] = c['score'] + hits * 0.008   # gentle boost
    chunks.sort(key=lambda x: x.get('_rr', x['score']), reverse=True)
    return chunks


def type_hint(qinfo: dict) -> str:
    """Inject type-specific answering instructions into the system prompt."""
    hints = {
        'draft':       "\n\n✍️ نوع الطلب: **إعداد مسودة** — تعليمات إلزامية:\n"
                       "1. **أنشئ المسودة فوراً** — لا تشرح الإجراءات، اكتب النص الكامل للوثيقة مباشرة\n"
                       "2. ابدأ بـ: ═══════════════════════════════\n"
                       "            مسودة أولية — [نوع الوثيقة]\n"
                       "            ═══════════════════════════════\n"
                       "3. استخدم صيغة رسمية قانونية مناسبة للقانون اللبناني\n"
                       "4. ضع [الاسم الكامل] أو [التاريخ] أو [العنوان] كـ placeholder لأي بيانات ناقصة\n"
                       "5. تضمّن جميع البنود والصياغات القانونية اللازمة لصحة الوثيقة\n"
                       "6. أنهِ المسودة بـ:\n"
                       "            ─────────────────────────────\n"
                       "            نهاية المسودة\n"
                       "            ─────────────────────────────\n"
                       "7. أضف بعدها فقرة قصيرة: 'ملاحظة: هذه مسودة أولية للإرشاد فقط. يُنصح بمراجعة محامٍ مرخّص قبل الاستخدام الرسمي.'\n"
                       "8. لا تضمن نتائج قانونية. لا تخترع أرقاماً أو جهات رسمية.",
        'procedural':  "\n\n📋 نوع السؤال: **إجرائي** — الإجابة الإلزامية:\n"
                       "  • ابدأ بنظرة عامة من جملتين\n"
                       "  • اذكر المستندات المطلوبة في قائمة\n"
                       "  • اشرح الخطوات مرقّمةً بالترتيب الزمني الصحيح\n"
                       "  • اذكر الرسوم والمدة الزمنية والجهة المختصة",
        'legal':       "\n\n⚖️ نوع السؤال: **قانوني** — الإجابة الإلزامية:\n"
                       "  • اذكر المادة والمرسوم أو القانون بالرقم الدقيق\n"
                       "  • ميّز بين نص القانون وتطبيقه الفعلي\n"
                       "  • إذا تعدّدت المراجع القانونية رتّبها من الأعلى إلى الأدنى في التسلسل\n"
                       "  • نبّه إلى أي تعديلات أو استثناءات",
        'comparative': "\n\n📊 نوع السؤال: **مقارن** — الإجابة الإلزامية:\n"
                       "  • افتح بجدول مقارنة أو نقاط موازية للخيارين/الخيارات\n"
                       "  • اذكر الفروق العملية (الوقت، التكلفة، الجهة، الشروط)\n"
                       "  • اختم بتوصية واضحة حسب الحالة الأكثر شيوعاً",
        'eligibility': "\n\n✅ نوع السؤال: **أهلية/استحقاق** — الإجابة الإلزامية:\n"
                       "  • اذكر الشروط الكاملة مرقّمةً\n"
                       "  • اذكر الحالات المستثناة\n"
                       "  • اختم بإجابة صريحة: نعم / لا / يعتمد على (مع توضيح)\n"
                       "  • اذكر الجهة المختصة لتقديم الطلب",
        'factual':     "\n\n📌 نوع السؤال: **معلوماتي** — ابدأ بالإجابة المباشرة (رقم / تاريخ / جهة) ثم الشرح.",
        'general':     "",
    }
    multipart = "\n\n📑 السؤال متعدد الأجزاء — أجب على **كل جزء بشكل مستقل** تحت عنوان فرعي واضح." \
                if qinfo.get('multipart') else ""
    return hints.get(qinfo.get('type', 'general'), '') + multipart


def context_str(chunks: list) -> str:
    if not chunks:
        return "[ملاحظة للنموذج: لم يُعثر على معلومات محددة في قاعدة البيانات. " \
               "أجب بناءً على معرفتك الموثوقة بالقانون اللبناني، " \
               "واذكر صراحةً أن المعلومات من معرفة عامة لا من قاعدة البيانات الرسمية.]"

    max_score = max(c['score'] for c in chunks)
    low_conf = max_score < 0.31

    parts = []
    if low_conf:
        parts.append("[تنبيه: الصلة بين السؤال والبيانات المتوفرة منخفضة. "
                     "استخدم المعلومات بحذر وأشر إلى أي غموض في نهاية إجابتك.]")
    parts.append("=== قاعدة البيانات — المعلومات ذات الصلة ===")
    for i, c in enumerate(chunks, 1):
        parts.append(f"\n[{i}] {c['title']}")
        if c.get("ministry"):        parts.append(f"الجهة المختصة: {c['ministry']}")
        if c.get("category"):        parts.append(f"القطاع: {c['category']}")
        parts.append(c["text"])
        if c.get("fees"):            parts.append(f"الرسوم: {c['fees']}")
        if c.get("processing_time"): parts.append(f"مدة الإنجاز: {c['processing_time']}")
        if c.get("website"):         parts.append(f"الموقع: {c['website']}")
        if c.get("phone"):           parts.append(f"الهاتف: {c['phone']}")
        parts.append("---")
    return "\n".join(parts)


def pick_model(msg: str, qinfo: Optional[dict] = None) -> str:
    """Route to fast or smart model. Smart (gpt-4o) for all real questions."""
    simple = ["مرحبا", "أهلا", "شكرا", "hello", "hi", "كيفك", "كيف حالك", "شو أخبارك"]
    if len(msg) < 35 and any(s in msg.lower() for s in simple):
        return MODEL_FAST
    return MODEL_SMART


def build_messages(ctx: str, history: list, user_msg: str,
                   qinfo: Optional[dict] = None) -> list:
    hint = type_hint(qinfo) if qinfo else ""
    base_reminder = (
        "\n\n⚡ تعليمات إلزامية لهذه الإجابة:\n"
        "1. اذكر المواد القانونية والمراسيم ذات الصلة في قسم \"📚 الأساس القانوني\"\n"
        "2. كن دقيقاً وشاملاً — لا تحذف خطوة أو وثيقة\n"
        "3. لا تخترع أي معلومة غير موجودة في السياق أو في معرفتك المؤكدة\n"
        "4. اختم بـ ⚠️ إذا كانت هناك شروط استثنائية أو تحذيرات مهمة"
    )
    system = SYSTEM_PROMPT + hint + base_reminder + (f"\n\n{ctx}" if ctx else "")
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
    return {"status": "ok", "name": "Dalilak AI", "version": "4.0.0"}

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
    # Validate input
    if len(req.username) < 3:
        raise HTTPException(400, detail="اسم المستخدم يجب أن يكون 3 أحرف على الأقل")
    if len(req.password) < 6:
        raise HTTPException(400, detail="كلمة المرور يجب أن تكون 6 أحرف على الأقل")
    if "@" not in req.email:
        raise HTTPException(400, detail="البريد الإلكتروني غير صالح")

    # Check uniqueness
    if db_get_user(req.username.lower()):
        raise HTTPException(409, detail="اسم المستخدم محجوز — اختر اسماً آخر")
    if db_get_user_by_email(req.email.lower()):
        raise HTTPException(409, detail="البريد الإلكتروني مسجّل مسبقاً")

    now = datetime.now(timezone.utc)
    trial_expires = (now + timedelta(days=TRIAL_DAYS)).isoformat()

    user = {
        "username":        req.username.lower(),
        "email":           req.email.lower(),
        "password_hash":   hash_pw(req.password),
        "full_name":       req.full_name,
        "phone":           req.phone,
        "plan":            "trial",
        "role":            "user",
        "active":          True,
        "trial_expires_at": trial_expires,
        "paid_until":      None,
        "created_at":      now.isoformat(),
        "last_login":      None,
    }
    db_save_user(user)

    token = create_token(req.username.lower())
    return {
        "token": token,
        "user": {
            "username":        user["username"],
            "email":           user["email"],
            "full_name":       user["full_name"],
            "plan":            user["plan"],
            "trial_expires_at": trial_expires,
        },
        "message": f"مرحباً! لديك {TRIAL_DAYS} أيام تجريبية مجانية.",
    }

@app.post("/auth/login")
async def login(req: LoginRequest):
    # Try username first, then email
    user = db_get_user(req.username.lower())
    if not user:
        user = db_get_user_by_email(req.username.lower())
    if not user:
        raise HTTPException(401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")
    if not user.get("active", True):
        raise HTTPException(403, detail="الحساب معطّل — تواصل مع الدعم")
    if not verify_pw(req.password, user.get("password_hash", "")):
        raise HTTPException(401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")

    # Update last login
    user["last_login"] = datetime.now(timezone.utc).isoformat()
    db_save_user(user)

    token = create_token(user["username"], user.get("role", "user"))

    # Check subscription status
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
            "username":        user["username"],
            "email":           user["email"],
            "full_name":       user.get("full_name", ""),
            "plan":            plan,
            "role":            user.get("role", "user"),
            "trial_expires_at": user.get("trial_expires_at"),
            "paid_until":      user.get("paid_until"),
            "subscription_status": subscription_status,
            "days_left":       days_left,
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
        "username":        user["username"],
        "email":           user["email"],
        "full_name":       user.get("full_name", ""),
        "plan":            plan,
        "role":            user.get("role", "user"),
        "trial_expires_at": user.get("trial_expires_at"),
        "paid_until":      user.get("paid_until"),
        "days_left":       days_left,
        "created_at":      user.get("created_at"),
    }

@app.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    user = db_get_user_by_email(req.email.lower())
    # Always return success (don't reveal if email exists)
    if not user:
        return {"message": "إذا كان البريد مسجّلاً، ستتلقى رمز الاستعادة من الدعم الفني."}

    # Generate 6-digit reset code valid for 1 hour
    token = str(secrets.randbelow(900000) + 100000)  # 100000–999999
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    db_save_reset(user["username"], token, expires)

    # In production: send email. For now: admin sees token in dashboard.
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

    # Check expiry
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
        "username":        req.username.lower(),
        "email":           req.email.lower(),
        "password_hash":   hash_pw(req.password),
        "full_name":       req.full_name,
        "phone":           req.phone,
        "plan":            req.plan,
        "role":            "admin" if req.plan == "admin" else "user",
        "active":          True,
        "trial_expires_at": trial_expires,
        "paid_until":      None,
        "created_at":      now.isoformat(),
        "last_login":      None,
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
    trial_active = 0
    trial_expired = 0
    suspended = 0
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
        "total": total,
        "paid": paid,
        "trial_active": trial_active,
        "trial_expired": trial_expired,
        "suspended": suspended,
        "conversion_rate": f"{round(paid / total * 100, 1)}%" if total else "0%",
    }

@app.get("/admin/resets")
async def admin_list_resets(admin: dict = Depends(get_admin_user)):
    """Admin can see pending reset codes to share with users manually."""
    _ensure_resets()
    results, _ = qdrant().scroll(collection_name=RESETS_COL, limit=100, with_payload=True)
    codes = [r.payload for r in results if r.payload and not r.payload.get("used")]
    # Only show unexpired codes
    now = datetime.now(timezone.utc)
    active = []
    for c in codes:
        try:
            exp = datetime.fromisoformat(c["expires_at"]).replace(tzinfo=timezone.utc)
            if now < exp:
                active.append({
                    "username": c["username"],
                    "token": c["token"],
                    "expires_at": c["expires_at"],
                })
        except Exception:
            pass
    return {"reset_codes": active}

# ═══════════════════════════════════════════════════════════════
#  CHAT ENDPOINTS (protected)
# ═══════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    ck = _ck(req.message, req.domain)
    cached = _cget(ck)
    if cached:
        return cached

    t0     = time.time()
    qinfo  = classify_query(req.message)
    chunks = await retrieve_multi(req.message, qinfo, req.domain)
    chunks = rerank_chunks(chunks, req.message)
    ctx    = context_str(chunks)
    model  = pick_model(req.message, qinfo)
    msgs   = build_messages(ctx, req.history, req.message, qinfo)

    resp = await oai().chat.completions.create(
        model=model, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.1,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    result = {
        "answer":      resp.choices[0].message.content,
        "model":       model,
        "chunks_used": len(chunks),
        "elapsed_s":   round(elapsed_ms / 1000, 2),
        "query_type":  qinfo['type'],
        "sources": [{"title": c["title"], "ministry": c["ministry"], "score": c["score"]} for c in chunks[:5]],
    }
    _cset(ck, result)
    log_query(user["username"], "chat", elapsed_ms)
    return result

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, user: dict = Depends(get_current_user)):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            t0     = time.time()
            qinfo  = classify_query(req.message)
            chunks = await retrieve_multi(req.message, qinfo, req.domain)
            chunks = rerank_chunks(chunks, req.message)
            ctx    = context_str(chunks)

            # ── Phase 9: Document context injection ──────────────────
            doc_context = ""
            if req.document_ids:
                try:
                    username = user.get("username", "")
                    with db_session() as _s:
                        docs = repo.documents.get_texts_for_ids(_s, req.document_ids[:5], username)
                    if docs:
                        doc_context = (
                            "\n\n---\n"
                            "📄 **وثائق المستخدم المرفقة** "
                            "(هذه وثائق خاصة بالمستخدم وليست مصادر رسمية):\n"
                        )
                        for d in docs:
                            excerpt = (d["extracted_text"] or "")[:3000]
                            if excerpt:
                                doc_context += f"\n**{d['file_name']}:**\n{excerpt}\n---\n"
                except Exception as _e:
                    logger.debug(f"Doc context injection failed: {_e}")
            # Append document context to the retrieved context
            if doc_context:
                ctx = ctx + doc_context

            model  = pick_model(req.message, qinfo)
            msgs   = build_messages(ctx, req.history, req.message, qinfo)

            retrieval_conf = _compute_retrieval_confidence(chunks)
            meta = {
                "type": "meta", "model": model, "chunks": len(chunks),
                "query_type": qinfo['type'],
                "confidence": retrieval_conf,
                "sources": [{"title": c["title"], "ministry": c.get("ministry",""), "score": c["score"]} for c in chunks[:5]],
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            stream = await oai().chat.completions.create(
                model=model, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.1, stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type':'token','text':delta,'choices':[{'delta':{'content':delta}}]}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
            log_query(user["username"], "chat_stream", int((time.time() - t0) * 1000))

            # Auto-log content gap when confidence is low
            if retrieval_conf in ("low", "unknown"):
                _log_content_gap(
                    user_question=req.message,
                    confidence=max((c.get("score", 0) for c in chunks), default=0.0),
                    username=user.get("username"),
                    gap_type="low_confidence",
                )

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','detail':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest, user: dict = Depends(get_current_user)):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            is_image = req.file_type.startswith("image/")
            is_pdf   = req.file_type == "application/pdf"
            is_word  = "word" in req.file_type or req.file_name.lower().endswith((".docx", ".doc"))
            is_text  = req.file_type.startswith("text/") or req.file_name.lower().endswith(".txt")

            extracted_text = ""
            if is_pdf:
                extracted_text = extract_text_from_pdf(req.file_base64)
            elif is_word:
                extracted_text = extract_text_from_docx(req.file_base64)
            elif is_text:
                try:
                    extracted_text = base64.b64decode(req.file_base64).decode("utf-8", errors="replace")[:15000]
                except Exception:
                    extracted_text = ""

            search_query = f"{req.file_name} {req.message} {extracted_text[:300]}"
            try:
                vec    = await embed(search_query)
                chunks = await search_qdrant(vec)
                ctx    = context_str(chunks)
            except Exception:
                ctx = ""

            ANALYSIS_PROMPT = SYSTEM_PROMPT + """

---

## قواعد تحليل الوثائق

أنت خبير متخصص في تحليل الوثائق الرسمية والقانونية اللبنانية.
عند تحليل أي وثيقة، اتبع هذا الهيكل الإلزامي بالترتيب:

### 1. 📋 تشخيص الوثيقة
- نوعها الدقيق (عقد / قرار / طلب / فاتورة / قيد / وكالة / حكم / إلخ)
- الجهة المُصدِرة والجهة المُستلِمة
- التاريخ ورقم المرجع إن وجد

### 2. 📌 استخراج البيانات الجوهرية
استخرج كل المعلومات المهمة: أسماء، أرقام، مبالغ، مواعيد، شروط، التزامات.

### 3. ⚠️ التنبيهات والمخاطر
هل هناك مواعيد نهائية قريبة؟ بنود مُلزِمة؟ إجراءات واجبة لم تُنفَّذ؟ تناقضات؟

### 4. ✅ الإجراءات العملية المطلوبة (بالترتيب)
خطوات واضحة ومرقّمة يجب على المواطن اتخاذها.

### 5. 📁 المستندات والمتطلبات
ما يجب تحضيره: وثائق، صور، طوابع، رسوم.

### 6. 🏛️ الجهة المختصة والتواصل
الوزارة أو الدائرة المختصة، رقم الهاتف، ساعات العمل.

### 7. 📝 النموذج أو المسودة الجاهزة
**إلزامي:** إذا استوجبت الوثيقة طلباً أو إفادةً: اكتب مسودة جاهزة بصيغة رسمية.

---
""" + (f"\n\n{ctx}" if ctx else "")

            user_text = f"سؤال/طلب المستخدم: {req.message}\n\nاسم الملف: {req.file_name}"
            if extracted_text and not extracted_text.startswith("[تعذّر"):
                user_text += f"\n\n--- نص الوثيقة المستخرج ---\n{extracted_text}\n--- نهاية النص ---"

            if is_image:
                user_content: list = [
                    {"type": "image_url", "image_url": {"url": f"data:{req.file_type};base64,{req.file_base64}", "detail": "high"}},
                    {"type": "text", "text": user_text},
                ]
            else:
                user_content = [{"type": "text", "text": user_text}]

            msgs: list = [{"role": "system", "content": ANALYSIS_PROMPT}]
            for m in req.history[-MAX_HISTORY:]:
                msgs.append({"role": m.role, "content": m.content})
            msgs.append({"role": "user", "content": user_content})

            stream = await oai().chat.completions.create(
                model=MODEL_SMART, messages=msgs, max_tokens=MAX_DOC_TOKENS,
                temperature=0.1, stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type':'token','text':delta,'choices':[{'delta':{'content':delta}}]}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','detail':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# Bootstrap first admin if ADMIN_USERNAME set
@app.on_event("startup")
async def startup():
    # Initialize SQLAlchemy database (SQLite default, PostgreSQL via DATABASE_URL)
    try:
        init_db()
        logger.info("✅ Database initialized (SQLite/PostgreSQL)")
    except Exception as e:
        logger.warning(f"DB init warning (non-fatal): {e}")

    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    admin_email    = os.environ.get("ADMIN_EMAIL", "admin@dalilak.ai")
    if admin_username and admin_password:
        if not db_get_user(admin_username.lower()):
            db_save_user({
                "username":        admin_username.lower(),
                "email":           admin_email,
                "password_hash":   hash_pw(admin_password),
                "full_name":       "Admin",
                "phone":           "",
                "plan":            "admin",
                "role":            "admin",
                "active":          True,
                "trial_expires_at": None,
                "paid_until":      None,
                "created_at":      datetime.now(timezone.utc).isoformat(),
                "last_login":      None,
            })

# ── Document Upload Stubs (Phase 8) ──────────────────────────────────────────
# TODO: Replace stubs with real storage (S3/Cloudflare R2) + PDF extraction

# ── Structured Agent Response Models (Phase 2) ───────────────────────────────

class StructuredDocument(BaseModel):
    title: str
    required: bool = True
    notes: Optional[str] = None
    alternative: Optional[str] = None

class StructuredStep(BaseModel):
    order: int
    title: str
    description: Optional[str] = None
    authority: Optional[str] = None
    estimatedTime: Optional[str] = None

class StructuredAuthority(BaseModel):
    name: str
    type: Optional[str] = None  # ministry|municipality|court|notary|registry|security|tax|other
    addressNotes: Optional[str] = None
    contactNotes: Optional[str] = None
    website: Optional[str] = None

class StructuredFee(BaseModel):
    label: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    notes: Optional[str] = None
    verified: bool = False

class StructuredForm(BaseModel):
    title: str
    type: str = "unknown"  # official|draft|unknown
    fileType: Optional[str] = None
    url: Optional[str] = None
    notes: Optional[str] = None
    verified: bool = False

class StructuredNextAction(BaseModel):
    label: str
    description: Optional[str] = None
    actionType: Optional[str] = "none"

class StructuredWarning(BaseModel):
    level: str = "info"  # info|warning|critical
    message: str

class StructuredSource(BaseModel):
    title: str
    type: str = "unknown"  # official|internal|user_uploaded|unknown
    url: Optional[str] = None
    excerpt: Optional[str] = None
    lastReviewed: Optional[str] = None
    reliability: Optional[str] = None  # high|medium|low|unknown

class StructuredConfidence(BaseModel):
    level: str = "unknown"  # high|medium|low|unknown
    reason: Optional[str] = None

class AgentResponseModel(BaseModel):
    kind: str = "structured_agent_response"
    language: str = "ar"
    country: Optional[str] = None
    procedureSlug: Optional[str] = None
    summary: str
    requiredDocuments: list[StructuredDocument] = []
    steps: list[StructuredStep] = []
    authority: Optional[StructuredAuthority] = None
    fees: list[StructuredFee] = []
    forms: list[StructuredForm] = []
    nextAction: Optional[StructuredNextAction] = None
    warnings: list[StructuredWarning] = []
    sources: list[StructuredSource] = []
    confidence: StructuredConfidence = StructuredConfidence()
    disclaimer: str = "هذه المعلومات للإرشاد العام فقط وليست بديلاً عن المشورة القانونية الرسمية."
    rawTextFallback: Optional[str] = None

class StructuredChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    domain: Optional[str] = None
    procedureSlug: Optional[str] = None
    flowAnswers: Optional[dict] = None   # answers collected by GuidedFlow

# ── Upload ─────────────────────────────────────────────────────────────────────

_ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
}

# NOTE: UploadDocumentRequest, delete_document, list_documents are defined in
# the DOCUMENT INTELLIGENCE section below (Phase 4/7/9) — real implementations.


# ── Feedback Endpoint (Phase 10) ──────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "up" or "down"
    confidence: Optional[str] = None
    sources: Optional[list] = None

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    """Store user feedback (thumbs up/down) on AI responses."""
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    try:
        from qdrant_client.models import PointStruct
        import uuid, time
        payload = {
            "type": "feedback",
            "username": user.get("username", "unknown"),
            "question": req.question[:500],
            "answer": req.answer[:1000],
            "rating": req.rating,
            "confidence": req.confidence,
            "timestamp": int(time.time()),
        }
        qdrant().upsert(
            collection_name="dalilak_logs",
            points=[PointStruct(id=str(uuid.uuid4()), vector=[0.0] * 3072, payload=payload)]
        )
        # Also log thumbs-down as content gap
        if req.rating == "down":
            _log_content_gap(
                user_question=req.question,
                confidence=0.0,
                username=user.get("username"),
                gap_type="user_reported_error",
                priority="high",
            )
    except Exception as e:
        logger.warning(f"Feedback store failed: {e}")
    return {"success": True}


# ── Admin Feedback Review (Phase 10) ─────────────────────────────────────────

@app.get("/admin/feedback")
async def get_feedback(limit: int = 50, rating: Optional[str] = None, user: dict = Depends(get_current_user)):
    """Return recent feedback entries. Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filt = Filter(must=[
            FieldCondition(key="type", match=MatchValue(value="feedback")),
        ])
        if rating in ("up", "down"):
            filt.must.append(FieldCondition(key="rating", match=MatchValue(value=rating)))
        results = qdrant().scroll(
            collection_name="dalilak_logs",
            scroll_filter=filt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        entries = [p.payload for p in results[0]]
        entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return {"feedback": entries, "total": len(entries)}
    except Exception as e:
        logger.error(f"Feedback fetch failed: {e}")
        return {"feedback": [], "total": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
#  TRANSACTION FILE ENDPOINTS (Phase 3)
# ═══════════════════════════════════════════════════════════════

@app.post("/transactions")
async def create_transaction(req: CreateTransactionRequest, user: dict = Depends(get_current_user)):
    """Create a new transaction file for the current user."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        tx = repo.transactions.create(
            session,
            user_id=user_id,
            title=req.title,
            procedure_slug=req.procedure_slug,
            country=req.country or "lebanon",
            user_type=req.user_type,
            summary=req.summary,
            notes=req.notes,
            required_documents=req.required_documents or [],
            steps=req.steps or [],
            sources=req.sources or [],
        )
        return tx.to_dict()


@app.get("/transactions")
async def list_transactions(user: dict = Depends(get_current_user)):
    """List all transaction files for the current user."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        txs = repo.transactions.list_by_user(session, user_id)
        return {"transactions": [t.to_dict() for t in txs], "total": len(txs)}


@app.get("/transactions/{tx_id}")
async def get_transaction(tx_id: str, user: dict = Depends(get_current_user)):
    """Get a specific transaction file."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        tx = repo.transactions.get(session, tx_id, user_id)
        if not tx:
            raise HTTPException(status_code=404, detail="ملف المعاملة غير موجود")
        return tx.to_dict()


@app.patch("/transactions/{tx_id}")
async def update_transaction(tx_id: str, req: UpdateTransactionRequest, user: dict = Depends(get_current_user)):
    """Update a transaction file."""
    user_id = user.get("username", "unknown")
    updates = {k: v for k, v in req.model_dump(exclude_none=True).items()}
    if not updates:
        raise HTTPException(status_code=400, detail="لا توجد حقول للتحديث")
    with db_session() as session:
        tx = repo.transactions.update(session, tx_id, user_id, **updates)
        if not tx:
            raise HTTPException(status_code=404, detail="ملف المعاملة غير موجود")
        return tx.to_dict()


@app.delete("/transactions/{tx_id}")
async def delete_transaction(tx_id: str, user: dict = Depends(get_current_user)):
    """Delete a transaction file."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        ok = repo.transactions.delete(session, tx_id, user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="ملف المعاملة غير موجود")
        return {"success": True}


@app.post("/transactions/{tx_id}/risk-score")
async def score_transaction_risk(tx_id: str, user: dict = Depends(get_current_user)):
    """Compute and persist risk score for a transaction."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        tx = repo.transactions.get(session, tx_id, user_id)
        if not tx:
            raise HTTPException(status_code=404, detail="ملف المعاملة غير موجود")
        missing = json.loads(tx.missing_documents or "[]")
        sources = json.loads(tx.sources or "[]")
        risk = compute_risk(
            confidence_level="medium",
            missing_documents_count=len(missing),
            has_sources=len(sources) > 0,
            procedure_slug=tx.procedure_slug,
        )
        repo.transactions.update(session, tx_id, user_id,
                                  risk_level=risk["level"],
                                  risk_score=float(risk["score"]),
                                  risk_reasons=risk["reasons"])
        return {"risk": risk, "transaction_id": tx_id}


@app.post("/procedures/{slug}/transaction-file")
async def create_from_procedure(slug: str, user: dict = Depends(get_current_user)):
    """Create a transaction file pre-populated from a procedure slug."""
    user_id = user.get("username", "unknown")
    slug_clean = slug.replace("-", " ")
    try:
        qinfo = classify_query(slug_clean)
        chunks = await retrieve_multi(slug_clean, qinfo, None)
        sources = [
            {"title": c.get("title", ""), "type": "internal", "ministry": c.get("ministry", "")}
            for c in chunks[:3]
        ]
    except Exception:
        sources = []
    with db_session() as session:
        tx = repo.transactions.create(
            session,
            user_id=user_id,
            title=f"ملف معاملة: {slug}",
            procedure_slug=slug,
            status="in_progress",
            sources=sources,
        )
        return tx.to_dict()


# ═══════════════════════════════════════════════════════════════
#  DOCUMENT INTELLIGENCE ENDPOINTS (Phases 4, 7, 9)
# ═══════════════════════════════════════════════════════════════

@app.post("/documents/upload")
async def upload_document_real(req: UploadDocumentRequest, user: dict = Depends(get_current_user)):
    """
    Real document upload — extract text server-side, store in SQLite.
    Document text is NEVER stored in localStorage (Phase 9 non-negotiable rule).
    """
    mime = req.file_type.lower()
    # Validate MIME type (allow common document types)
    allowed = any(t in mime for t in ["pdf", "word", "docx", "msword", "officedocument", "text/plain"])
    if not allowed:
        raise HTTPException(status_code=415, detail=f"نوع الملف غير مسموح به: {req.file_type}")
    try:
        raw_bytes = base64.b64decode(req.file_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="محتوى الملف غير صالح (base64)")
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="حجم الملف يتجاوز 10MB")

    # Extract text
    extracted_text = ""
    if "pdf" in mime:
        extracted_text = extract_text_from_pdf(req.file_base64)
    elif "word" in mime or "docx" in mime or "msword" in mime or "officedocument" in mime:
        extracted_text = extract_text_from_docx(req.file_base64)
    elif "text/plain" in mime:
        try:
            extracted_text = raw_bytes.decode("utf-8", errors="replace")[:15000]
        except Exception:
            extracted_text = ""

    user_id = user.get("username", "unknown")
    with db_session() as session:
        doc = repo.documents.create(
            session,
            user_id=user_id,
            file_name=req.file_name,
            file_type=req.file_type,
            file_size=len(raw_bytes),
            extracted_text=extracted_text[:15000] if extracted_text else None,
            transaction_id=req.transaction_id,
        )
        return {
            "success": True,
            "doc_id": doc.id,
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "size_bytes": doc.file_size,
            "has_text": bool(extracted_text),
            "text_length": len(extracted_text),
        }


@app.get("/documents")
async def list_user_documents(user: dict = Depends(get_current_user)):
    """List documents uploaded by the current user."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        docs = repo.documents.list_by_user(session, user_id)
        return {"documents": [d.to_dict() for d in docs], "total": len(docs)}


@app.get("/documents/{doc_id}")
async def get_document_meta(doc_id: str, user: dict = Depends(get_current_user)):
    """Get document metadata (no text returned)."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        doc = repo.documents.get(session, doc_id, user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="الوثيقة غير موجودة")
        return doc.to_dict(include_text=False)


@app.delete("/documents/{doc_id}")
async def delete_user_document(doc_id: str, user: dict = Depends(get_current_user)):
    """Delete a document and remove all stored text."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        ok = repo.documents.delete(session, doc_id, user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="الوثيقة غير موجودة")
        return {"success": True}


@app.post("/documents/{doc_id}/analyze")
async def analyze_doc_endpoint(doc_id: str, user: dict = Depends(get_current_user)):
    """
    Analyze document with GPT-4o — extract fields, detect type, suggest actions.
    Result is persisted; call GET /documents/{id} to check has_analysis.
    """
    user_id = user.get("username", "unknown")
    with db_session() as session:
        doc = repo.documents.get(session, doc_id, user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="الوثيقة غير موجودة")
        if not doc.extracted_text:
            raise HTTPException(
                status_code=422,
                detail="لم يتم استخراج النص من هذه الوثيقة — تأكد من رفعها بصيغة PDF أو Word"
            )
        result = await analyze_document(doc.extracted_text, doc.file_name, oai())
        repo.documents.update(
            session, doc_id, user_id,
            analysis_result=json.dumps(result, ensure_ascii=False),
            doc_type=result.get("document_type"),
            detected_country=result.get("detected_country"),
        )
        return result


@app.post("/documents/{doc_id}/risk-review")
async def risk_review_doc_endpoint(doc_id: str, user: dict = Depends(get_current_user)):
    """
    Deep contract risk review — clause analysis, missing/weak clauses, recommendations.
    Works best on lease/sale contracts and powers of attorney.
    """
    user_id = user.get("username", "unknown")
    with db_session() as session:
        doc = repo.documents.get(session, doc_id, user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="الوثيقة غير موجودة")
        if not doc.extracted_text:
            raise HTTPException(
                status_code=422,
                detail="لم يتم استخراج النص — تأكد من رفع الوثيقة بصيغة PDF أو Word"
            )
        result = await review_contract(doc.extracted_text, doc.file_name, oai())
        repo.documents.update(
            session, doc_id, user_id,
            risk_review=json.dumps(result, ensure_ascii=False),
        )
        return result


@app.post("/documents/{doc_id}/risk-score")
async def score_document_risk_endpoint(doc_id: str, user: dict = Depends(get_current_user)):
    """Compute risk score for a document based on its analysis/review results."""
    user_id = user.get("username", "unknown")
    with db_session() as session:
        doc = repo.documents.get(session, doc_id, user_id)
        if not doc:
            raise HTTPException(status_code=404, detail="الوثيقة غير موجودة")
        review_data = json.loads(doc.risk_review or "{}") if doc.risk_review else {}
        analysis_data = json.loads(doc.analysis_result or "{}") if doc.analysis_result else {}
        missing_clauses = len(review_data.get("missing_or_weak_clauses", []))
        missing_docs = len(analysis_data.get("missing_documents", []))
        confidence = analysis_data.get("confidence", {}).get("level", "unknown")
        risk = compute_risk(
            confidence_level=confidence,
            missing_documents_count=missing_docs,
            has_sources=True,
            procedure_slug=doc.doc_type,
            contract_missing_clauses_count=missing_clauses,
        )
        return {"risk": risk, "doc_id": doc_id}


# ═══════════════════════════════════════════════════════════════
#  STRUCTURED CHAT ENDPOINT (Phase 2) ──────────────────────────
# ═══════════════════════════════════════════════════════════════

# ── Structured Chat Endpoint (Phase 2) ───────────────────────────────────────
# Asks GPT-4o to return a validated AgentResponse JSON.
# Falls back to raw text if JSON parsing fails.

_STRUCTURED_SYSTEM = """
أنت دليلك AI — محرك إرشاد المعاملات الإدارية واللبنانية.
يجب أن تُرجع إجابتك دائماً كـ JSON صالح وفق المخطط التالي بالضبط.
لا تُضف أي نص خارج كتلة JSON.
لا تخترع رسوماً أو نماذج أو جهات رسمية غير متأكد منها — ضع verified: false لأي قيمة غير مؤكدة.

JSON Schema:
{
  "kind": "structured_agent_response",
  "language": "ar" | "en",
  "country": "lebanon" | "syria" | "both" | "unknown",
  "procedureSlug": string | null,
  "summary": string,                          // 2-4 جمل تلخيصية
  "requiredDocuments": [
    { "title": string, "required": bool, "notes": string | null, "alternative": string | null }
  ],
  "steps": [
    { "order": int, "title": string, "description": string | null, "authority": string | null, "estimatedTime": string | null }
  ],
  "authority": {
    "name": string,
    "type": "ministry"|"municipality"|"court"|"notary"|"registry"|"security"|"tax"|"other"|null,
    "addressNotes": string | null,
    "contactNotes": string | null,
    "website": string | null
  } | null,
  "fees": [
    { "label": string, "amount": string | null, "currency": string | null, "notes": string | null, "verified": bool }
  ],
  "forms": [
    { "title": string, "type": "official"|"draft"|"unknown", "fileType": "pdf"|"docx"|"link"|"unknown"|null, "url": string | null, "notes": string | null, "verified": bool }
  ],
  "nextAction": { "label": string, "description": string | null, "actionType": "download_checklist"|"generate_form"|"upload_document"|"start_flow"|"ask_followup"|"contact_human"|"none" },
  "warnings": [ { "level": "info"|"warning"|"critical", "message": string } ],
  "sources": [
    { "title": string, "type": "official"|"internal"|"unknown", "url": string | null, "excerpt": string | null, "lastReviewed": string | null, "reliability": "high"|"medium"|"low"|"unknown" }
  ],
  "confidence": { "level": "high"|"medium"|"low"|"unknown", "reason": string | null },
  "disclaimer": string
}
"""

def _compute_retrieval_confidence(chunks: list[dict]) -> str:
    """Compute confidence level from Qdrant retrieval scores."""
    if not chunks:
        return "low"
    max_score = max(c.get("score", 0) for c in chunks)
    avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
    if max_score >= 0.50 and avg_score >= 0.40:
        return "high"
    if max_score >= 0.35:
        return "medium"
    return "low"

@app.post("/chat/structured", response_model=None)
async def chat_structured(req: StructuredChatRequest, user: dict = Depends(get_current_user)):
    """
    Non-streaming endpoint: returns a typed AgentResponse JSON.
    Used by GuidedFlow final step and procedure detail Ask AI.
    Falls back to { rawTextFallback: ... } on parse error.
    """
    qinfo = classify_query(req.message)
    chunks = await retrieve_multi(req.message, qinfo, req.domain)
    chunks = rerank_chunks(chunks, req.message)
    ctx = context_str(chunks)
    retrieval_conf = _compute_retrieval_confidence(chunks)

    # Build flow context if answers were provided by GuidedFlow
    flow_ctx = ""
    if req.flowAnswers:
        flow_ctx = "\n\nإجابات المستخدم على أسئلة المعالج:\n" + "\n".join(
            f"- {k}: {v}" for k, v in req.flowAnswers.items()
        )
    if req.procedureSlug:
        flow_ctx += f"\n\nمعرّف المعاملة: {req.procedureSlug}"

    system = _STRUCTURED_SYSTEM + (f"\n\n{ctx}" if ctx else "") + flow_ctx
    msgs = [{"role": "system", "content": system}]
    for m in req.history[-6:]:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": req.message})

    try:
        completion = await oai().chat.completions.create(
            model=MODEL_SMART,
            messages=msgs,
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw_json = completion.choices[0].message.content or "{}"
        data = json.loads(raw_json)

        # Validate and add retrieval sources
        retrieved_sources = [
            {"title": c["title"], "type": "internal", "excerpt": c.get("text", "")[:200],
             "reliability": "medium" if c.get("score", 0) > 0.4 else "low"}
            for c in chunks[:5]
        ]
        if not data.get("sources"):
            data["sources"] = retrieved_sources

        # Enforce confidence from retrieval if model returned "unknown"
        if data.get("confidence", {}).get("level") == "unknown":
            data["confidence"] = {"level": retrieval_conf, "reason": "مبني على نتائج الاسترجاع من قاعدة البيانات"}

        data["kind"] = "structured_agent_response"

        # ── Content Gap auto-logging ──────────────────────────────
        final_conf = data.get("confidence", {}).get("level", "unknown")
        if final_conf in ("low", "unknown"):
            _log_content_gap(
                user_question=req.message,
                confidence=retrieval_conf,
                username=user.get("username"),
                procedure_slug=req.procedureSlug,
                gap_type="low_confidence",
            )

        return data

    except json.JSONDecodeError as e:
        # Fallback: return raw text wrapped in minimal structure
        text = completion.choices[0].message.content if completion else ""
        _log_content_gap(
            user_question=req.message,
            confidence=0.0,
            username=user.get("username"),
            gap_type="low_confidence",
        )
        return {
            "kind": "structured_agent_response",
            "language": "ar",
            "summary": "",
            "requiredDocuments": [], "steps": [], "fees": [], "forms": [],
            "warnings": [{"level": "warning", "message": "تعذّر تحليل الرد كـ JSON منظّم."}],
            "sources": [],
            "confidence": {"level": "low", "reason": f"JSON parse error: {e}"},
            "disclaimer": "هذه المعلومات للإرشاد العام فقط.",
            "rawTextFallback": text,
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Structured chat error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONTENT GAP ENGINE (Phase 4)
# ══════════════════════════════════════════════════════════════════════════════

def _log_content_gap(
    user_question: str,
    confidence: float = 0.0,
    username: Optional[str] = None,
    procedure_slug: Optional[str] = None,
    gap_type: str = "low_confidence",
    priority: str = "medium",
) -> None:
    """
    Silently log a content gap to the database.
    Called automatically when retrieval confidence is low.
    Never raises — failure is non-fatal.
    """
    try:
        from retrieval_service import _detect_country, _detect_procedure
        detected_country  = _detect_country(user_question)
        detected_procedure = procedure_slug or _detect_procedure(user_question)
        if confidence <= 0.20:
            priority = "high"
        elif confidence <= 0.35:
            priority = "medium"
        else:
            priority = "low"

        with db_session() as session:
            repo.content_gaps.create(
                session,
                user_question=user_question,
                gap_type=gap_type,
                detected_country=detected_country,
                detected_procedure=detected_procedure,
                confidence_score=round(confidence, 4),
                username=username,
                priority=priority,
            )
    except Exception as e:
        logger.debug(f"Content gap log skipped: {e}")


class ContentGapUpdateRequest(BaseModel):
    status: str          # open|in_review|resolved|ignored
    admin_notes: Optional[str] = None


@app.get("/admin/content-gaps")
async def admin_content_gaps(
    status: Optional[str] = None,
    limit: int = 100,
    admin: dict = Depends(get_admin_user),
):
    """List content gaps. Filter by status: open|in_review|resolved|ignored"""
    try:
        with db_session() as session:
            if status:
                gaps = (
                    session.query(__import__('database').ContentGap)
                    .filter_by(status=status)
                    .order_by(__import__('database').ContentGap.created_at.desc())
                    .limit(limit).all()
                )
            else:
                gaps = repo.content_gaps.list_all(session, limit=limit)
            stats = repo.content_gaps.stats(session)
            return {
                "gaps": [g.to_dict() for g in gaps],
                "total": len(gaps),
                "stats": stats,
            }
    except Exception as e:
        logger.error(f"Content gaps fetch error: {e}")
        return {"gaps": [], "total": 0, "stats": {}, "error": str(e)}


@app.patch("/admin/content-gaps/{gap_id}")
async def admin_update_content_gap(
    gap_id: str,
    req: ContentGapUpdateRequest,
    admin: dict = Depends(get_admin_user),
):
    """Update content gap status (open → in_review → resolved|ignored)."""
    valid = {"open", "in_review", "resolved", "ignored"}
    if req.status not in valid:
        raise HTTPException(400, detail=f"status must be one of {valid}")
    try:
        with db_session() as session:
            gap = repo.content_gaps.update_status(
                session,
                gap_id=gap_id,
                status=req.status,
                admin_notes=req.admin_notes,
                reviewer=admin.get("username"),
            )
            if not gap:
                raise HTTPException(404, detail="Gap not found")
            return {"success": True, "gap": gap.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/admin/content-gaps/log")
async def admin_log_content_gap(
    data: dict,
    admin: dict = Depends(get_admin_user),
):
    """Manually log a content gap (admin can create from observed feedback)."""
    try:
        with db_session() as session:
            gap = repo.content_gaps.create(
                session,
                user_question=data.get("user_question", ""),
                gap_type=data.get("gap_type", "other"),
                detected_country=data.get("detected_country"),
                detected_procedure=data.get("detected_procedure"),
                confidence_score=data.get("confidence_score"),
                username="admin_manual",
                priority=data.get("priority", "medium"),
            )
            return {"success": True, "gap_id": gap.id}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/admin/content-gaps/stats")
async def admin_content_gap_stats(admin: dict = Depends(get_admin_user)):
    """Return content gap statistics for admin overview."""
    try:
        with db_session() as session:
            return repo.content_gaps.stats(session)
    except Exception as e:
        return {"error": str(e)}


# ── Admin: Content Management Stubs (Phase 10) ───────────────────────────────

@app.get("/admin/procedures")
async def admin_list_procedures(user: dict = Depends(get_admin_user)):
    """STUB: Return list of procedures from the knowledge layer. Replace with DB query."""
    return {
        "procedures": [],
        "total": 0,
        "message": "STUB: Connect to PostgreSQL procedures table when ready.",
    }

@app.post("/admin/procedures")
async def admin_create_procedure(data: dict, user: dict = Depends(get_admin_user)):
    """STUB: Create or update a procedure entry."""
    return {"success": True, "message": "STUB: Procedure creation not yet persisted."}

@app.get("/admin/sources")
async def admin_list_sources(user: dict = Depends(get_admin_user)):
    """STUB: Return list of knowledge sources."""
    return {
        "sources": [],
        "total": 0,
        "message": "STUB: Connect to sources table when ready.",
    }

@app.post("/admin/sources")
async def admin_create_source(data: dict, user: dict = Depends(get_admin_user)):
    """STUB: Add a new knowledge source."""
    return {"success": True, "message": "STUB: Source creation not yet persisted."}

@app.get("/admin/failed-questions")
async def admin_failed_questions(limit: int = 50, user: dict = Depends(get_admin_user)):
    """Return low-confidence questions from logs for content gap analysis."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filt = Filter(must=[
            FieldCondition(key="type", match=MatchValue(value="feedback")),
            FieldCondition(key="rating", match=MatchValue(value="down")),
        ])
        results = client.scroll(
            collection_name="dalilak_logs",
            scroll_filter=filt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        entries = [p.payload for p in results[0]]
        entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return {"questions": entries, "total": len(entries)}
    except Exception as e:
        return {"questions": [], "total": 0, "error": str(e)}


# ── Human Escalation (Phase 16) ───────────────────────────────────────────────

class EscalationRequest(BaseModel):
    request_type: str   # "lawyer_review" | "document_review" | "consultation" | "whatsapp"
    question: str
    context: Optional[str] = None
    contact_preference: Optional[str] = None  # "email" | "whatsapp" | "callback"
    user_email: Optional[str] = None
    user_phone: Optional[str] = None

@app.post("/escalate")
async def request_escalation(req: EscalationRequest, user: dict = Depends(get_current_user)):
    """
    STUB: Log an escalation request. Replace with CRM/ticketing integration.
    Production: integrate with Calendly, WhatsApp Business API, or legal CRM.
    """
    try:
        from qdrant_client.models import PointStruct
        payload = {
            "type": "escalation",
            "username": user.get("username", "unknown"),
            "request_type": req.request_type,
            "question": req.question[:500],
            "context": (req.context or "")[:500],
            "contact_preference": req.contact_preference,
            "user_email": req.user_email,
            "user_phone": req.user_phone,
            "status": "pending",
            "timestamp": int(time.time()),
        }
        qdrant().upsert(
            collection_name="dalilak_logs",
            points=[PointStruct(id=str(uuid.uuid4()), vector=[0.0] * 3072, payload=payload)]
        )
    except Exception as e:
        logger.warning(f"Escalation log failed: {e}")
    return {
        "success": True,
        "message": "تم استلام طلبك. سيتواصل معك أحد المختصين قريباً.",
        "estimated_response": "خلال 24 ساعة عمل",
    }

@app.get("/admin/escalations")
async def admin_escalations(limit: int = 50, user: dict = Depends(get_admin_user)):
    """Return pending escalation requests."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filt = Filter(must=[FieldCondition(key="type", match=MatchValue(value="escalation"))])
        results = qdrant().scroll(
            collection_name="dalilak_logs",
            scroll_filter=filt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        entries = [p.payload for p in results[0]]
        entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return {"escalations": entries, "total": len(entries)}
    except Exception as e:
        return {"escalations": [], "total": 0, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  PDF EXPORT (Phase 8)
# ══════════════════════════════════════════════════════════════════════════════

class ChecklistExportRequest(BaseModel):
    title_ar: str
    title_en: Optional[str] = None
    country: Optional[str] = "lebanon"
    required_documents: list[dict] = []
    steps: list[dict] = []
    authority: Optional[dict] = None
    fees: list[dict] = []
    warnings: list[dict] = []
    sources: list[dict] = []
    confidence: Optional[str] = "medium"
    procedure_slug: Optional[str] = None
    language: str = "ar"


def _build_pdf_html(req: ChecklistExportRequest) -> str:
    """Generate a clean bilingual HTML that can be printed/saved as PDF."""
    is_ar = req.language == "ar"
    dir_attr = 'rtl' if is_ar else 'ltr'
    title = req.title_ar if is_ar else (req.title_en or req.title_ar)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conf_color = {"high": "#2E7D32", "medium": "#B8860B", "low": "#C62828"}.get(req.confidence or "low", "#666")

    docs_html = ""
    for d in req.required_documents:
        t = d.get("title") or d.get("name_ar") or d.get("name", "")
        notes = d.get("notes") or d.get("notes_ar") or ""
        req_mark = "✓" if d.get("required", True) else "(اختياري)"
        docs_html += f'<li><b>{req_mark}</b> {t}{"" if not notes else f" <span style=color:#666;font-size:12px>— {notes}</span>"}</li>'

    steps_html = ""
    for i, s in enumerate(req.steps, 1):
        t = s.get("title") or s.get("title_ar") or ""
        desc = s.get("description") or s.get("description_ar") or ""
        auth = s.get("authority") or ""
        steps_html += f'<li><b>{i}. {t}</b>{"" if not desc else f"<br><span style=color:#444;font-size:12px>{desc}</span>"}{"" if not auth else f"<br><span style=color:#8B1A1A;font-size:11px>📍 {auth}</span>"}</li>'

    auth_html = ""
    if req.authority:
        a = req.authority
        n = a.get("name") or a.get("name_ar") or ""
        web = a.get("website") or a.get("url") or ""
        ph = a.get("phone") or a.get("contactNotes") or ""
        auth_html = f"""<div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;background:#fafafa">
            <b>🏛️ {n}</b>{"" if not ph else f"<br>📞 {ph}"}{"" if not web else f'<br>🌐 <a href="{web}">{web}</a>'}</div>"""

    fees_html = ""
    for f in req.fees:
        lbl = f.get("label") or f.get("label_ar") or ""
        amt = f.get("amount") or ""
        verified = f.get("verified", False)
        tag = "" if verified else ' <span style="color:#B8860B;font-size:11px">(غير موثّق)</span>'
        fees_html += f"<li>{lbl}: <b>{amt}</b>{tag}</li>"

    warn_html = ""
    for w in req.warnings:
        lvl = w.get("level", "info")
        clr = {"critical": "#C62828", "warning": "#E65100", "info": "#1565C0"}.get(lvl, "#444")
        warn_html += f'<p style="color:{clr};font-size:12px">⚠️ {w.get("message","")}</p>'

    sources_html = ""
    for s in req.sources[:4]:
        t = s.get("title") or ""
        url = s.get("url") or ""
        sources_html += f'<li>{t}{"" if not url else f" — <a href={url}>{url}</a>"}</li>'

    return f"""<!DOCTYPE html>
<html lang="{'ar' if is_ar else 'en'}" dir="{dir_attr}">
<head>
<meta charset="UTF-8">
<title>{title} — دليلك AI</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: {'Cairo,Arial' if is_ar else 'Inter,Arial'}, sans-serif; font-size:14px; color:#1a1a1a; padding:32px; direction:{dir_attr}; }}
  h1 {{ font-size:22px; color:#8B1A1A; margin-bottom:4px; }}
  h2 {{ font-size:15px; color:#8B1A1A; margin:18px 0 8px; border-bottom:1px solid #EAE4D9; padding-bottom:4px; }}
  ul,ol {{ padding-inline-start:20px; }}
  li {{ margin:5px 0; line-height:1.6; }}
  .header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px; border-bottom:2px solid #8B1A1A; padding-bottom:12px; }}
  .meta {{ font-size:11px; color:#888; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; color:{conf_color}; border:1px solid {conf_color}; }}
  .disclaimer {{ margin-top:24px; font-size:11px; color:#888; border-top:1px solid #eee; padding-top:12px; }}
  @media print {{ body {{ padding:16px; }} }}
</style>
</head>
<body>
<div class="header">
  <div>
    <p style="font-size:11px;color:#888;margin-bottom:2px">دليلك AI — منصة إرشاد المعاملات الحكومية</p>
    <h1>{title}</h1>
    <p class="meta">{req.country or ''} · {date_str}</p>
  </div>
  <div style="text-align:{'left' if is_ar else 'right'}">
    <p class="meta">{'مستوى الثقة' if is_ar else 'Confidence'}</p>
    <span class="badge">{req.confidence or 'medium'}</span>
  </div>
</div>

{'<h2>📋 المستندات المطلوبة</h2><ul>' + docs_html + '</ul>' if docs_html else ''}
{'<h2>📝 الخطوات</h2><ol>' + steps_html + '</ol>' if steps_html else ''}
{auth_html}
{'<h2>💰 الرسوم</h2><ul>' + fees_html + '</ul>' if fees_html else ''}
{warn_html}
{'<h2>📚 المصادر</h2><ul>' + sources_html + '</ul>' if sources_html else ''}

<div class="disclaimer">
  ⚠️ {'هذه القائمة للإرشاد فقط وليست وثيقة رسمية. تأكد من المتطلبات الحالية من الجهة المختصة.' if is_ar else 'This checklist is for guidance only and is not an official document. Verify current requirements with the competent authority.'}
  | دليلك AI · dalilak.ai
</div>
</body>
</html>"""


@app.post("/export/checklist")
async def export_checklist(req: ChecklistExportRequest, user: dict = Depends(get_current_user)):
    """
    Generate a printable HTML checklist.
    Returns HTML content the client can open in a new tab and print/save as PDF.
    Plan gating: trial users get basic (documents only). Paid users get full.
    """
    plan = user.get("plan", "trial")

    # Simplify for trial users
    if plan == "trial":
        req.steps = req.steps[:3]
        req.fees = []
        req.sources = []

    html_content = _build_pdf_html(req)
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content, headers={
        "Content-Disposition": f'inline; filename="dalilak-checklist.html"',
        "Cache-Control": "no-cache",
    })


# ═══════════════════════════════════════════════════════════════
#  SERVICE GROUPS — Phase 6
#  JSON-backed repository. No DB required.
# ═══════════════════════════════════════════════════════════════

_SERVICE_GROUPS_DATA = [
    {
        "id": "sg-1", "slug": "expat", "priority": 1,
        "titleAr": "معاملات المغتربين", "titleEn": "Expat Services",
        "icon": "✈️", "color": "#1E40AF",
        "descriptionAr": "إنجاز معاملات لبنانية من الخارج عبر السفارات والتوكيل",
        "descriptionEn": "Complete Lebanese procedures from abroad via embassies or power of attorney",
        "services": [
            {"id":"si-1-1","slug":"poa-from-abroad","groupSlug":"expat","titleAr":"وكالة من الخارج","titleEn":"Power of Attorney from Abroad","icon":"📜","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["expat","lawyer"],"procedureSlug":"power-of-attorney"},
            {"id":"si-1-2","slug":"property-sale-abroad","groupSlug":"expat","titleAr":"بيع عقار من الخارج","titleEn":"Property Sale from Abroad","icon":"🏠","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["expat","lawyer"]},
            {"id":"si-1-3","slug":"document-attestation","groupSlug":"expat","titleAr":"تصديق مستندات","titleEn":"Document Attestation","icon":"🔏","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["expat","citizen","lawyer"],"procedureSlug":"document-attestation"},
            {"id":"si-1-4","slug":"register-abroad","groupSlug":"expat","titleAr":"تسجيل زواج أو ولادة من الخارج","titleEn":"Register Marriage/Birth from Abroad","icon":"📋","defaultAction":"start_flow","verificationStatus":"partially_verified","availableFor":["expat"]},
            {"id":"si-1-5","slug":"track-via-agent","groupSlug":"expat","titleAr":"متابعة معاملة عبر وكيل","titleEn":"Track Procedure via Agent","icon":"🔄","defaultAction":"ask_ai","verificationStatus":"draft","availableFor":["expat"]},
        ],
    },
    {
        "id": "sg-2", "slug": "property", "priority": 2,
        "titleAr": "العقارات", "titleEn": "Property Transactions",
        "icon": "🏛️", "color": "#854D0E",
        "descriptionAr": "بيع وشراء وتسجيل العقارات والحصول على الإفادات العقارية",
        "descriptionEn": "Buy, sell, register property and obtain real estate certificates",
        "services": [
            {"id":"si-2-1","slug":"property-sale","groupSlug":"property","titleAr":"بيع عقار","titleEn":"Property Sale","icon":"🏠","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","expat","lawyer"],"procedureSlug":"property-transfer"},
            {"id":"si-2-2","slug":"property-certificate","groupSlug":"property","titleAr":"إفادة عقارية","titleEn":"Property Certificate","icon":"📋","defaultAction":"ask_ai","verificationStatus":"verified","availableFor":["citizen","lawyer"]},
            {"id":"si-2-3","slug":"debt-clearance","groupSlug":"property","titleAr":"براءة ذمة","titleEn":"Debt Clearance","icon":"✅","defaultAction":"ask_ai","verificationStatus":"partially_verified","availableFor":["citizen","lawyer"]},
            {"id":"si-2-4","slug":"property-tax","groupSlug":"property","titleAr":"ضريبة العقار والرسوم","titleEn":"Property Tax & Fees","icon":"💰","defaultAction":"ask_ai","verificationStatus":"partially_verified","availableFor":["citizen","lawyer","company"]},
            {"id":"si-2-5","slug":"inheritance-property","groupSlug":"property","titleAr":"عقار ضمن إرث","titleEn":"Inherited Property","icon":"⚖️","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","lawyer"]},
            {"id":"si-2-6","slug":"poa-sale","groupSlug":"property","titleAr":"وكالة للبيع","titleEn":"Power of Attorney for Sale","icon":"📜","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","expat","lawyer"],"procedureSlug":"power-of-attorney"},
        ],
    },
    {
        "id": "sg-3", "slug": "contracts", "priority": 3,
        "titleAr": "العقود", "titleEn": "Contracts",
        "icon": "📝", "color": "#7C3AED",
        "descriptionAr": "تحليل العقود وكشف الثغرات ومراجعة البنود قبل التوقيع",
        "descriptionEn": "Analyze contracts, detect gaps, and review clauses before signing",
        "services": [
            {"id":"si-3-1","slug":"lease-review","groupSlug":"contracts","titleAr":"تحليل عقد إيجار","titleEn":"Lease Contract Review","icon":"🔍","defaultAction":"upload_document","requiresDocument":True,"verificationStatus":"verified","availableFor":["citizen","expat","lawyer","company"]},
            {"id":"si-3-2","slug":"contract-gaps","groupSlug":"contracts","titleAr":"كشف الثغرات والمخاطر","titleEn":"Gap & Risk Detection","icon":"⚠️","defaultAction":"upload_document","requiresDocument":True,"verificationStatus":"verified","availableFor":["citizen","lawyer","company"]},
            {"id":"si-3-3","slug":"missing-clauses","groupSlug":"contracts","titleAr":"البنود الناقصة","titleEn":"Missing Clauses","icon":"✏️","defaultAction":"upload_document","requiresDocument":True,"verificationStatus":"verified","availableFor":["citizen","lawyer","company"]},
            {"id":"si-3-4","slug":"obligations-map","groupSlug":"contracts","titleAr":"التزامات كل طرف","titleEn":"Party Obligations Map","icon":"📊","defaultAction":"upload_document","requiresDocument":True,"verificationStatus":"verified","availableFor":["citizen","lawyer","company"]},
            {"id":"si-3-5","slug":"signing-checklist","groupSlug":"contracts","titleAr":"Checklist قبل التوقيع","titleEn":"Pre-Signing Checklist","icon":"✅","defaultAction":"generate_checklist","verificationStatus":"verified","availableFor":["citizen","expat","lawyer","company"]},
        ],
    },
    {
        "id": "sg-4", "slug": "civil-records", "priority": 4,
        "titleAr": "الأحوال الشخصية والقيود", "titleEn": "Civil Records",
        "icon": "👨‍👩‍👦", "color": "#065F46",
        "descriptionAr": "استخراج وثائق الأحوال الشخصية وقيود السجل المدني",
        "descriptionEn": "Extract civil status documents and registry records",
        "services": [
            {"id":"si-4-1","slug":"civil-extract","groupSlug":"civil-records","titleAr":"إخراج قيد","titleEn":"Civil Registry Extract","icon":"📋","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","expat"],"procedureSlug":"civil-registry-extract"},
            {"id":"si-4-2","slug":"criminal-record","groupSlug":"civil-records","titleAr":"سجل عدلي","titleEn":"Criminal Record","icon":"📌","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","expat"],"procedureSlug":"criminal-record"},
            {"id":"si-4-3","slug":"birth-cert","groupSlug":"civil-records","titleAr":"شهادة ميلاد","titleEn":"Birth Certificate","icon":"👶","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","expat"],"procedureSlug":"birth-certificate"},
            {"id":"si-4-4","slug":"marriage-cert","groupSlug":"civil-records","titleAr":"وثيقة زواج","titleEn":"Marriage Certificate","icon":"💍","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","expat"],"procedureSlug":"marriage-registration"},
            {"id":"si-4-5","slug":"inheritance-cert","groupSlug":"civil-records","titleAr":"حصر إرث","titleEn":"Inheritance Certificate","icon":"⚖️","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["citizen","lawyer"],"procedureSlug":"inheritance-certificate"},
        ],
    },
    {
        "id": "sg-5", "slug": "business", "priority": 5,
        "titleAr": "الشركات والأعمال", "titleEn": "Business & Companies",
        "icon": "🏭", "color": "#9D174D",
        "descriptionAr": "تأسيس الشركات والتراخيص التجارية والضمان الاجتماعي",
        "descriptionEn": "Company formation, commercial licenses, and social security",
        "services": [
            {"id":"si-5-1","slug":"company-formation","groupSlug":"business","titleAr":"تأسيس شركة","titleEn":"Company Formation","icon":"🏭","defaultAction":"start_flow","verificationStatus":"verified","availableFor":["company","lawyer","citizen"],"procedureSlug":"company-registration"},
            {"id":"si-5-2","slug":"commercial-registry","groupSlug":"business","titleAr":"السجل التجاري","titleEn":"Commercial Registry","icon":"📋","defaultAction":"ask_ai","verificationStatus":"verified","availableFor":["company","lawyer"]},
            {"id":"si-5-3","slug":"business-tax","groupSlug":"business","titleAr":"ضريبة وإلتزامات الشركات","titleEn":"Corporate Tax & Obligations","icon":"💰","defaultAction":"ask_ai","verificationStatus":"partially_verified","availableFor":["company","accountant"]},
            {"id":"si-5-4","slug":"social-security-reg","groupSlug":"business","titleAr":"الضمان الاجتماعي","titleEn":"Social Security","icon":"🏥","defaultAction":"ask_ai","verificationStatus":"verified","availableFor":["company","citizen"],"procedureSlug":"social-security"},
        ],
    },
    {
        "id": "sg-6", "slug": "forms-docs", "priority": 6,
        "titleAr": "النماذج والمستندات", "titleEn": "Forms & Documents",
        "icon": "📄", "color": "#374151",
        "descriptionAr": "البحث عن نماذج رسمية وتحليل المستندات وتوليد مسودات",
        "descriptionEn": "Find official forms, analyze documents, and generate drafts",
        "services": [
            {"id":"si-6-1","slug":"find-form","groupSlug":"forms-docs","titleAr":"البحث عن نموذج","titleEn":"Find a Form","icon":"🔍","defaultAction":"ask_ai","verificationStatus":"verified","availableFor":["citizen","expat","lawyer","company","service_office"]},
            {"id":"si-6-2","slug":"analyze-document","groupSlug":"forms-docs","titleAr":"تحليل مستند","titleEn":"Analyze Document","icon":"🔎","defaultAction":"upload_document","requiresDocument":True,"verificationStatus":"verified","availableFor":["citizen","expat","lawyer","company"]},
            {"id":"si-6-3","slug":"detect-missing","groupSlug":"forms-docs","titleAr":"كشف النواقص","titleEn":"Detect Missing Items","icon":"⚠️","defaultAction":"upload_document","requiresDocument":True,"verificationStatus":"verified","availableFor":["citizen","lawyer"]},
            {"id":"si-6-4","slug":"download-checklist","groupSlug":"forms-docs","titleAr":"تحميل Checklist","titleEn":"Download Checklist","icon":"✅","defaultAction":"generate_checklist","verificationStatus":"verified","availableFor":["citizen","expat","lawyer","company"]},
            {"id":"si-6-5","slug":"generate-draft","groupSlug":"forms-docs","titleAr":"توليد مسودة","titleEn":"Generate Draft","icon":"✏️","defaultAction":"ask_ai","verificationStatus":"draft","availableFor":["citizen","lawyer","company"]},
        ],
    },
]

class ServiceStartRequest(BaseModel):
    user_type: Optional[str] = None
    context: Optional[str] = None


@app.get("/service-groups")
async def get_service_groups():
    """Return all service groups with their services."""
    return {"groups": _SERVICE_GROUPS_DATA}


@app.get("/service-groups/{slug}")
async def get_service_group(slug: str):
    """Return a single service group by slug."""
    group = next((g for g in _SERVICE_GROUPS_DATA if g["slug"] == slug), None)
    if not group:
        raise HTTPException(status_code=404, detail=f"Service group '{slug}' not found")
    return group


@app.get("/services/{slug}")
async def get_service_item(slug: str):
    """Return a single service item by slug."""
    for group in _SERVICE_GROUPS_DATA:
        for svc in group["services"]:
            if svc["slug"] == slug:
                return {**svc, "group": {k: v for k, v in group.items() if k != "services"}}
    raise HTTPException(status_code=404, detail=f"Service '{slug}' not found")


@app.post("/services/{slug}/start")
async def start_service(slug: str, req: ServiceStartRequest, user: dict = Depends(get_current_user)):
    """
    Start a service journey. Returns routing instructions:
    - guided_flow: trigger the GuidedFlow wizard with a pre-selected procedure
    - upload_document: prompt the user to upload a document
    - ask_ai: pre-fill the chat input with a context-aware prompt
    - generate_checklist: generate and return a checklist prompt
    """
    for group in _SERVICE_GROUPS_DATA:
        for svc in group["services"]:
            if svc["slug"] == slug:
                action = svc["defaultAction"]
                prompt_ar = None
                if action == "ask_ai":
                    prompt_ar = f"أريد معلومات عن: {svc['titleAr']}"
                elif action == "generate_checklist":
                    prompt_ar = f"أعطني checklist شامل لـ: {svc['titleAr']}"
                elif action == "upload_document":
                    prompt_ar = f"قم بتحليل المستند المرفوع بخصوص: {svc['titleAr']}"
                return {
                    "service": svc,
                    "action": action,
                    "procedureSlug": svc.get("procedureSlug"),
                    "requiresDocument": svc.get("requiresDocument", False),
                    "chatPrompt": prompt_ar,
                    "userId": user.get("username"),
                }
    raise HTTPException(status_code=404, detail=f"Service '{slug}' not found")


# ── Human Review Request ──────────────────────────────────────────────────────

class HumanReviewRequest(BaseModel):
    transaction_id: Optional[str] = None
    document_ids: list[str] = []
    request_type: str = "general"  # contract_review | document_review | property | lawyer | pre_signing
    summary: Optional[str] = None
    urgency: str = "normal"  # normal | high | critical


@app.post("/human-review/request")
async def request_human_review(req: HumanReviewRequest, user: dict = Depends(get_current_user)):
    """
    Log a human review request.
    TODO: integrate with CRM / notification service.
    """
    review_id = str(uuid.uuid4())[:8]
    return {
        "review_id": review_id,
        "status": "received",
        "message_ar": "تم استلام طلب المراجعة. سيتواصل معك فريق دليلك AI خلال 24–48 ساعة.",
        "message_en": "Review request received. The Dalilak AI team will contact you within 24–48 hours.",
        "estimated_response": "24–48h",
        "user": user.get("username"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL DOCUMENT INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Configurable document-type → draft templates map ─────────────────────────

_DOCUMENT_DRAFTS_MAP: dict = {
    "contract": [
        "eviction-notice", "rent-payment-notice", "repair-request",
        "deposit-return-request", "contract-addendum", "final-settlement",
        "property-handover",
    ],
    "property": [
        "property-statement-req", "municipal-clearance-req", "property-sale-poa",
        "property-sale-checklist", "lawyer-referral-letter", "docs-handover-report",
    ],
    "notarial": [
        "poa-scope-review", "poa-amendment-request", "follow-up-poa",
        "notary-letter", "agent-letter", "poa-validity-checklist",
        "property-sale-poa",
    ],
    "civil_status": [
        "civil-record-request", "event-registration-req", "record-correction-req",
        "doc-certification-req", "civil-checklist", "mukhtar-letter",
        "inheritance-checklist", "inheritance-request", "heirs-letter",
        "heirs-docs-table",
    ],
    "company": [
        "commercial-registry-req", "board-resolution", "tax-ministry-letter",
        "tax-registration-req", "nssf-registration-req", "company-setup-checklist",
    ],
    "tax": [
        "tax-assessment-objection", "tax-clearance-req", "installment-request",
        "tax-docs-checklist", "tax-ministry-letter",
    ],
    "judicial": [
        "notice-reply", "extension-request", "legal-objection",
        "lawyer-review-request", "case-summary", "legal-docs-checklist",
    ],
    "administrative": [
        "admin-objection", "review-request", "transaction-follow-up",
        "reminder-letter", "admin-complaint", "admin-review-checklist",
    ],
    "expat_consular": [
        "consular-poa", "certification-request", "consulate-letter",
        "expat-checklist", "local-agent-letter", "expat-transaction-file",
    ],
}

# ── Draft template catalogue (titles + metadata) ──────────────────────────────

_DRAFT_CATALOGUE: dict = {
    "eviction-notice":         {"ar": "إنذار بالإخلاء",                     "en": "Eviction Notice",              "lawyer": True},
    "rent-payment-notice":     {"ar": "إنذار بدفع بدلات الإيجار",            "en": "Rent Payment Notice",          "lawyer": False},
    "repair-request":          {"ar": "طلب إصلاحات",                         "en": "Repair Request Letter",        "lawyer": False},
    "deposit-return-request":  {"ar": "طلب إعادة التأمين",                   "en": "Deposit Return Request",       "lawyer": False},
    "contract-addendum":       {"ar": "ملحق تعديل عقد",                      "en": "Contract Addendum",            "lawyer": False},
    "final-settlement":        {"ar": "مخالصة نهائية",                       "en": "Final Settlement Receipt",     "lawyer": False},
    "property-handover":       {"ar": "محضر تسليم مأجور/عقار",               "en": "Property Handover Report",     "lawyer": False},
    "property-statement-req":  {"ar": "طلب إفادة عقارية",                    "en": "Property Statement Request",   "lawyer": False},
    "municipal-clearance-req": {"ar": "طلب براءة ذمة بلدية",                 "en": "Municipal Clearance Request",  "lawyer": False},
    "property-sale-poa":       {"ar": "وكالة بيع عقار",                      "en": "Property Sale POA",            "lawyer": True},
    "property-sale-checklist": {"ar": "Checklist بيع عقار",                  "en": "Property Sale Checklist",      "lawyer": False},
    "lawyer-referral-letter":  {"ar": "رسالة إلى محامٍ",                     "en": "Attorney Referral Letter",     "lawyer": False},
    "docs-handover-report":    {"ar": "محضر تسليم مستندات عقارية",            "en": "Property Docs Handover",       "lawyer": False},
    "poa-scope-review":        {"ar": "مراجعة صلاحيات الوكالة",              "en": "POA Scope Review",             "lawyer": False},
    "poa-amendment-request":   {"ar": "طلب تعديل وكالة",                     "en": "POA Amendment Request",        "lawyer": True},
    "follow-up-poa":           {"ar": "وكالة متابعة معاملة",                 "en": "Follow-up POA",                "lawyer": False},
    "notary-letter":           {"ar": "كتاب إلى كاتب عدل",                   "en": "Notary Letter",                "lawyer": False},
    "agent-letter":            {"ar": "رسالة إلى الوكيل",                    "en": "Letter to Agent",              "lawyer": False},
    "poa-validity-checklist":  {"ar": "Checklist صلاحية وكالة",              "en": "POA Validity Checklist",       "lawyer": False},
    "civil-record-request":    {"ar": "طلب إخراج قيد",                       "en": "Civil Record Request",         "lawyer": False},
    "event-registration-req":  {"ar": "طلب تسجيل واقعة",                     "en": "Event Registration Request",   "lawyer": False},
    "record-correction-req":   {"ar": "طلب تصحيح قيد",                       "en": "Record Correction Request",    "lawyer": False},
    "doc-certification-req":   {"ar": "طلب تصديق مستند",                     "en": "Document Certification Request","lawyer": False},
    "civil-checklist":         {"ar": "Checklist معاملة نفوس",               "en": "Civil Status Checklist",       "lawyer": False},
    "mukhtar-letter":          {"ar": "رسالة إلى مختار أو دائرة نفوس",       "en": "Letter to Mukhtar / Registry", "lawyer": False},
    "inheritance-checklist":   {"ar": "Checklist حصر إرث",                   "en": "Inheritance Checklist",        "lawyer": False},
    "inheritance-request":     {"ar": "طلب حصر إرث",                         "en": "Inheritance Petition",         "lawyer": True},
    "heirs-letter":            {"ar": "رسالة إلى الورثة",                    "en": "Letter to Heirs",              "lawyer": False},
    "heirs-docs-table":        {"ar": "جدول مستندات الورثة",                 "en": "Heirs Documents Table",        "lawyer": False},
    "commercial-registry-req": {"ar": "طلب سجل تجاري",                       "en": "Commercial Registry Request",  "lawyer": False},
    "board-resolution":        {"ar": "محضر جمعية / قرار شركاء",             "en": "Board Resolution",             "lawyer": False},
    "tax-ministry-letter":     {"ar": "كتاب إلى وزارة المالية",              "en": "Letter to Ministry of Finance","lawyer": False},
    "tax-registration-req":    {"ar": "طلب تسجيل ضريبي",                     "en": "Tax Registration Request",     "lawyer": False},
    "nssf-registration-req":   {"ar": "طلب تسجيل ضمان",                      "en": "NSSF Registration Request",    "lawyer": False},
    "company-setup-checklist": {"ar": "Checklist تأسيس شركة",                "en": "Company Setup Checklist",      "lawyer": False},
    "tax-assessment-objection":{"ar": "اعتراض على تكليف أو غرامة",           "en": "Tax Assessment Objection",     "lawyer": True},
    "tax-clearance-req":       {"ar": "طلب براءة ذمة",                       "en": "Tax Clearance Request",        "lawyer": False},
    "installment-request":     {"ar": "طلب تقسيط",                           "en": "Installment Request",          "lawyer": False},
    "tax-docs-checklist":      {"ar": "Checklist مستندات ضريبية",            "en": "Tax Documents Checklist",      "lawyer": False},
    "notice-reply":            {"ar": "جواب على إنذار",                      "en": "Notice Reply",                 "lawyer": True},
    "extension-request":       {"ar": "طلب مهلة",                            "en": "Extension Request",            "lawyer": False},
    "legal-objection":         {"ar": "كتاب اعتراض",                         "en": "Legal Objection Letter",       "lawyer": True},
    "lawyer-review-request":   {"ar": "طلب مراجعة محامٍ",                   "en": "Request Legal Review",         "lawyer": False},
    "case-summary":            {"ar": "ملخص ملف قضائي",                      "en": "Case File Summary",            "lawyer": False},
    "legal-docs-checklist":    {"ar": "Checklist للمستندات القانونية",        "en": "Legal Documents Checklist",    "lawyer": False},
    "admin-objection":         {"ar": "اعتراض إداري",                        "en": "Administrative Objection",     "lawyer": False},
    "review-request":          {"ar": "طلب إعادة نظر",                       "en": "Reconsideration Request",      "lawyer": False},
    "transaction-follow-up":   {"ar": "طلب متابعة معاملة",                   "en": "Transaction Follow-up Request","lawyer": False},
    "reminder-letter":         {"ar": "كتاب تذكير",                          "en": "Reminder Letter",              "lawyer": False},
    "admin-complaint":         {"ar": "شكوى إدارية",                         "en": "Administrative Complaint",     "lawyer": False},
    "admin-review-checklist":  {"ar": "Checklist للمراجعة",                  "en": "Review Checklist",             "lawyer": False},
    "consular-poa":            {"ar": "وكالة من الخارج",                     "en": "Consular Power of Attorney",   "lawyer": False},
    "certification-request":   {"ar": "طلب تصديق",                           "en": "Certification Request",        "lawyer": False},
    "consulate-letter":        {"ar": "رسالة إلى القنصلية",                  "en": "Consulate Letter",             "lawyer": False},
    "expat-checklist":         {"ar": "Checklist تصديق وترجمة",              "en": "Certification & Translation Checklist","lawyer": False},
    "local-agent-letter":      {"ar": "رسالة إلى وكيل داخل لبنان/سوريا",    "en": "Letter to Local Agent",        "lawyer": False},
    "expat-transaction-file":  {"ar": "ملف متابعة معاملة مغترب",             "en": "Expat Transaction File",       "lawyer": False},
}

# ── GPT-4o Prompt for Universal Document Analysis ────────────────────────────

_UNIVERSAL_ANALYSIS_SYSTEM = """
أنت محلل مستندات قانوني وإداري متخصص في المعاملات اللبنانية والسورية.
مهمتك تحليل المستند المرفوع وإعادة بيانات منظّمة بصيغة JSON صارمة.

قواعد حاسمة:
- لا تخترع معلومات. إذا لم تجد بياناً فاذكر "غير محدد" أو اترك القيمة فارغة.
- لا تخترع مواعيد أو رسوم أو متطلبات رسمية.
- كل تفسير غير موثّق ضعه في "ai_inferred".
- لا تضمن نتائج قانونية.
- إذا كان المستند غير واضح فقل ذلك في confidence.

أعد JSON فقط، بدون أي نص خارجه.
"""

_UNIVERSAL_ANALYSIS_USER = """
حلّل المستند التالي وأعد JSON دقيق يتبع هذا الهيكل بالضبط:

{{
  "documentType": {{
    "category": "<contract|property|civil_status|notarial|company|tax|judicial|administrative|expat_consular|unknown>",
    "subtype": "<نوع فرعي مثل: عقد إيجار / وكالة بيع / إخراج قيد فردي ...>",
    "confidence": "<high|medium|low>"
  }},
  "detectedCountry": "<lebanon|syria|both|unknown>",
  "detectedLanguage": "<ar|en|fr|mixed|unknown>",
  "extractedFacts": [
    {{"label": "...", "value": "...", "normalizedKey": "...", "confidence": "high|medium|low", "sourceExcerpt": "..."}}
  ],
  "relatedProcedures": [
    {{"procedureSlug": "...", "titleAr": "...", "titleEn": "...", "relevance": "high|medium|low", "reason": "..."}}
  ],
  "possibleUses": [
    {{"titleAr": "...", "titleEn": "...", "descriptionAr": "...", "relatedProcedureSlug": "..."}}
  ],
  "missingInformation": [
    {{"field": "...", "whyItMatters": "...", "requiredFor": "...", "priority": "low|medium|high|critical"}}
  ],
  "missingDocuments": [
    {{"titleAr": "...", "titleEn": "...", "reason": "...", "priority": "low|medium|high|critical", "status": "missing|unclear|needs_review"}}
  ],
  "risks": [
    {{"title": "...", "level": "low|medium|high|critical", "explanation": "...", "recommendedAction": "..."}}
  ],
  "recommendedDraftSlugs": ["slug1", "slug2"],
  "nextActions": [
    {{"labelAr": "...", "labelEn": "...", "actionType": "<create_transaction_file|generate_checklist|generate_draft|upload_missing_document|start_guided_flow|ask_followup|request_human_review|compare_with_template|none>", "priority": "primary|secondary"}}
  ],
  "evidence": [
    {{"claim": "...", "sourceType": "user_uploaded_document|official|internal|ai_inferred|unknown", "sourceTitle": "...", "excerpt": "...", "verified": false, "reliability": "high|medium|low"}}
  ],
  "confidence": {{
    "extraction": "high|medium|low",
    "procedureMatching": "high|medium|low",
    "legalInterpretation": "high|medium|low",
    "overall": "high|medium|low",
    "reason": "..."
  }}
}}

للمجالات الفارغة أو الغير قابلة للتحديد: استخدم [] أو "" أو "unknown".

المستند:
---
{document_text}
---

اسم الملف: {filename}
"""


# ── DocumentIntelligenceService ───────────────────────────────────────────────

class DocumentIntelligenceService:
    """
    Universal document analysis service.
    Uses GPT-4o to classify, extract facts, match procedures, detect risks,
    and recommend drafts for any legal/administrative document.
    """

    def __init__(self, client):
        self.client = client

    async def analyze(
        self,
        document_text: str,
        filename: str,
        document_id: str,
    ) -> dict:
        """
        Full analysis pipeline:
        1. Classify document
        2. Extract facts
        3. Match procedures
        4. Detect risks
        5. Recommend drafts
        6. Build next actions
        """
        if not document_text or not document_text.strip():
            return self._empty_analysis(document_id, filename)

        # Truncate to ~6000 chars to stay well within GPT-4o context
        truncated = document_text[:6000]
        if len(document_text) > 6000:
            truncated += "\n\n[... المستند مقتطع للتحليل ...]"

        prompt = _UNIVERSAL_ANALYSIS_USER.format(
            document_text=truncated,
            filename=filename,
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _UNIVERSAL_ANALYSIS_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2500,
            )
            raw = response.choices[0].message.content or "{}"
            gpt_data = json.loads(raw)
        except Exception as e:
            logger.warning(f"Document intelligence GPT call failed: {e}")
            return self._empty_analysis(document_id, filename, error=str(e))

        # Enrich recommended drafts from slugs
        draft_slugs = gpt_data.get("recommendedDraftSlugs", [])
        recommended_drafts = self._build_draft_recommendations(
            draft_slugs,
            gpt_data.get("documentType", {}).get("category", "unknown"),
        )

        # Build disclaimer
        disclaimer = (
            "هذا التحليل إرشادي فقط استناداً إلى المستند المرفوع وقاعدة معرفة دليلك AI. "
            "لا يُعدّ استشارة قانونية رسمية. تأكد من المتطلبات الفعلية من الجهة المختصة "
            "أو من محامٍ مرخّص قبل اتخاذ أي إجراء."
        )

        return {
            "kind": "universal_document_analysis",
            "documentId": document_id,
            "fileName": filename,
            "documentType": gpt_data.get("documentType", {"category": "unknown", "confidence": "low"}),
            "detectedCountry": gpt_data.get("detectedCountry", "unknown"),
            "detectedLanguage": gpt_data.get("detectedLanguage", "unknown"),
            "extractedFacts": gpt_data.get("extractedFacts", []),
            "relatedProcedures": gpt_data.get("relatedProcedures", []),
            "possibleUses": gpt_data.get("possibleUses", []),
            "missingInformation": gpt_data.get("missingInformation", []),
            "missingDocuments": gpt_data.get("missingDocuments", []),
            "risks": gpt_data.get("risks", []),
            "recommendedDrafts": recommended_drafts,
            "nextActions": gpt_data.get("nextActions", self._default_next_actions()),
            "evidence": gpt_data.get("evidence", []),
            "confidence": gpt_data.get("confidence", {
                "extraction": "low", "procedureMatching": "low",
                "legalInterpretation": "low", "overall": "low",
            }),
            "disclaimer": disclaimer,
        }

    def _build_draft_recommendations(self, slugs: list, category: str) -> list:
        """Convert slug list + category into full draft recommendation objects."""
        # Merge GPT-suggested slugs with category defaults
        category_defaults = _DOCUMENT_DRAFTS_MAP.get(category, [])
        all_slugs = list(dict.fromkeys(slugs + category_defaults))[:8]  # dedupe, cap at 8

        drafts = []
        for slug in all_slugs:
            meta = _DRAFT_CATALOGUE.get(slug)
            if not meta:
                continue
            drafts.append({
                "templateSlug": slug,
                "titleAr": meta["ar"],
                "titleEn": meta["en"],
                "category": self._infer_draft_category(slug),
                "recommendedBecause": f"مناسب لنوع المستند ({category}) وما تم استخراجه",
                "requiresLawyerReview": meta["lawyer"],
                "requiredFields": [],
            })
        return drafts

    def _infer_draft_category(self, slug: str) -> str:
        if "checklist" in slug or "list" in slug: return "checklist"
        if "notice" in slug or "إنذار" in slug:   return "notice"
        if "request" in slug or "req" in slug:    return "request"
        if "objection" in slug or "اعتراض" in slug: return "objection"
        if "letter" in slug or "كتاب" in slug:    return "administrative_letter"
        if "poa" in slug or "وكالة" in slug:      return "declaration"
        if "settlement" in slug:                   return "settlement"
        if "addendum" in slug:                     return "contract_addendum"
        return "form_draft"

    def _default_next_actions(self) -> list:
        return [
            {"labelAr": "إنشاء ملف معاملة", "labelEn": "Create Transaction File",
             "actionType": "create_transaction_file", "priority": "primary"},
            {"labelAr": "إنشاء Checklist", "labelEn": "Generate Checklist",
             "actionType": "generate_checklist", "priority": "primary"},
            {"labelAr": "طلب مراجعة بشرية", "labelEn": "Request Human Review",
             "actionType": "request_human_review", "priority": "secondary"},
        ]

    def _empty_analysis(self, document_id: str, filename: str, error: str = "") -> dict:
        return {
            "kind": "universal_document_analysis",
            "documentId": document_id,
            "fileName": filename,
            "documentType": {"category": "unknown", "confidence": "low"},
            "detectedCountry": "unknown",
            "detectedLanguage": "unknown",
            "extractedFacts": [],
            "relatedProcedures": [],
            "possibleUses": [],
            "missingInformation": [],
            "missingDocuments": [],
            "risks": [],
            "recommendedDrafts": [],
            "nextActions": self._default_next_actions(),
            "evidence": [],
            "confidence": {
                "extraction": "low", "procedureMatching": "low",
                "legalInterpretation": "low", "overall": "low",
                "reason": error or "لم يتم العثور على محتوى قابل للتحليل",
            },
            "disclaimer": "تعذّر تحليل المستند. تأكد من صحة الملف وحاول مرة أخرى.",
        }


# Singleton service instance (reuses the global openai_client)
_doc_intelligence_service: Optional[DocumentIntelligenceService] = None


def get_doc_intelligence() -> DocumentIntelligenceService:
    global _doc_intelligence_service
    if _doc_intelligence_service is None:
        _doc_intelligence_service = DocumentIntelligenceService(openai_client)
    return _doc_intelligence_service


# ── Universal Analysis Request Models ────────────────────────────────────────

class UniversalAnalysisRequest(BaseModel):
    document_text: str
    filename: str = "document"
    document_id: Optional[str] = None


class DraftGenerateRequest(BaseModel):
    template_slug: str
    document_id: Optional[str] = None
    transaction_id: Optional[str] = None
    related_procedure_slug: Optional[str] = None
    language: str = "ar"
    extracted_facts: Optional[dict] = None
    user_inputs: Optional[dict] = None
    redaction_mode: str = "none"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/documents/universal-analysis")
async def universal_document_analysis(
    req: UniversalAnalysisRequest,
    user: dict = Depends(get_current_user),
):
    """
    Full universal analysis of any uploaded document.
    Classifies, extracts facts, matches procedures, detects risks, recommends drafts.
    """
    doc_id = req.document_id or str(uuid.uuid4())[:8]
    svc = get_doc_intelligence()
    result = await svc.analyze(req.document_text, req.filename, doc_id)
    result["requestedBy"] = user.get("username")
    return result


@app.post("/documents/{document_id}/match-procedures")
async def match_document_procedures(
    document_id: str,
    req: UniversalAnalysisRequest,
    user: dict = Depends(get_current_user),
):
    """Return only the procedure matches for a document."""
    svc = get_doc_intelligence()
    result = await svc.analyze(req.document_text, req.filename, document_id)
    return {
        "documentId": document_id,
        "relatedProcedures": result.get("relatedProcedures", []),
        "possibleUses": result.get("possibleUses", []),
    }


@app.post("/documents/{document_id}/missing-requirements")
async def document_missing_requirements(
    document_id: str,
    req: UniversalAnalysisRequest,
    user: dict = Depends(get_current_user),
):
    """Return only missing fields + missing documents for a given document."""
    svc = get_doc_intelligence()
    result = await svc.analyze(req.document_text, req.filename, document_id)
    return {
        "documentId": document_id,
        "missingInformation": result.get("missingInformation", []),
        "missingDocuments":   result.get("missingDocuments", []),
    }


@app.post("/documents/{document_id}/recommended-drafts")
async def document_recommended_drafts(
    document_id: str,
    req: UniversalAnalysisRequest,
    user: dict = Depends(get_current_user),
):
    """Return recommended drafts for a document."""
    svc = get_doc_intelligence()
    result = await svc.analyze(req.document_text, req.filename, document_id)
    return {
        "documentId": document_id,
        "recommendedDrafts": result.get("recommendedDrafts", []),
    }


@app.get("/drafts/templates")
async def list_draft_templates(
    category: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """List all available draft templates, optionally filtered by document category."""
    templates = []
    for slug, meta in _DRAFT_CATALOGUE.items():
        applicable = [
            cat for cat, slugs in _DOCUMENT_DRAFTS_MAP.items() if slug in slugs
        ]
        if category and category not in applicable:
            continue
        templates.append({
            "slug": slug,
            "titleAr": meta["ar"],
            "titleEn": meta["en"],
            "requiresLawyerReview": meta["lawyer"],
            "applicableDocCategories": applicable,
        })
    return {"templates": templates, "total": len(templates)}


@app.get("/drafts/templates/{slug}")
async def get_draft_template(
    slug: str,
    user: dict = Depends(get_current_user),
):
    """Get a single draft template by slug."""
    meta = _DRAFT_CATALOGUE.get(slug)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Template '{slug}' not found")
    applicable = [cat for cat, slugs in _DOCUMENT_DRAFTS_MAP.items() if slug in slugs]
    return {
        "slug": slug,
        "titleAr": meta["ar"],
        "titleEn": meta["en"],
        "requiresLawyerReview": meta["lawyer"],
        "applicableDocCategories": applicable,
    }


@app.post("/drafts/generate")
async def generate_draft(
    req: DraftGenerateRequest,
    user: dict = Depends(get_current_user),
):
    """
    Generate a draft document using GPT-4o.
    Pre-fills from extracted facts and user inputs.
    Marks output clearly as a preliminary draft.
    """
    meta = _DRAFT_CATALOGUE.get(req.template_slug)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Template '{req.template_slug}' not found")

    title = meta["ar"] if req.language == "ar" else meta["en"]
    requires_lawyer = meta["lawyer"]

    # Build context from facts + user inputs
    facts_str = ""
    if req.extracted_facts:
        facts_str = "\n".join(f"{k}: {v}" for k, v in req.extracted_facts.items() if v)
    inputs_str = ""
    if req.user_inputs:
        inputs_str = "\n".join(f"{k}: {v}" for k, v in req.user_inputs.items() if v)

    lang_instruction = "باللغة العربية الرسمية" if req.language == "ar" else "in formal English"

    system_prompt = (
        "أنت محرر قانوني متخصص في إعداد مسودات الوثائق القانونية والإدارية اللبنانية. "
        "مهمتك صياغة مسودات أولية واضحة ومنظّمة. "
        "تأكد دائماً من وضع عبارة 'مسودة أولية' في أعلى الوثيقة. "
        "استخدم [PLACEHOLDER] لأي بيانات غير متوفرة. "
        "لا تخترع معلومات. لا تضمن نتائج قانونية."
    )

    user_prompt = f"""أعد مسودة "{title}" {lang_instruction}.

البيانات المتوفرة من المستند:
{facts_str or "(لا يوجد)"}

بيانات إضافية من المستخدم:
{inputs_str or "(لا يوجد)"}

متطلبات التنسيق:
- ابدأ بـ: ═══ مسودة أولية | {title} ═══
- نهاية الوثيقة: ─── نهاية المسودة ───
- استخدم [PLACEHOLDER] لكل بيان ناقص
- أضف في الأسفل: "ملاحظة: هذه مسودة أولية صادرة عن دليلك AI وليست وثيقة رسمية."
{"- أضف: 'يوصى بمراجعة محامٍ مرخّص قبل استعمال هذه المسودة.'" if requires_lawyer else ""}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        draft_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Draft generation failed: {e}")
        raise HTTPException(status_code=500, detail="فشل في توليد المسودة. حاول مرة أخرى.")

    # Detect missing placeholders
    import re
    placeholders = re.findall(r'\[PLACEHOLDER[^\]]*\]', draft_text)

    draft_id = str(uuid.uuid4())[:8]

    return {
        "draftId": draft_id,
        "title": title,
        "templateSlug": req.template_slug,
        "language": req.language,
        "draftText": draft_text,
        "missingFields": placeholders,
        "assumptions": [],
        "warnings": (
            ["يوصى بمراجعة محامٍ مرخّص قبل استعمال هذه المسودة."]
            if requires_lawyer else []
        ),
        "requiresLawyerReview": requires_lawyer,
        "sourceContext": {
            "type": "internal_template",
            "documentId": req.document_id,
            "procedureSlug": req.related_procedure_slug,
        },
        "status": "needs_review" if requires_lawyer else "draft",
        "disclaimer": (
            "هذه مسودة أولية صادرة عن دليلك AI وليست وثيقة رسمية. "
            "تأكد من صحة البيانات واستشر محامياً عند الحاجة."
        ),
        "requestedBy": user.get("username"),
    }



# ══ PHASE 2: PROCEDURE FLOWCHARTS ═══════════════════════════════════════════

_FLOWCHART_SEED = {
    "property-sale": {
        "procedureSlug": "property-sale",
        "titleAr": "بيع عقار",
        "titleEn": "Property Sale",
        "country": "lebanon",
        "version": "1.0",
        "verificationStatus": "partially_verified",
        "estimatedDurationAr": "4-8 أسابيع",
        "estimatedDurationEn": "4-8 weeks",
        "nodes": [
            {"id":"start","type":"start","titleAr":"بداية معاملة بيع العقار","titleEn":"Start Property Sale","status":"current"},
            {"id":"docs","type":"document","titleAr":"تجهيز المستندات","titleEn":"Prepare Documents","status":"not_started","descriptionAr":"سند الملكية، هوية البائع والمشتري، مخطط العقار","requiredDocuments":["سند الملكية","هوية شخصية","مخطط العقار","قيد عائلي"]},
            {"id":"notary","type":"authority","titleAr":"مراجعة الكاتب العدل","titleEn":"Visit Notary","status":"not_started","relatedAuthority":"notary-public"},
            {"id":"contract","type":"action","titleAr":"إبرام عقد البيع","titleEn":"Sign Sale Contract","status":"not_started","riskLevel":"high"},
            {"id":"risk_check","type":"risk","titleAr":"تحقق من بند الضمان","titleEn":"Verify Guarantee Clause","status":"not_started","descriptionAr":"تأكد من وجود بند الضمان وعدم وجود رهون"},
            {"id":"registry","type":"authority","titleAr":"التسجيل في السجل العقاري","titleEn":"Register in Land Registry","status":"not_started","relatedAuthority":"real-estate-registry"},
            {"id":"fees","type":"action","titleAr":"دفع رسوم التسجيل","titleEn":"Pay Registration Fees","status":"not_started"},
            {"id":"completion","type":"completion","titleAr":"اكتمال النقل","titleEn":"Transfer Complete","status":"not_started"}
        ],
        "edges": [
            {"id":"e1","from":"start","to":"docs"},
            {"id":"e2","from":"docs","to":"notary"},
            {"id":"e3","from":"notary","to":"contract"},
            {"id":"e4","from":"contract","to":"risk_check"},
            {"id":"e5","from":"risk_check","to":"registry","labelAr":"بعد التحقق"},
            {"id":"e6","from":"registry","to":"fees"},
            {"id":"e7","from":"fees","to":"completion"}
        ]
    },
    "power-of-attorney": {
        "procedureSlug": "power-of-attorney",
        "titleAr": "وكالة قانونية",
        "titleEn": "Power of Attorney",
        "country": "lebanon",
        "version": "1.0",
        "verificationStatus": "partially_verified",
        "estimatedDurationAr": "1-3 أيام",
        "estimatedDurationEn": "1-3 days",
        "nodes": [
            {"id":"start","type":"start","titleAr":"بداية إجراء الوكالة","titleEn":"Start POA","status":"current"},
            {"id":"scope","type":"question","titleAr":"تحديد صلاحيات الوكالة","titleEn":"Define POA Scope","descriptionAr":"هل هي وكالة بيع؟ متابعة معاملة؟ قبض؟"},
            {"id":"draft","type":"draft","titleAr":"إعداد نص الوكالة","titleEn":"Draft POA Text","descriptionAr":"يُنصح بصياغتها من محامٍ","riskLevel":"medium"},
            {"id":"risk","type":"risk","titleAr":"مراجعة الصلاحيات","titleEn":"Review Scope Risk","descriptionAr":"وكالة مفتوحة تحمل مخاطر عالية"},
            {"id":"notary","type":"authority","titleAr":"التوقيع أمام الكاتب العدل","titleEn":"Sign Before Notary","relatedAuthority":"notary-public"},
            {"id":"abroad","type":"question","titleAr":"هل الوكيل خارج لبنان؟","titleEn":"Attorney Abroad?"},
            {"id":"apostille","type":"action","titleAr":"تصديق وزارة الخارجية","titleEn":"Foreign Affairs Apostille"},
            {"id":"completion","type":"completion","titleAr":"الوكالة جاهزة","titleEn":"POA Ready"}
        ],
        "edges": [
            {"id":"e1","from":"start","to":"scope"},
            {"id":"e2","from":"scope","to":"draft"},
            {"id":"e3","from":"draft","to":"risk"},
            {"id":"e4","from":"risk","to":"notary","labelAr":"بعد المراجعة"},
            {"id":"e5","from":"notary","to":"abroad"},
            {"id":"e6","from":"abroad","to":"apostille","labelAr":"نعم"},
            {"id":"e7","from":"abroad","to":"completion","labelAr":"لا"},
            {"id":"e8","from":"apostille","to":"completion"}
        ]
    }
}

def _generic_flowchart(slug: str) -> dict:
    return {
        "procedureSlug": slug, "titleAr": "إجراء", "titleEn": "Procedure",
        "country": "lebanon", "version": "1.0", "verificationStatus": "draft",
        "nodes": [
            {"id":"start","type":"start","titleAr":"بداية الإجراء","titleEn":"Start","status":"current"},
            {"id":"docs","type":"document","titleAr":"تجهيز المستندات","titleEn":"Prepare Documents","status":"not_started"},
            {"id":"authority","type":"authority","titleAr":"مراجعة الجهة المختصة","titleEn":"Visit Authority","status":"not_started"},
            {"id":"submit","type":"action","titleAr":"تقديم الطلب","titleEn":"Submit Request","status":"not_started"},
            {"id":"completion","type":"completion","titleAr":"اكتمال المعاملة","titleEn":"Complete","status":"not_started"},
        ],
        "edges": [
            {"id":"e1","from":"start","to":"docs"},{"id":"e2","from":"docs","to":"authority"},
            {"id":"e3","from":"authority","to":"submit"},{"id":"e4","from":"submit","to":"completion"}
        ]
    }

@app.get("/procedures/{slug}/flowchart")
async def get_procedure_flowchart(slug: str, user: dict = Depends(get_current_user)):
    return _FLOWCHART_SEED.get(slug) or _generic_flowchart(slug)

# ══ PHASE 6: TRANSACTION COMPLETION SCORE ══════════════════════════════════

class CompletionScoreRequest(BaseModel):
    transaction_id: Optional[str] = None
    uploaded_doc_count: int = 0
    required_doc_count: int = 0
    has_missing_critical: bool = False
    risk_level: str = "low"
    procedure_slug: Optional[str] = None

@app.post("/transactions/completion-score")
async def get_completion_score(req: CompletionScoreRequest, user: dict = Depends(get_current_user)):
    doc_score = min(100, int((req.uploaded_doc_count / max(req.required_doc_count, 1)) * 100))
    risk_penalty = {"low": 0, "medium": 10, "high": 25, "critical": 40}.get(req.risk_level, 0)
    critical_penalty = 30 if req.has_missing_critical else 0
    overall = max(0, min(100, doc_score - risk_penalty - critical_penalty))
    if overall >= 80: status = "ready_for_review"
    elif overall >= 50: status = "partially_ready"
    else: status = "not_ready"
    return {
        "transactionId": req.transaction_id or "temp", "score": overall, "status": status,
        "missingCriticalItems": [],
        "blockingIssues": (["مستندات إلزامية ناقصة"] if req.has_missing_critical else []),
        "recommendedNextAction": "أكمل رفع المستندات المطلوبة" if overall < 80 else "راجع الملف مع محامٍ",
        "breakdown": {"documentsScore": doc_score, "dataScore": 70, "consistencyScore": 80, "riskScore": max(0, 100 - risk_penalty)}
    }

# ══ PHASE 9: AUTOPILOT SCAFFOLD ════════════════════════════════════════════

class AutopilotStartRequest(BaseModel):
    procedure_slug: str
    language: str = "ar"

@app.post("/autopilot/start")
async def autopilot_start(req: AutopilotStartRequest, user: dict = Depends(get_current_user)):
    return {
        "sessionId": str(uuid.uuid4())[:8], "procedureSlug": req.procedure_slug,
        "status": "waiting_answer", "currentStepIndex": 0, "totalSteps": 5,
        "answers": {}, "uploadedDocIds": [],
        "nextQuestion": {"questionAr": "ما هو هدفك من هذه المعاملة؟", "questionEn": "What is your goal?", "key": "goal", "type": "text"}
    }

@app.post("/autopilot/{session_id}/answer")
async def autopilot_answer(session_id: str, body: dict = {}, user: dict = Depends(get_current_user)):
    return {"sessionId": session_id, "status": "waiting_answer", "message": "تم تسجيل إجابتك — ميزة قيد التطوير"}

# ══ PHASE 10: AUTHORITY DIRECTORY ══════════════════════════════════════════

_AUTHORITY_DATA = [
    {"slug":"ministry-of-justice","nameAr":"وزارة العدل","nameEn":"Ministry of Justice","country":"lebanon","type":"ministry","proceduresHandled":["inheritance","civil-record","judicial"],"formsLinked":[],"confidence":"high","website":"https://justice.gov.lb"},
    {"slug":"real-estate-registry","nameAr":"دائرة السجل العقاري","nameEn":"Real Estate Registry","country":"lebanon","type":"registry","proceduresHandled":["property-sale","property-registration"],"formsLinked":[],"confidence":"high"},
    {"slug":"notary-public","nameAr":"الكاتب العدل","nameEn":"Notary Public","country":"lebanon","type":"notary","proceduresHandled":["power-of-attorney","contract-notarization","sale-contract"],"formsLinked":[],"confidence":"high"},
    {"slug":"ministry-of-finance","nameAr":"وزارة المالية","nameEn":"Ministry of Finance","country":"lebanon","type":"ministry","proceduresHandled":["tax-registration","company-formation","vat"],"formsLinked":[],"confidence":"high","website":"https://finance.gov.lb"},
    {"slug":"civil-registry","nameAr":"دائرة النفوس","nameEn":"Civil Registry","country":"lebanon","type":"registry","proceduresHandled":["civil-record","birth-registration","marriage-registration","name-change"],"formsLinked":[],"confidence":"high"},
    {"slug":"ministry-of-foreign-affairs","nameAr":"وزارة الخارجية","nameEn":"Ministry of Foreign Affairs","country":"lebanon","type":"ministry","proceduresHandled":["document-certification","apostille","expat-consular"],"formsLinked":[],"confidence":"high","website":"https://www.foreign.gov.lb"},
    {"slug":"commercial-registry","nameAr":"السجل التجاري","nameEn":"Commercial Registry","country":"lebanon","type":"registry","proceduresHandled":["company-formation","company-dissolution","company-amendment"],"formsLinked":[],"confidence":"high"},
]

@app.get("/authorities")
async def list_authorities(country: Optional[str] = None, user: dict = Depends(get_current_user)):
    data = [a for a in _AUTHORITY_DATA if not country or a.get("country") == country or a.get("country") == "both"]
    return {"authorities": data, "total": len(data)}

@app.get("/authorities/{slug}")
async def get_authority_detail(slug: str, user: dict = Depends(get_current_user)):
    found = next((a for a in _AUTHORITY_DATA if a["slug"] == slug), None)
    if not found:
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(404, "Authority not found")
    return found

# ══ PHASE 12: HUMAN REVIEW ══════════════════════════════════════════════════

class HumanReviewCreateRequest(BaseModel):
    request_type: str = "general"
    urgency: str = "normal"
    summary: str
    transaction_id: Optional[str] = None
    document_ids: list[str] = []

@app.post("/human-review/request")
async def create_human_review(req: HumanReviewCreateRequest, user: dict = Depends(get_current_user)):
    return {
        "id": str(uuid.uuid4())[:8], "userId": user.get("username"),
        "requestType": req.request_type, "urgency": req.urgency,
        "summary": req.summary, "status": "pending",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "message": "تم استلام طلب المراجعة. سيتواصل معك فريقنا خلال 24-48 ساعة.",
        "transactionId": req.transaction_id,
    }

@app.get("/human-review/requests")
async def list_human_reviews(user: dict = Depends(get_current_user)):
    return {"requests": [], "message": "Human review marketplace coming soon"}

# ══ PHASE 17: SHARE PACKAGE ════════════════════════════════════════════════

class SharePackageRequest(BaseModel):
    type: str = "checklist"
    title_ar: str
    title_en: str = ""
    transaction_id: Optional[str] = None

@app.post("/share/package")
async def create_share_package(req: SharePackageRequest, user: dict = Depends(get_current_user)):
    return {
        "shareId": str(uuid.uuid4())[:12], "type": req.type,
        "titleAr": req.title_ar, "expiresAt": None,
        "message": "⚠️ مشاركة الملفات قيد التطوير — احفظ في مساحة العمل في الوقت الحالي"
    }



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
