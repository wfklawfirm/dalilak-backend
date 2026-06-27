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
    username: str        # username OR email
    password: str

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

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
      type       : 'comparative' | 'legal' | 'eligibility' | 'procedural' | 'factual' | 'general'
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
            model  = pick_model(req.message, qinfo)
            msgs   = build_messages(ctx, req.history, req.message, qinfo)

            meta = {
                "type": "meta", "model": model, "chunks": len(chunks),
                "query_type": qinfo['type'],
                "sources": [{"title": c["title"], "ministry": c["ministry"], "score": c["score"]} for c in chunks[:5]],
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

class UploadDocumentRequest(BaseModel):
    file_base64: str
    file_name: str
    file_type: str  # "application/pdf", "application/msword", "image/jpeg", etc.
    user_note: Optional[str] = None

class DeleteDocumentRequest(BaseModel):
    doc_id: str

@app.post("/documents/upload")
async def upload_document(req: UploadDocumentRequest, user: dict = Depends(get_current_user)):
    """STUB: Accept a base64-encoded file, return a document ID. Replace with real storage."""
    import uuid, base64
    try:
        _ = base64.b64decode(req.file_base64)  # validate base64
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 content")
    doc_id = str(uuid.uuid4())
    return {
        "success": True,
        "doc_id": doc_id,
        "file_name": req.file_name,
        "file_type": req.file_type,
        "message": "STUB: Document received. Storage not yet implemented.",
        # TODO: Store in R2/S3, extract text via PyPDF2 or Textract, index in Qdrant
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    """STUB: Delete a stored document by ID."""
    return {
        "success": True,
        "doc_id": doc_id,
        "message": "STUB: Deletion acknowledged. Storage not yet implemented.",
    }

@app.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """STUB: List documents uploaded by the current user."""
    return {
        "documents": [],
        "message": "STUB: Document listing not yet implemented.",
    }


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
        client.upsert(
            collection_name="dalilak_logs",
            points=[PointStruct(id=str(uuid.uuid4()), vector=[0.0] * 3072, payload=payload)]
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
        results = client.scroll(
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
