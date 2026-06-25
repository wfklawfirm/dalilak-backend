#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dalilak AI — FastAPI Backend
=============================
POST /chat        → إجابة كاملة (JSON)
POST /chat/stream → إجابة مباشرة (SSE streaming)
GET  /health      → حالة الخادم
"""

import os
import json
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ──────────────────────────────────────────────────────────
# الإعدادات
# ──────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY",  "")
QDRANT_URL      = os.environ.get("QDRANT_URL",      "")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY",  "")

COLLECTION_NAME = "dalilak_ai_v2"
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIM      = 3072

MODEL_SIMPLE    = "gpt-4o-mini"   # للأسئلة العامة
MODEL_COMPLEX   = "gpt-4o"        # للأسئلة المعقدة والنماذج

MIN_SCORE       = 0.28
MAX_CONTEXT     = 12
MAX_TOKENS      = 2000
MAX_HISTORY     = 6

# ──────────────────────────────────────────────────────────
# تحميل System Prompt
# ──────────────────────────────────────────────────────────
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "أنت دليلك AI، مساعد المواطن اللبناني في كل الشؤون الحكومية."

# ──────────────────────────────────────────────────────────
# تهيئة العملاء (lazy — تُقرأ المتغيرات عند أول طلب)
# ──────────────────────────────────────────────────────────
_oai    = None
_qdrant = None

def get_oai():
    global _oai
    if _oai is None:
        _oai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _oai

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=os.environ.get("QDRANT_URL", ""),
            api_key=os.environ.get("QDRANT_API_KEY", ""),
            timeout=30,
        )
    return _qdrant

# ──────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Dalilak AI API",
    description="دليل المواطن اللبناني — API",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"status": "ok", "name": "Dalilak AI", "version": "1.0.0"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# المودل
# ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    domain: str | None = None      # تصفية اختيارية بالمجال
    stream: bool = False

# ──────────────────────────────────────────────────────────
# دوال مساعدة
# ──────────────────────────────────────────────────────────
async def get_embedding(text: str) -> list[float]:
    resp = await get_oai().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text[:12000]],
        dimensions=VECTOR_DIM,
    )
    return resp.data[0].embedding


def search_qdrant(query_vec: list[float], domain: str | None = None) -> list[dict]:
    """يبحث في Qdrant ويرجع chunks ذات الصلة"""
    qdrant_filter = None
    if domain:
        qdrant_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    results = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=MAX_CONTEXT,
        score_threshold=MIN_SCORE,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points

    chunks = []
    for r in results:
        p = r.payload
        chunks.append({
            "score":    round(r.score, 3),
            "title":    p.get("title", ""),
            "text":     p.get("text", ""),
            "domain":   p.get("domain", ""),
            "ministry": p.get("ministry", ""),
            "source":   p.get("source", ""),
            "website":  p.get("website", ""),
            "phone":    p.get("phone", ""),
            "fees":     p.get("fees", ""),
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """يبني نص السياق من الـ chunks"""
    if not chunks:
        return ""
    parts = ["=== المعلومات المتاحة ===\n"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['title']}")
        if c.get("ministry"):
            parts.append(f"الجهة: {c['ministry']}")
        parts.append(c["text"])
        if c.get("website"):
            parts.append(f"الموقع: {c['website']}")
        if c.get("phone"):
            parts.append(f"الهاتف: {c['phone']}")
        parts.append("---")
    return "\n".join(parts)


def is_complex(message: str) -> bool:
    """يحدد إذا كان السؤال يحتاج GPT-4o أم GPT-4o-mini"""
    complex_keywords = [
        "نموذج", "فورم", "وثيقة", "مقارن", "الفرق بين", "اشرح بالتفصيل",
        "خطوات", "إجراءات", "كيف أسجل", "كيف أؤسس", "ما هي شروط",
        "form", "document", "compare", "detailed",
    ]
    return any(k in message.lower() for k in complex_keywords) or len(message) > 200


def build_messages(system: str, context: str, history: list[Message], user_msg: str) -> list:
    """يبني قائمة الرسائل لـ OpenAI"""
    full_system = system
    if context:
        full_system += f"\n\n{context}"

    messages = [{"role": "system", "content": full_system}]

    # آخر MAX_HISTORY رسائل
    for m in history[-(MAX_HISTORY):]:
        messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": user_msg})
    return messages


# ──────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────
@app.get("/debug")
async def debug():
    import httpx
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_key = os.environ.get("QDRANT_API_KEY", "")
    qdrant_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{qdrant_url}/collections",
                headers={"api-key": qdrant_key},
            )
            qdrant_status = f"ok ({r.status_code})"
    except Exception as e:
        qdrant_status = f"error: {e}"
    return {
        "OPENAI_API_KEY": "set" if os.environ.get("OPENAI_API_KEY") else "MISSING",
        "QDRANT_URL": qdrant_url,
        "QDRANT_API_KEY": "set" if qdrant_key else "MISSING",
        "qdrant_connectivity": qdrant_status,
    }

@app.get("/health")
async def health():
    try:
        info = get_qdrant().get_collection(COLLECTION_NAME)
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "points": info.points_count,
            "timestamp": int(time.time()),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """إجابة كاملة — مناسبة للاختبار والتطبيقات البسيطة"""
    t0 = time.time()

    # 1. embedding للسؤال
    query_vec = await get_embedding(req.message)

    # 2. بحث في Qdrant
    chunks = search_qdrant(query_vec, req.domain)
    context = build_context(chunks)

    # 3. اختيار الموديل
    model = MODEL_COMPLEX if is_complex(req.message) else MODEL_SIMPLE

    # 4. بناء الرسائل
    messages = build_messages(SYSTEM_PROMPT, context, req.history, req.message)

    # 5. OpenAI
    resp = await oai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
    )

    answer = resp.choices[0].message.content
    elapsed = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "model": model,
        "chunks_used": len(chunks),
        "elapsed_s": elapsed,
        "sources": [
            {"title": c["title"], "ministry": c["ministry"], "score": c["score"]}
            for c in chunks[:5]
        ],
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """إجابة مباشرة بـ SSE — للواجهة الأمامية"""

    async def generate() -> AsyncGenerator[str, None]:
        # 1. embedding
        query_vec = await get_embedding(req.message)

        # 2. بحث
        chunks = search_qdrant(query_vec, req.domain)
        context = build_context(chunks)

        # 3. موديل
        model = MODEL_COMPLEX if is_complex(req.message) else MODEL_SIMPLE

        # 4. رسائل
        messages = build_messages(SYSTEM_PROMPT, context, req.history, req.message)

        # 5. metadata أولاً
        meta = {
            "type": "meta",
            "model": model,
            "chunks": len(chunks),
            "sources": [
                {"title": c["title"], "ministry": c["ministry"], "score": c["score"]}
                for c in chunks[:5]
            ],
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 6. stream الإجابة
        stream = await oai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                payload = {"type": "token", "text": delta}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 7. إشارة الانتهاء
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────
# تشغيل محلي
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Dalilak AI Backend يعمل على http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
