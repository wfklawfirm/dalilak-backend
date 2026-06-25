#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dalilak AI — FastAPI Backend (Clean v2)"""

import os, json, time
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx

# ── Config ─────────────────────────────────────────────────
COLLECTION   = "dalilak_ai_v2"
EMBED_MODEL  = "text-embedding-3-large"
VECTOR_DIM   = 3072
MODEL_FAST   = "gpt-4o-mini"
MODEL_SMART  = "gpt-4o"
MIN_SCORE    = 0.28
MAX_CTX      = 12
MAX_TOKENS   = 2000
MAX_HISTORY  = 6
MAX_CHARS    = 12000

# ── System Prompt ──────────────────────────────────────────
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except:
    SYSTEM_PROMPT = "أنت دليلك AI، مساعد المواطن اللبناني في كل الشؤون الحكومية."

# ── Lazy clients ───────────────────────────────────────────
_oai: Optional[AsyncOpenAI] = None

def oai() -> AsyncOpenAI:
    global _oai
    if _oai is None:
        _oai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _oai

def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "").rstrip("/")

def qdrant_headers() -> dict:
    return {"api-key": os.environ.get("QDRANT_API_KEY", ""), "Content-Type": "application/json"}

# ── App ────────────────────────────────────────────────────
app = FastAPI(title="Dalilak AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

# ── Models ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    domain: Optional[str] = None

# ── Helpers ────────────────────────────────────────────────
async def embed(text: str) -> list:
    r = await oai().embeddings.create(
        model=EMBED_MODEL,
        input=[text[:MAX_CHARS]],
        dimensions=VECTOR_DIM,
    )
    return r.data[0].embedding

async def search(vec: list, domain: Optional[str] = None) -> list:
    body: dict = {
        "vector": vec,
        "limit": MAX_CTX,
        "score_threshold": MIN_SCORE,
        "with_payload": True,
    }
    if domain:
        body["filter"] = {"must": [{"key": "domain", "match": {"value": domain}}]}

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            f"{qdrant_url()}/collections/{COLLECTION}/points/search",
            headers=qdrant_headers(),
            json=body,
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
    keywords = ["نموذج","وثيقة","خطوات","إجراءات","اشرح","مقارن","form","document"]
    return MODEL_SMART if any(k in msg for k in keywords) or len(msg) > 200 else MODEL_FAST

def build_messages(ctx: str, history: list, user_msg: str) -> list:
    system = SYSTEM_PROMPT + (f"\n\n{ctx}" if ctx else "")
    msgs = [{"role": "system", "content": system}]
    for m in history[-MAX_HISTORY:]:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "name": "Dalilak AI", "version": "2.0.0"}

@app.get("/debug")
async def debug():
    qdrant_ok = "unknown"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{qdrant_url()}/collections/{COLLECTION}", headers=qdrant_headers())
            qdrant_ok = f"ok ({r.status_code}) — {r.json().get('result',{}).get('points_count','?')} points"
    except Exception as e:
        qdrant_ok = f"error: {e}"
    return {
        "OPENAI_API_KEY": "set" if os.environ.get("OPENAI_API_KEY") else "MISSING",
        "QDRANT_URL": qdrant_url(),
        "QDRANT_API_KEY": "set" if os.environ.get("QDRANT_API_KEY") else "MISSING",
        "qdrant": qdrant_ok,
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{qdrant_url()}/collections/{COLLECTION}", headers=qdrant_headers())
        pts = r.json().get("result", {}).get("points_count", 0)
        return {"status": "ok", "collection": COLLECTION, "points": pts, "timestamp": int(time.time())}
    except Exception as e:
        raise HTTPException(503, detail=str(e))

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        vec    = await embed(req.message)
        chunks = await search(vec, req.domain)
        ctx    = context_str(chunks)
        model  = pick_model(req.message)
        msgs   = build_messages(ctx, req.history, req.message)
        t0     = time.time()

        resp = await oai().chat.completions.create(
            model=model, messages=msgs,
            max_tokens=MAX_TOKENS, temperature=0.3,
        )
        return {
            "answer":      resp.choices[0].message.content,
            "model":       model,
            "chunks_used": len(chunks),
            "elapsed_s":   round(time.time() - t0, 2),
            "sources": [
                {"title": c["title"], "ministry": c["ministry"], "score": c["score"]}
                for c in chunks[:5]
            ],
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            vec    = await embed(req.message)
            chunks = await search(vec, req.domain)
            ctx    = context_str(chunks)
            model  = pick_model(req.message)
            msgs   = build_messages(ctx, req.history, req.message)

            meta = {
                "type": "meta", "model": model, "chunks": len(chunks),
                "sources": [{"title": c["title"], "ministry": c["ministry"], "score": c["score"]} for c in chunks[:5]],
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            stream = await oai().chat.completions.create(
                model=model, messages=msgs,
                max_tokens=MAX_TOKENS, temperature=0.3, stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type':'token','text':delta}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','detail':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
