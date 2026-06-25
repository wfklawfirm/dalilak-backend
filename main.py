#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dalilak AI — FastAPI Backend v3 (Document Intelligence)"""

import os, json, time, base64, io
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx

# ── Document text extraction ───────────────────────────────
def extract_text_from_pdf(b64: str) -> str:
    """Extract text from PDF base64 string using pdfplumber."""
    try:
        import pdfplumber
        raw = base64.b64decode(b64)
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages[:20]:  # max 20 pages
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts)[:15000]
    except Exception:
        try:
            import fitz  # PyMuPDF fallback
            raw = base64.b64decode(b64)
            doc = fitz.open(stream=raw, filetype="pdf")
            parts = [doc[i].get_text() for i in range(min(20, len(doc)))]
            doc.close()
            return "\n\n".join(parts)[:15000]
        except Exception as e:
            return f"[تعذّر استخراج نص PDF: {e}]"

def extract_text_from_docx(b64: str) -> str:
    """Extract text from Word .docx base64 string."""
    try:
        from docx import Document
        raw = base64.b64decode(b64)
        doc = Document(io.BytesIO(raw))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        # also grab tables
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
        return "\n".join(lines)[:15000]
    except Exception as e:
        return f"[تعذّر استخراج نص Word: {e}]"

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
MAX_DOC_TOKENS = 3500   # more tokens for document analysis

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

class AnalyzeRequest(BaseModel):
    file_base64: str
    file_type: str          # e.g. "image/jpeg", "application/pdf"
    file_name: str
    message: str = "حلل هذه الوثيقة واقترح الإجراءات المناسبة"
    history: list[Message] = []

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
                    # Send both formats for compatibility
                    token_data = {
                        "type": "token",
                        "text": delta,
                        "choices": [{"delta": {"content": delta}}]
                    }
                    yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','detail':str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            is_image  = req.file_type.startswith("image/")
            is_pdf    = req.file_type == "application/pdf"
            is_word   = "word" in req.file_type or req.file_name.lower().endswith((".docx", ".doc"))
            is_text   = req.file_type.startswith("text/") or req.file_name.lower().endswith(".txt")

            # ── Extract text from non-image documents ──────────────────
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

            # ── Search Qdrant for relevant context ─────────────────────
            # Use filename + user message + first 300 chars of extracted text as search query
            search_query = f"{req.file_name} {req.message} {extracted_text[:300]}"
            try:
                vec    = await embed(search_query)
                chunks = await search(vec)
                ctx    = context_str(chunks)
            except Exception:
                ctx = ""

            # ── Build the master analysis prompt ───────────────────────
            ANALYSIS_PROMPT = SYSTEM_PROMPT + """

---

## قواعد تحليل الوثائق

أنت خبير متخصص في تحليل الوثائق الرسمية والقانونية اللبنانية.
عند تحليل أي وثيقة، اتبع هذا الهيكل الإلزامي بالترتيب:

### 1. 📋 تشخيص الوثيقة
- نوعها الدقيق (عقد / قرار / طلب / فاتورة / قيد / وكالة / حكم / مراسيم / إلخ)
- الجهة المُصدِرة والجهة المُستلِمة
- التاريخ ورقم المرجع إن وجد

### 2. 📌 استخراج البيانات الجوهرية
استخرج كل المعلومات المهمة: أسماء، أرقام، مبالغ، مواعيد، شروط، التزامات.

### 3. ⚠️ التنبيهات والمخاطر
هل هناك:
- مواعيد نهائية قريبة؟
- بنود مُلزِمة أو غرامات؟
- إجراءات واجبة قانونياً لم تُنفَّذ بعد؟
- تناقضات أو ثغرات يجب الانتباه إليها؟

### 4. ✅ الإجراءات العملية المطلوبة (بالترتيب)
خطوات واضحة ومرقّمة يجب على المواطن اتخاذها فوراً وعلى المدى القريب.

### 5. 📁 المستندات والمتطلبات
ما يجب تحضيره: وثائق، صور، طوابع، رسوم، أشخاص مطلوب حضورهم.

### 6. 🏛️ الجهة المختصة والتواصل
الوزارة أو الدائرة المختصة، رقم الهاتف إن توفّر في قاعدة البيانات، ساعات العمل.

### 7. 📝 النموذج أو المسودة الجاهزة
**إلزامي:** إذا كانت الوثيقة تستوجب تقديم طلب أو إفادة أو عقد:
- إذا وُجد النموذج الرسمي في قاعدة البيانات: اذكره باسمه وأين يُحصل عليه.
- إذا لم يُوجد: **اكتب مسودة جاهزة للتعديل** بصيغة رسمية كاملة، مع جميع الحقول اللازمة.

---
""" + (f"\n\n{ctx}" if ctx else "")

            # ── Compose the user turn ──────────────────────────────────
            user_text = f"سؤال/طلب المستخدم: {req.message}\n\nاسم الملف: {req.file_name}"

            if extracted_text and not extracted_text.startswith("[تعذّر"):
                user_text += f"\n\n--- نص الوثيقة المستخرج ---\n{extracted_text}\n--- نهاية النص ---"
            elif not is_image:
                user_text += f"\n\n(نوع الملف: {req.file_type} — يُرجى التحليل بناءً على اسم الملف والسياق)"

            if is_image:
                user_content: list = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{req.file_type};base64,{req.file_base64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": user_text},
                ]
            else:
                user_content = [{"type": "text", "text": user_text}]

            msgs: list = [{"role": "system", "content": ANALYSIS_PROMPT}]
            for m in req.history[-MAX_HISTORY:]:
                msgs.append({"role": m.role, "content": m.content})
            msgs.append({"role": "user", "content": user_content})

            # ── Stream response ────────────────────────────────────────
            stream = await oai().chat.completions.create(
                model=MODEL_SMART,
                messages=msgs,
                max_tokens=MAX_DOC_TOKENS,
                temperature=0.2,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type':'token','text':delta,'choices':[{'delta':{'content':delta}}]}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
