"""
RetrievalService — Professional RAG retrieval layer for Dalilak AI.
Abstracts all Qdrant retrieval with Arabic/English normalization,
confidence scoring, source extraction, and hybrid search preparation.
"""

import re
import time
import logging
import unicodedata
from typing import Optional

logger = logging.getLogger("dalilak.retrieval")

# ── Arabic normalization ─────────────────────────────────────────
_AR_ALEF = re.compile(r'[إأآا]')
_AR_YEH  = re.compile(r'[يى]')
_AR_HEH  = re.compile(r'[ةه]')


def _normalize_arabic(text: str) -> str:
    text = _AR_ALEF.sub('ا', text)
    text = _AR_YEH.sub('ي', text)
    text = _AR_HEH.sub('ه', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text.strip()


def _detect_language(text: str) -> str:
    arabic_chars = sum(1 for ch in text if '؀' <= ch <= 'ۿ')
    return 'ar' if arabic_chars > len(text) * 0.2 else 'en'


def _detect_country(text: str) -> Optional[str]:
    lb = ['لبنان', 'لبناني', 'بيروت', 'lebanon', 'lebanese', 'beirut']
    sy = ['سوريا', 'سوري', 'دمشق', 'syria', 'syrian', 'damascus']
    tl = text.lower()
    if any(k in tl for k in lb): return 'lebanon'
    if any(k in tl for k in sy): return 'syria'
    return None


_PROCEDURE_MAP: dict[str, list[str]] = {
    'passport': ['جواز', 'سفر', 'passport', 'travel document'],
    'civil-registry-extract': ['إخراج قيد', 'قيد', 'سجل مدني', 'civil record', 'civil registry', 'extract'],
    'birth-certificate': ['ولادة', 'مولود', 'شهادة ميلاد', 'birth certificate', 'birth registration'],
    'criminal-record': ['سجل عدلي', 'عدلية', 'criminal record', 'good conduct', 'عدم محكومية'],
    'marriage-registration': ['زواج', 'عقد زواج', 'marriage', 'wedding', 'تسجيل زواج'],
    'death-registration': ['وفاة', 'شهادة وفاة', 'death certificate', 'death registration'],
    'inheritance-certificate': ['إرث', 'وراثة', 'حصر إرث', 'inheritance', 'estate', 'تركة'],
    'power-of-attorney': ['وكالة', 'توكيل', 'power of attorney', 'poa'],
    'document-attestation': ['تصديق', 'تعميد', 'attestation', 'legalization', 'apostille', 'تثبيت'],
    'company-registration': ['شركة', 'تأسيس', 'company', 'incorporation', 'sarl', 'sal', 'تسجيل شركة'],
    'building-permit': ['بناء', 'رخصة بناء', 'building permit', 'construction permit', 'تصريح بناء'],
    'property-transfer': ['عقار', 'بيع', 'شراء', 'نقل ملكية', 'property', 'real estate transfer', 'تحويل ملكية'],
    'tax-registration': ['ضريبة', 'tax', 'vat', 'ضريبة القيمة المضافة', 'الضريبة'],
    'social-security': ['ضمان', 'اجتماعي', 'nssf', 'social security', 'الصندوق الوطني'],
    'driver-license': ['رخصة قيادة', 'driver license', "driver's license", 'driving license'],
    'vehicle-registration': ['تسجيل سيارة', 'vehicle registration', 'car registration', 'لوحة'],
    'municipality-permit': ['بلدية', 'إجازة بلدية', 'municipality permit', 'رخصة بلدية'],
}


def _detect_procedure(text: str) -> Optional[str]:
    tl = text.lower()
    for slug, keywords in _PROCEDURE_MAP.items():
        if any(k in tl for k in keywords):
            return slug
    return None


class RetrievalService:
    """
    Professional retrieval service for Dalilak AI.
    Dense semantic retrieval + keyword fallback + confidence scoring.
    """

    def __init__(self, qdrant_client, openai_client, collection: str, embed_model: str, embed_dim: int):
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.collection = collection
        self.embed_model = embed_model
        self.embed_dim = embed_dim

    # ── Query normalization ──────────────────────────────────────

    def normalize_query(self, query: str, language: Optional[str] = None) -> str:
        query = query.strip()
        lang = language or _detect_language(query)
        if lang == 'ar':
            query = _normalize_arabic(query)
        return re.sub(r'\s+', ' ', query)

    # ── Filter building ──────────────────────────────────────────

    def build_filters(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        procedure: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Optional[dict]:
        conditions = []
        if country:
            conditions.append({"key": "country", "match": {"value": country}})
        if category:
            conditions.append({"key": "category", "match": {"value": category}})
        if authority:
            conditions.append({"key": "authority", "match": {"value": authority}})
        if procedure:
            conditions.append({"key": "procedure_slug", "match": {"value": procedure}})
        if language:
            conditions.append({"key": "language", "match": {"value": language}})
        if not conditions:
            return None
        return {"must": conditions} if len(conditions) > 1 else conditions[0]

    # ── Embedding ────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        resp = self.openai.embeddings.create(
            model=self.embed_model,
            input=text,
            dimensions=self.embed_dim,
        )
        return resp.data[0].embedding

    # ── Semantic search ──────────────────────────────────────────

    def semantic_search(self, query: str, filters: Optional[dict] = None, top_k: int = 5) -> list[dict]:
        try:
            vector = self._embed(query)
            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                query_filter=filters,
                with_payload=True,
                score_threshold=0.20,
            )
            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "text": r.payload.get("text", r.payload.get("content", "")),
                    "title": r.payload.get("title", r.payload.get("source", "مصدر")),
                    "ministry": r.payload.get("ministry", r.payload.get("authority", "")),
                    "country": r.payload.get("country", ""),
                    "category": r.payload.get("category", ""),
                    "procedure_slug": r.payload.get("procedure_slug", ""),
                    "url": r.payload.get("url", ""),
                    "last_reviewed": r.payload.get("last_reviewed", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    # ── Keyword fallback ─────────────────────────────────────────

    def keyword_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Keyword fallback via Qdrant scroll with text match."""
        try:
            results, _ = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter={
                    "must": [{"key": "text", "match": {"text": query[:50]}}]
                },
                limit=top_k,
                with_payload=True,
            )
            return [
                {
                    "id": str(r.id),
                    "score": 0.30,
                    "text": r.payload.get("text", ""),
                    "title": r.payload.get("title", "مصدر"),
                    "ministry": r.payload.get("ministry", ""),
                    "country": r.payload.get("country", ""),
                    "category": r.payload.get("category", ""),
                    "procedure_slug": r.payload.get("procedure_slug", ""),
                    "url": r.payload.get("url", ""),
                    "last_reviewed": r.payload.get("last_reviewed", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.debug(f"Keyword search skipped: {e}")
            return []

    # ── Hybrid search ────────────────────────────────────────────

    def hybrid_search(self, query: str, filters: Optional[dict] = None, top_k: int = 5) -> list[dict]:
        semantic = self.semantic_search(query, filters, top_k=top_k)
        if len(semantic) >= 3:
            return semantic
        seen = {r["id"] for r in semantic}
        for r in self.keyword_search(query, top_k=3):
            if r["id"] not in seen:
                semantic.append(r)
                seen.add(r["id"])
        return semantic[:top_k]

    # ── Query analysis ───────────────────────────────────────────

    def analyze_query(self, query: str) -> dict:
        return {
            "language": _detect_language(query),
            "country": _detect_country(query),
            "procedure_slug": _detect_procedure(query),
        }

    # ── Reranking (placeholder) ──────────────────────────────────

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        # TODO: integrate Cohere Rerank or cross-encoder model
        return sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    # ── Confidence scoring ───────────────────────────────────────

    def calculate_confidence(self, chunks: list[dict]) -> str:
        """
        high   = max_score >= 0.50 AND avg_score >= 0.40
        medium = max_score >= 0.35
        low    = below thresholds or empty
        """
        if not chunks:
            return "low"
        scores = [c.get("score", 0) for c in chunks]
        max_s = max(scores)
        avg_s = sum(scores) / len(scores)
        if max_s >= 0.50 and avg_s >= 0.40:
            return "high"
        if max_s >= 0.35:
            return "medium"
        return "low"

    # ── Source extraction ────────────────────────────────────────

    def extract_sources(self, chunks: list[dict]) -> list[dict]:
        seen: set[str] = set()
        sources = []
        for c in chunks:
            title = c.get("title") or c.get("ministry") or "مصدر"
            if title in seen:
                continue
            seen.add(title)
            sources.append({
                "title": title,
                "ministry": c.get("ministry", ""),
                "score": round(c.get("score", 0), 3),
                "country": c.get("country", ""),
                "url": c.get("url", ""),
                "last_reviewed": c.get("last_reviewed", ""),
            })
        return sources

    # ── Low confidence prefix ────────────────────────────────────

    @staticmethod
    def low_confidence_prefix(language: str) -> str:
        if language == 'ar':
            return (
                "⚠️ **تنبيه:** لم أتمكن من التحقق من هذه المعلومات من مصادر موثوقة في قاعدة البيانات الحالية. "
                "ما يلي هو إرشاد عام فقط — يرجى التحقق من الجهة الرسمية المختصة قبل اتخاذ أي إجراء.\n\n"
            )
        return (
            "⚠️ **Notice:** I could not verify this from reliable sources in the current database. "
            "The following is general guidance only — please verify with the competent official authority before acting.\n\n"
        )

    # ── Full pipeline ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        country: Optional[str] = None,
        procedure: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Full pipeline: normalize → analyze → filter → search → rerank → score → sources.
        Returns: { chunks, confidence, sources, query_meta }
        """
        start = time.time()
        meta = self.analyze_query(query)
        eff_lang = language or meta["language"]
        eff_country = country or meta["country"]
        eff_proc = procedure or meta["procedure_slug"]

        norm_q = self.normalize_query(query, eff_lang)
        filters = self.build_filters(country=eff_country, procedure=eff_proc)
        chunks = self.hybrid_search(norm_q, filters=filters, top_k=top_k)
        chunks = self.rerank(norm_q, chunks)
        confidence = self.calculate_confidence(chunks)
        sources = self.extract_sources(chunks)

        elapsed = time.time() - start
        logger.info(
            f"Retrieval lang={eff_lang} country={eff_country} proc={eff_proc} "
            f"chunks={len(chunks)} conf={confidence} {elapsed*1000:.0f}ms"
        )

        return {
            "chunks": chunks,
            "confidence": confidence,
            "sources": sources,
            "query_meta": {
                "language": eff_lang,
                "country": eff_country,
                "procedure": eff_proc,
                "normalized_query": norm_q,
                "elapsed_ms": round(elapsed * 1000),
            },
        }
