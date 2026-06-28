"""
Dalilak AI — Database Models & Repository Layer (Phase 1)

SQLAlchemy-based domain models for the full platform.
• Works with SQLite (default, zero-config for dev/Render free tier)
• Swap DATABASE_URL to postgresql://... for production PostgreSQL
• All Qdrant usage stays for vector search ONLY — relational data moves here

Usage:
  from database import init_db, db_session, repo
  init_db()   # call once at startup
  with db_session() as session:
      repo.content_gaps.create(session, {...})
"""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ── Engine Setup ──────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(os.path.dirname(__file__), 'dalilak.db')}"
)

# SQLite needs WAL mode for concurrent access on Render
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)

# Enable SQLite WAL mode
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════
#  DOMAIN MODELS
# ═══════════════════════════════════════════════════════════════

class ContentGap(Base):
    """
    Auto-created when:
    - Retrieval confidence is low
    - User submits thumbs-down feedback
    - Procedure/form/fee is explicitly unknown
    """
    __tablename__ = "content_gaps"

    id              = Column(String, primary_key=True, default=_uuid)
    user_question   = Column(Text, nullable=False)
    detected_country    = Column(String(20), nullable=True)   # lebanon|syria|both|unknown
    detected_procedure  = Column(String(100), nullable=True)  # procedure slug
    detected_category   = Column(String(100), nullable=True)
    gap_type        = Column(String(50), nullable=False)
    # missing_procedure|missing_form|missing_fee|missing_source
    # |low_confidence|user_reported_error|unclear_authority|other
    related_answer_id   = Column(String, nullable=True)   # session/message id
    confidence_score    = Column(Float, nullable=True)    # 0.0–1.0 from retrieval
    status          = Column(String(20), default="open")  # open|in_review|resolved|ignored
    priority        = Column(String(10), default="medium")  # low|medium|high|critical
    admin_notes     = Column(Text, nullable=True)
    username        = Column(String(100), nullable=True)  # user who triggered it (anonymized)
    created_at      = Column(DateTime, default=_now)
    resolved_at     = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_question": self.user_question,
            "detected_country": self.detected_country,
            "detected_procedure": self.detected_procedure,
            "detected_category": self.detected_category,
            "gap_type": self.gap_type,
            "related_answer_id": self.related_answer_id,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "priority": self.priority,
            "admin_notes": self.admin_notes,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class UpdateLog(Base):
    """
    Audit trail for admin review actions on any entity.
    """
    __tablename__ = "update_logs"

    id              = Column(String, primary_key=True, default=_uuid)
    entity_type     = Column(String(50), nullable=False)  # procedure|form|source|content_gap|user
    entity_id       = Column(String, nullable=False)
    action          = Column(String(50), nullable=False)  # created|updated|verified|deactivated|resolved
    previous_status = Column(String(50), nullable=True)
    new_status      = Column(String(50), nullable=True)
    reviewer        = Column(String(100), nullable=True)  # admin username
    notes           = Column(Text, nullable=True)
    timestamp       = Column(DateTime, default=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "reviewer": self.reviewer,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class EvaluationQuestion(Base):
    """
    Golden question used for accuracy evaluation.
    """
    __tablename__ = "evaluation_questions"

    id                      = Column(String, primary_key=True, default=_uuid)
    external_id             = Column(String(50), nullable=True)  # from golden_questions.json
    category                = Column(String(100), nullable=False)
    language                = Column(String(5), nullable=False)   # ar|en
    country                 = Column(String(20), nullable=False)
    question                = Column(Text, nullable=False)
    expected_procedure_slug = Column(String(100), nullable=True)
    expected_keywords       = Column(Text, nullable=True)          # JSON array
    expected_authority      = Column(String(200), nullable=True)
    expected_confidence_min = Column(String(10), nullable=True)    # high|medium|low
    notes                   = Column(Text, nullable=True)
    active                  = Column(Boolean, default=True)
    created_at              = Column(DateTime, default=_now)


class EvaluationRun(Base):
    """
    A single run of the evaluation harness.
    """
    __tablename__ = "evaluation_runs"

    id              = Column(String, primary_key=True, default=_uuid)
    run_date        = Column(DateTime, default=_now)
    total_questions = Column(Integer, default=0)
    passed          = Column(Integer, default=0)
    failed          = Column(Integer, default=0)
    avg_confidence  = Column(Float, nullable=True)
    retrieval_hit_rate  = Column(Float, nullable=True)
    notes           = Column(Text, nullable=True)
    run_by          = Column(String(100), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_date": self.run_date.isoformat() if self.run_date else None,
            "total_questions": self.total_questions,
            "passed": self.passed,
            "failed": self.failed,
            "avg_confidence": self.avg_confidence,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "notes": self.notes,
            "run_by": self.run_by,
        }


class HumanEscalationRecord(Base):
    """
    Persisted escalation requests (supplements Qdrant logs).
    """
    __tablename__ = "human_escalations"

    id              = Column(String, primary_key=True, default=_uuid)
    username        = Column(String(100), nullable=True)
    request_type    = Column(String(50), nullable=False)
    procedure_slug  = Column(String(100), nullable=True)
    question        = Column(Text, nullable=True)
    context         = Column(Text, nullable=True)
    contact_preference  = Column(String(20), nullable=True)
    user_email      = Column(String(200), nullable=True)
    user_phone      = Column(String(50), nullable=True)
    status          = Column(String(20), default="pending")  # pending|assigned|completed|cancelled
    priority        = Column(String(10), default="medium")
    admin_notes     = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=_now)
    resolved_at     = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "request_type": self.request_type,
            "procedure_slug": self.procedure_slug,
            "question": self.question,
            "contact_preference": self.contact_preference,
            "user_email": self.user_email,
            "user_phone": self.user_phone,
            "status": self.status,
            "priority": self.priority,
            "admin_notes": self.admin_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class TransactionFile(Base):
    """Persisted transaction file — the core document of user's procedure journey."""
    __tablename__ = "transaction_files"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id         = Column(String(100), nullable=False, index=True)
    title           = Column(String(500), nullable=False)
    procedure_slug  = Column(String(100), nullable=True)
    country         = Column(String(20), default="lebanon")
    user_type       = Column(String(50), nullable=True)
    status          = Column(String(20), default="draft")
    summary         = Column(Text, nullable=True)
    required_documents  = Column(Text, default="[]")
    uploaded_doc_ids    = Column(Text, default="[]")
    missing_documents   = Column(Text, default="[]")
    steps               = Column(Text, default="[]")
    risk_level          = Column(String(20), nullable=True)
    risk_score          = Column(Float, nullable=True)
    risk_reasons        = Column(Text, default="[]")
    next_actions        = Column(Text, default="[]")
    sources             = Column(Text, default="[]")
    notes               = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=_now)
    updated_at      = Column(DateTime, default=_now)

    def to_dict(self) -> dict:
        import json as _json
        def _j(v):
            try: return _json.loads(v) if v else []
            except Exception: return []
        return {
            "id": self.id, "user_id": self.user_id, "title": self.title,
            "procedure_slug": self.procedure_slug, "country": self.country,
            "user_type": self.user_type, "status": self.status, "summary": self.summary,
            "required_documents": _j(self.required_documents),
            "uploaded_doc_ids": _j(self.uploaded_doc_ids),
            "missing_documents": _j(self.missing_documents),
            "steps": _j(self.steps), "risk_level": self.risk_level,
            "risk_score": self.risk_score, "risk_reasons": _j(self.risk_reasons),
            "next_actions": _j(self.next_actions), "sources": _j(self.sources),
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class UploadedDocument(Base):
    """Persisted uploaded document — text extracted server-side, never in localStorage."""
    __tablename__ = "uploaded_documents"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id         = Column(String(100), nullable=False, index=True)
    transaction_id  = Column(String, nullable=True)
    file_name       = Column(String(500), nullable=False)
    file_type       = Column(String(100), nullable=False)
    file_size       = Column(Integer, nullable=True)
    extracted_text  = Column(Text, nullable=True)
    doc_type        = Column(String(100), nullable=True)
    detected_country = Column(String(20), nullable=True)
    analysis_result = Column(Text, nullable=True)
    risk_review     = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=_now)

    def to_dict(self, include_text: bool = False) -> dict:
        d = {
            "id": self.id, "user_id": self.user_id,
            "transaction_id": self.transaction_id,
            "file_name": self.file_name, "file_type": self.file_type,
            "file_size": self.file_size, "doc_type": self.doc_type,
            "detected_country": self.detected_country,
            "has_analysis": bool(self.analysis_result),
            "has_risk_review": bool(self.risk_review),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_text:
            d["extracted_text_preview"] = (self.extracted_text or "")[:500]
        return d


# ═══════════════════════════════════════════════════════════════
#  REPOSITORY LAYER
# ═══════════════════════════════════════════════════════════════

class ContentGapRepo:
    """Repository for ContentGap CRUD operations."""

    def create(
        self,
        session: Session,
        *,
        user_question: str,
        gap_type: str = "low_confidence",
        detected_country: Optional[str] = None,
        detected_procedure: Optional[str] = None,
        detected_category: Optional[str] = None,
        confidence_score: Optional[float] = None,
        related_answer_id: Optional[str] = None,
        username: Optional[str] = None,
        priority: str = "medium",
    ) -> ContentGap:
        gap = ContentGap(
            id=_uuid(),
            user_question=user_question[:1000],
            gap_type=gap_type,
            detected_country=detected_country,
            detected_procedure=detected_procedure,
            detected_category=detected_category,
            confidence_score=confidence_score,
            related_answer_id=related_answer_id,
            username=username,
            priority=priority,
        )
        session.add(gap)
        return gap

    def list_open(self, session: Session, limit: int = 100) -> list[ContentGap]:
        return (
            session.query(ContentGap)
            .filter(ContentGap.status.in_(["open", "in_review"]))
            .order_by(ContentGap.created_at.desc())
            .limit(limit)
            .all()
        )

    def list_all(self, session: Session, limit: int = 200) -> list[ContentGap]:
        return (
            session.query(ContentGap)
            .order_by(ContentGap.created_at.desc())
            .limit(limit)
            .all()
        )

    def get(self, session: Session, gap_id: str) -> Optional[ContentGap]:
        return session.query(ContentGap).filter_by(id=gap_id).first()

    def update_status(
        self,
        session: Session,
        gap_id: str,
        status: str,
        admin_notes: Optional[str] = None,
        reviewer: Optional[str] = None,
    ) -> Optional[ContentGap]:
        gap = self.get(session, gap_id)
        if not gap:
            return None
        prev = gap.status
        gap.status = status
        if admin_notes:
            gap.admin_notes = admin_notes
        if status == "resolved":
            gap.resolved_at = _now()
        log = UpdateLog(
            entity_type="content_gap",
            entity_id=gap_id,
            action=f"status_changed_to_{status}",
            previous_status=prev,
            new_status=status,
            reviewer=reviewer,
            notes=admin_notes,
        )
        session.add(log)
        return gap

    def stats(self, session: Session) -> dict:
        total = session.query(ContentGap).count()
        open_ = session.query(ContentGap).filter_by(status="open").count()
        in_review = session.query(ContentGap).filter_by(status="in_review").count()
        high = session.query(ContentGap).filter(
            ContentGap.priority.in_(["high", "critical"]),
            ContentGap.status.in_(["open", "in_review"]),
        ).count()
        return {"total": total, "open": open_, "in_review": in_review, "high_priority": high}


class UpdateLogRepo:
    def list(self, session: Session, entity_type: Optional[str] = None, limit: int = 100) -> list[UpdateLog]:
        q = session.query(UpdateLog)
        if entity_type:
            q = q.filter_by(entity_type=entity_type)
        return q.order_by(UpdateLog.timestamp.desc()).limit(limit).all()


class EscalationRepo:
    def create(self, session: Session, **kwargs) -> HumanEscalationRecord:
        r = HumanEscalationRecord(id=_uuid(), **kwargs)
        session.add(r)
        return r

    def list_pending(self, session: Session, limit: int = 50) -> list[HumanEscalationRecord]:
        return (
            session.query(HumanEscalationRecord)
            .filter(HumanEscalationRecord.status.in_(["pending", "assigned"]))
            .order_by(HumanEscalationRecord.created_at.desc())
            .limit(limit)
            .all()
        )

    def update_status(self, session: Session, esc_id: str, status: str, notes: str = None):
        r = session.query(HumanEscalationRecord).filter_by(id=esc_id).first()
        if r:
            r.status = status
            if notes:
                r.admin_notes = notes
            if status in ("completed", "cancelled"):
                r.resolved_at = _now()
        return r


class EvalRepo:
    def create_run(self, session: Session, **kwargs) -> EvaluationRun:
        r = EvaluationRun(id=_uuid(), **kwargs)
        session.add(r)
        return r

    def list_runs(self, session: Session, limit: int = 20) -> list[EvaluationRun]:
        return (
            session.query(EvaluationRun)
            .order_by(EvaluationRun.run_date.desc())
            .limit(limit)
            .all()
        )


class TransactionRepo:
    """Repository for TransactionFile CRUD operations."""

    _JSON_FIELDS = frozenset({
        "required_documents", "missing_documents", "steps",
        "risk_reasons", "next_actions", "sources", "uploaded_doc_ids",
    })

    def create(self, session: Session, *, user_id: str, title: str, **kwargs) -> TransactionFile:
        import json as _json
        init: dict = {"id": _uuid(), "user_id": user_id, "title": title}
        for k, v in kwargs.items():
            if k in self._JSON_FIELDS:
                init[k] = _json.dumps(v if v is not None else [])
            else:
                init[k] = v
        tx = TransactionFile(**init)
        session.add(tx)
        return tx

    def list_by_user(self, session: Session, user_id: str, limit: int = 50) -> list[TransactionFile]:
        return (
            session.query(TransactionFile)
            .filter(TransactionFile.user_id == user_id,
                    TransactionFile.status != "archived")
            .order_by(TransactionFile.updated_at.desc())
            .limit(limit).all()
        )

    def get(self, session: Session, tx_id: str, user_id: str) -> Optional[TransactionFile]:
        return session.query(TransactionFile).filter_by(id=tx_id, user_id=user_id).first()

    def update(self, session: Session, tx_id: str, user_id: str, **fields) -> Optional[TransactionFile]:
        import json as _json
        tx = self.get(session, tx_id, user_id)
        if not tx:
            return None
        for k, v in fields.items():
            if k in self._JSON_FIELDS:
                setattr(tx, k, _json.dumps(v if v is not None else []))
            else:
                setattr(tx, k, v)
        tx.updated_at = _now()
        return tx

    def delete(self, session: Session, tx_id: str, user_id: str) -> bool:
        tx = self.get(session, tx_id, user_id)
        if not tx:
            return False
        session.delete(tx)
        return True


class DocumentRepo:
    """Repository for UploadedDocument CRUD operations."""

    def create(self, session: Session, *, user_id: str, file_name: str, file_type: str, **kwargs) -> UploadedDocument:
        doc = UploadedDocument(id=_uuid(), user_id=user_id, file_name=file_name, file_type=file_type, **kwargs)
        session.add(doc)
        return doc

    def get(self, session: Session, doc_id: str, user_id: str) -> Optional[UploadedDocument]:
        return session.query(UploadedDocument).filter_by(id=doc_id, user_id=user_id).first()

    def list_by_user(self, session: Session, user_id: str, limit: int = 50) -> list[UploadedDocument]:
        return (
            session.query(UploadedDocument)
            .filter_by(user_id=user_id)
            .order_by(UploadedDocument.created_at.desc())
            .limit(limit).all()
        )

    def update(self, session: Session, doc_id: str, user_id: str, **fields) -> Optional[UploadedDocument]:
        doc = self.get(session, doc_id, user_id)
        if not doc:
            return None
        for k, v in fields.items():
            setattr(doc, k, v)
        return doc

    def delete(self, session: Session, doc_id: str, user_id: str) -> bool:
        doc = self.get(session, doc_id, user_id)
        if not doc:
            return False
        session.delete(doc)
        return True

    def get_texts_for_ids(self, session: Session, doc_ids: list, user_id: str) -> list[dict]:
        """Return [{id, file_name, extracted_text}] for doc_ids owned by user."""
        if not doc_ids:
            return []
        docs = (
            session.query(UploadedDocument)
            .filter(
                UploadedDocument.id.in_(doc_ids),
                UploadedDocument.user_id == user_id,
            ).all()
        )
        return [{"id": d.id, "file_name": d.file_name, "extracted_text": d.extracted_text or ""} for d in docs]


class _Repo:
    """Namespace for all repositories."""
    content_gaps = ContentGapRepo()
    update_logs  = UpdateLogRepo()
    escalations  = EscalationRepo()
    evals        = EvalRepo()
    transactions = TransactionRepo()
    documents    = DocumentRepo()


repo = _Repo()


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables (idempotent). Call once at startup."""
    Base.metadata.create_all(bind=engine)
