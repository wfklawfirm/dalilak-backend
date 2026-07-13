# -*- coding: utf-8 -*-
"""
Phase 18 — Evidence Gate Unit Tests — Tier A (offline, direct function calls)

Tests the _evaluate_evidence function against all outcome branches:
  INSUFFICIENT, SUFFICIENT, PARTIAL, CONFLICTING.

No network, no Qdrant, no auth. Pure unit tests of business logic.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JWT_SECRET", "test-secret-32-chars-minimum-ok!")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.pop("ADMIN_USERNAME", None)
os.environ.pop("REDIS_URL", None)

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import main  # noqa: E402
from main import _evaluate_evidence, EvidenceOutcome  # noqa: E402

# Constants from main — replicated here so tests are self-documenting
_TOP_THRESHOLD = 0.35   # SUFFICIENCY_TOP_SCORE
_HIGH_Q_SCORE  = 0.45   # high-quality threshold in _evaluate_evidence


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunk(score: float, country: str = "lebanon") -> dict:
    return {
        "score": score,
        "title": "test chunk",
        "text": "test content",
        "ministry": "",
        "website": "",
        "phone": "",
        "country": country,
    }


# ── T18-01: empty chunks → INSUFFICIENT ──────────────────────────────────────

def test_p18_01_empty_chunks_insufficient():
    outcome, reason = _evaluate_evidence([])
    assert outcome == EvidenceOutcome.INSUFFICIENT
    assert "no_chunks" in reason


# ── T18-02: single chunk below threshold → INSUFFICIENT ──────────────────────

def test_p18_02_single_chunk_below_threshold_insufficient():
    # Score 0.28 < 0.35 threshold
    outcome, reason = _evaluate_evidence([_chunk(0.28)])
    assert outcome == EvidenceOutcome.INSUFFICIENT
    assert "top_score" in reason


def test_p18_02b_multiple_chunks_all_below_threshold():
    chunks = [_chunk(0.20), _chunk(0.30), _chunk(0.34)]
    outcome, _ = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.INSUFFICIENT


# ── T18-03: two high-quality chunks (score ≥ 0.45) → SUFFICIENT ──────────────

def test_p18_03_two_high_quality_chunks_sufficient():
    chunks = [_chunk(0.72), _chunk(0.65)]
    outcome, reason = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.SUFFICIENT
    assert "hq" in reason or "top" in reason


# ── T18-04: single chunk with top score ≥ 0.45 → SUFFICIENT ─────────────────

def test_p18_04_single_high_score_sufficient():
    # One chunk ≥ 0.45 but only one → still SUFFICIENT (top >= 0.45)
    outcome, reason = _evaluate_evidence([_chunk(0.50)])
    assert outcome == EvidenceOutcome.SUFFICIENT
    assert "top" in reason


def test_p18_04b_exactly_045_sufficient():
    outcome, _ = _evaluate_evidence([_chunk(0.45)])
    assert outcome == EvidenceOutcome.SUFFICIENT


# ── T18-05: chunks 0.37–0.44 (above 0.35, below 0.45) → PARTIAL ─────────────

def test_p18_05_above_threshold_below_high_quality_partial():
    # 0.40 > 0.35 but < 0.45 → PARTIAL
    chunks = [_chunk(0.40)]
    outcome, reason = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.PARTIAL
    assert "partial" in reason.lower() or "top" in reason


def test_p18_05b_two_partial_quality_chunks():
    chunks = [_chunk(0.38), _chunk(0.42)]
    outcome, _ = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.PARTIAL


def test_p18_05c_mix_partial_and_insufficient():
    # Top is 0.37 (above 0.35) — only one above threshold — PARTIAL
    chunks = [_chunk(0.20), _chunk(0.37)]
    outcome, _ = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.PARTIAL


# ── T18-06: Lebanon + Syria cross-country → CONFLICTING ──────────────────────

def test_p18_06_lebanon_and_syria_top_chunks_conflicting():
    # Both countries appear in top chunks (score >= 0.35)
    chunks = [
        _chunk(0.60, country="lebanon"),
        _chunk(0.55, country="syria"),
    ]
    outcome, reason = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.CONFLICTING
    assert "cross_country" in reason


def test_p18_06b_arabic_country_names_conflicting():
    chunks = [
        _chunk(0.50, country="لبنان"),
        _chunk(0.48, country="سوريا"),
    ]
    outcome, reason = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.CONFLICTING


def test_p18_06c_single_country_not_conflicting():
    # Only Lebanon — no conflict
    chunks = [_chunk(0.60, country="lebanon"), _chunk(0.55, country="lb")]
    outcome, _ = _evaluate_evidence(chunks)
    assert outcome != EvidenceOutcome.CONFLICTING


# ── T18-07: all chunks below 0.35 → INSUFFICIENT ────────────────────────────

def test_p18_07_all_below_threshold_insufficient():
    chunks = [_chunk(0.10), _chunk(0.25), _chunk(0.34)]
    outcome, _ = _evaluate_evidence(chunks)
    assert outcome == EvidenceOutcome.INSUFFICIENT


# ── T18-08: outcome string values are correct ─────────────────────────────────

def test_p18_08_outcome_constants_correct():
    assert EvidenceOutcome.SUFFICIENT == "SUFFICIENT"
    assert EvidenceOutcome.PARTIAL == "PARTIAL"
    assert EvidenceOutcome.INSUFFICIENT == "INSUFFICIENT"
    assert EvidenceOutcome.CONFLICTING == "CONFLICTING"


# ── T18-09: query parameter is accepted (smoke test) ─────────────────────────

def test_p18_09_query_parameter_accepted():
    outcome, reason = _evaluate_evidence([_chunk(0.50)], query="ما هو شرط تجديد الجواز؟")
    assert outcome == EvidenceOutcome.SUFFICIENT


# ── T18-10: _is_evidence_sufficient backward compat ─────────────────────────

def test_p18_10_is_evidence_sufficient_backward_compat():
    from main import _is_evidence_sufficient
    assert _is_evidence_sufficient([_chunk(0.50)]) is True
    assert _is_evidence_sufficient([_chunk(0.20)]) is False
    assert _is_evidence_sufficient([]) is False
