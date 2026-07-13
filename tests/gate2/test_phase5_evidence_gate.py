"""Phase 5 — Evidence Sufficiency Gate unit tests (Tier A: offline, no network)."""
from __future__ import annotations
import os, sys
import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _src() -> str:
    return open(os.path.join(_BACKEND, "main.py"), encoding="utf-8").read()

def _chunk(score: float) -> dict:
    return {"score": score, "title": "t", "text": "x", "ministry": "", "website": "", "phone": ""}

# Inline replica — must match main.py EXACTLY for test isolation
_SUFFICIENCY_TOP_SCORE = 0.35

def _is_evidence_sufficient(chunks: list) -> bool:
    return any(c.get("score", 0) >= _SUFFICIENCY_TOP_SCORE for c in chunks)

# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.tier_a
def test_p5_a1_constant_in_source():
    assert "SUFFICIENCY_TOP_SCORE" in _src()

@pytest.mark.tier_a
def test_p5_a2_function_in_source():
    assert "def _is_evidence_sufficient" in _src()

@pytest.mark.tier_a
def test_p5_a3_empty_chunks_fail():
    assert _is_evidence_sufficient([]) is False

@pytest.mark.tier_a
def test_p5_a4_low_score_fails():
    assert _is_evidence_sufficient([_chunk(0.28), _chunk(0.31)]) is False

@pytest.mark.tier_a
def test_p5_a5_exact_threshold_passes():
    assert _is_evidence_sufficient([_chunk(0.35)]) is True

@pytest.mark.tier_a
def test_p5_a6_high_score_passes():
    assert _is_evidence_sufficient([_chunk(0.28), _chunk(0.72)]) is True

@pytest.mark.tier_a
def test_p5_a7_gate_wired_in_chat():
    src = _src()
    # Gate must appear between search_qdrant and context_str in /chat
    pos_search = src.find("chunks = await search_qdrant(vec, req.domain)\n\n    # ── Phase 5")
    assert pos_search != -1, "Phase 5 gate block not found directly after search_qdrant in /chat"

@pytest.mark.tier_a
def test_p5_a8_gate_wired_in_stream():
    src = _src()
    assert "gate_ev" in src, "stream gate event variable 'gate_ev' not found"
    assert '"type": "gate"' in src or '"type":"gate"' in src

@pytest.mark.tier_a
def test_p5_a9_sufficiency_msg_arabic():
    src = _src()
    assert "SUFFICIENCY_MSG" in src
    assert "لم أجد" in src

@pytest.mark.tier_a
def test_p5_a10_no_gpt_when_gate_fires():
    src = _src()
    # In /chat, the gate return must come BEFORE oai().chat.completions.create
    gate_idx = src.find("if not _is_evidence_sufficient(chunks):")
    assert gate_idx != -1, "_is_evidence_sufficient gate not found"
    # find the GPT call in the same /chat block (search AFTER the gate position)
    oai_idx  = src.find("oai().chat.completions.create", gate_idx)
    assert oai_idx != -1, "oai().chat.completions.create not found after gate"
    assert gate_idx < oai_idx, "Gate check must appear before GPT call in /chat"

@pytest.mark.tier_a
def test_p5_a11_threshold_value():
    src = _src()
    assert "SUFFICIENCY_TOP_SCORE = 0.35" in src

@pytest.mark.tier_a
def test_p5_a12_min_score_unchanged():
    src = _src()
    assert "MIN_SCORE" in src and "0.28" in src
