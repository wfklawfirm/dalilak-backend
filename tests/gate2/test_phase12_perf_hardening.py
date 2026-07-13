"""
Phase 12 -- Performance Hardening -- Tier A (offline, source-inspection) tests
8 tests, zero network calls, zero env vars required.
"""
import pathlib
import re

BACKEND  = pathlib.Path(__file__).parent.parent.parent
SRC_PATH = BACKEND / "main.py"


def _src() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


# -- Timeout constants --------------------------------------------------------

def test_p12_a01_qdrant_timeout_constant_exists():
    src = _src()
    assert "QDRANT_TIMEOUT_SEC" in src, "QDRANT_TIMEOUT_SEC constant missing from main.py"


def test_p12_a02_openai_timeout_constant_exists():
    src = _src()
    assert "OPENAI_TIMEOUT_SEC" in src, "OPENAI_TIMEOUT_SEC constant missing from main.py"


def test_p12_a03_timeout_values_are_positive_integers():
    src = _src()
    m_qdrant = re.search(r"QDRANT_TIMEOUT_SEC\s*=\s*(\d+)", src)
    m_openai = re.search(r"OPENAI_TIMEOUT_SEC\s*=\s*(\d+)", src)
    assert m_qdrant, "QDRANT_TIMEOUT_SEC not assigned a numeric value"
    assert m_openai, "OPENAI_TIMEOUT_SEC not assigned a numeric value"
    assert int(m_qdrant.group(1)) > 0, "QDRANT_TIMEOUT_SEC must be positive"
    assert int(m_openai.group(1)) > 0, "OPENAI_TIMEOUT_SEC must be positive"


def test_p12_a04_qdrant_timeout_less_than_openai_timeout():
    src = _src()
    m_qdrant = re.search(r"QDRANT_TIMEOUT_SEC\s*=\s*(\d+)", src)
    m_openai = re.search(r"OPENAI_TIMEOUT_SEC\s*=\s*(\d+)", src)
    assert m_qdrant and m_openai
    assert int(m_qdrant.group(1)) < int(m_openai.group(1)), (
        "QDRANT_TIMEOUT_SEC should be < OPENAI_TIMEOUT_SEC"
    )


# -- Message size limit -------------------------------------------------------

def test_p12_a05_max_message_len_constant_exists():
    src = _src()
    assert "MAX_MESSAGE_LEN" in src, "MAX_MESSAGE_LEN constant missing from main.py"


def test_p12_a06_max_message_len_is_reasonable():
    src = _src()
    m = re.search(r"MAX_MESSAGE_LEN\s*=\s*(\d+)", src)
    assert m, "MAX_MESSAGE_LEN not assigned a numeric value"
    val = int(m.group(1))
    assert 1000 <= val <= 10000, (
        f"MAX_MESSAGE_LEN should be in [1000, 10000], got {val}"
    )


def test_p12_a07_chat_endpoint_checks_message_len():
    src = _src()
    chat_idx = src.find('@app.post("/chat")')
    assert chat_idx >= 0, "/chat endpoint not found"
    snippet = src[chat_idx: chat_idx + 600]
    assert "MAX_MESSAGE_LEN" in snippet, (
        "MAX_MESSAGE_LEN check not present in /chat endpoint"
    )


def test_p12_a08_chat_stream_checks_message_len():
    src = _src()
    stream_idx = src.find('@app.post("/chat/stream")')
    assert stream_idx >= 0, "/chat/stream endpoint not found"
    snippet = src[stream_idx: stream_idx + 600]
    assert "MAX_MESSAGE_LEN" in snippet, (
        "MAX_MESSAGE_LEN check not present in /chat/stream endpoint"
    )
