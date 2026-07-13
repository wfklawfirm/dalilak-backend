"""
Phase 9 -- Observability + Privacy -- Tier A (offline, source-inspection) tests
10 tests, zero network calls, zero env vars required.
"""
import pathlib
import re

BACKEND = pathlib.Path(__file__).parent.parent.parent
SRC_PATH = BACKEND / "main.py"


def _src() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


def test_p9_a01_logging_import():
    src = _src()
    assert "import logging" in src, "logging module not imported"
    assert "from contextvars import ContextVar" in src, "ContextVar not imported"


def test_p9_a02_log_instance_created():
    src = _src()
    assert '_log = logging.getLogger' in src, "_log logger not created"
    assert "dalilak" in src, "logger name 'dalilak' not found"


def test_p9_a03_req_id_contextvar_exists():
    src = _src()
    assert "_req_id_var" in src, "_req_id_var ContextVar not found"
    assert "ContextVar" in src, "ContextVar type not used"


def test_p9_a04_request_id_middleware_exists():
    src = _src()
    assert "_request_id_middleware" in src, "_request_id_middleware not found"
    assert "@app.middleware" in src, "@app.middleware decorator not found"


def test_p9_a05_middleware_sets_x_request_id_header():
    src = _src()
    mw_idx = src.find("_request_id_middleware")
    assert mw_idx >= 0
    snippet = src[mw_idx: mw_idx + 600]
    assert "X-Request-ID" in snippet, "X-Request-ID header not set in middleware"
    assert "response.headers" in snippet, "response.headers not modified"


def test_p9_a06_middleware_does_not_log_query_content():
    src = _src()
    mw_idx = src.find("_request_id_middleware")
    assert mw_idx >= 0
    # Find the function body (up to next @app or def at col 0)
    next_fn = src.find("\n@app.", mw_idx + 10)
    snippet = src[mw_idx: next_fn if next_fn > 0 else mw_idx + 800]
    # Must NOT log req.query, req.message, payload content
    for forbidden in ("req.query", "req.message", "payload", ".body"):
        assert forbidden not in snippet, (
            f"middleware logs forbidden field: {forbidden}"
        )


def test_p9_a07_global_exception_handler_exists():
    src = _src()
    assert "_global_exception_handler" in src, "global exception handler not found"
    assert "exception_handler" in src, "@app.exception_handler decorator missing"


def test_p9_a08_exception_handler_returns_opaque_message():
    src = _src()
    exc_idx = src.find("_global_exception_handler")
    assert exc_idx >= 0
    snippet = src[exc_idx: exc_idx + 500]
    # Must return JSONResponse, not re-raise or expose traceback
    assert "JSONResponse" in snippet, "exception handler must return JSONResponse"
    # Must include req_id for correlation
    assert "req_id" in snippet, "exception handler must include req_id in response"


def test_p9_a09_exception_handler_logs_internally():
    src = _src()
    exc_idx = src.find("_global_exception_handler")
    assert exc_idx >= 0
    snippet = src[exc_idx: exc_idx + 500]
    # Must log the exception (not just silently swallow it)
    assert "_log." in snippet, "exception handler must log the error"


def test_p9_a10_health_endpoint_no_exception_detail_leak():
    src = _src()
    health_idx = src.find("@app.get(\"/health\")")
    assert health_idx >= 0, "/health endpoint not found"
    snippet = src[health_idx: health_idx + 600]
    # Must NOT use str(e) in HTTPException detail
    assert "detail=str(e)" not in snippet, (
        "/health leaks exception details via str(e) — use opaque message"
    )
    # Should log the exception instead
    assert "_log." in snippet, "/health should log the exception"
