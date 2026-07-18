# -*- coding: utf-8 -*-
"""
Phase 16 — Procedures Catalog API — Tier A (offline, TestClient)

8 procedures total (4 original + 4 new from Phase 16 task).
Tests verify: listing, filtering, search, detail, 404, and categories.
No auth required for any of these endpoints.
No network calls — TestClient intercepts all HTTP.
"""
from __future__ import annotations

import os
import sys

# Set env vars BEFORE importing main so module-level constants are correct.
os.environ.setdefault("JWT_SECRET", "test-secret-32-chars-minimum-ok!")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.pop("ADMIN_USERNAME", None)  # prevent startup from calling Qdrant

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import main  # noqa: E402 — must come after sys.path manipulation
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app, raise_server_exceptions=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(path: str, **params) -> "httpx.Response":  # type: ignore[name-defined]
    return client.get(path, params=params)


# ── T16-01: list returns 8+ procedures ───────────────────────────────────────

def test_p16_01_list_returns_all_procedures():
    r = _get("/procedures")
    assert r.status_code == 200
    data = r.json()
    assert "procedures" in data
    assert "total" in data
    assert data["total"] >= 8, f"Expected >= 8 procedures, got {data['total']}"
    assert len(data["procedures"]) >= 8


# ── T16-02: category filter ───────────────────────────────────────────────────

def test_p16_02_filter_by_category_vehicles():
    r = _get("/procedures", category="vehicles")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 2, "Expected at least 2 vehicles procedures"
    for p in data["procedures"]:
        assert p["category"] == "vehicles", f"Non-vehicles procedure returned: {p['id']}"


# ── T16-03: text search for passport ─────────────────────────────────────────

def test_p16_03_search_returns_passport_for_jawaz():
    r = _get("/procedures", q="جواز")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1, "Search for 'جواز' returned no results"
    ids = [p["id"] for p in data["procedures"]]
    assert "passport-renewal-lb" in ids, f"passport-renewal-lb not found in: {ids}"


# ── T16-04: procedure detail ──────────────────────────────────────────────────

def test_p16_04_get_passport_detail():
    r = _get("/procedures/passport-renewal-lb")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "passport-renewal-lb"
    assert "steps" in data
    assert len(data["steps"]) >= 1
    assert "documents" in data
    assert "authority" in data


# ── T16-05: non-existent slug returns 404 ────────────────────────────────────

def test_p16_05_nonexistent_slug_returns_404():
    r = _get("/procedures/nonexistent-slug-xyz")
    assert r.status_code == 404


# ── T16-06: categories list returns multiple categories ──────────────────────

def test_p16_06_categories_list():
    r = _get("/procedures/categories/list")
    assert r.status_code == 200
    data = r.json()
    assert "categories" in data
    cats = data["categories"]
    assert len(cats) >= 3, f"Expected >= 3 categories, got {len(cats)}: {cats}"
    # Our procedures span: civil_status, vehicles, business, real_estate, education
    assert "vehicles" in cats
    assert "civil_status" in cats


# ── T16-07: all 8 catalog IDs are present in list ────────────────────────────

def test_p16_07_all_eight_procedure_ids_present():
    expected_ids = {
        "passport-renewal-lb",
        "birth-registration-lb",
        "vehicle-registration-lb",
        "company-registration-lb",
        "driving-license-lb",
        "marriage-certificate-lb",
        "land-registration-lb",
        "work-permit-lb",
    }
    r = _get("/procedures", limit=20)
    assert r.status_code == 200
    returned_ids = {p["id"] for p in r.json()["procedures"]}
    missing = expected_ids - returned_ids
    assert not missing, f"Missing procedure IDs: {missing}"


# ── T16-08: new procedures have correct metadata ─────────────────────────────

def test_p16_08_new_procedures_have_correct_fields():
    new_slugs = [
        "driving-license-lb",
        "marriage-certificate-lb",
        "land-registration-lb",
        "work-permit-lb",
    ]
    for slug in new_slugs:
        r = _get(f"/procedures/{slug}")
        assert r.status_code == 200, f"Expected 200 for {slug}, got {r.status_code}"
        data = r.json()
        assert data.get("status") == "verified", f"{slug} status != verified"
        assert len(data.get("steps", [])) >= 4, f"{slug} has fewer than 4 steps"
        assert len(data.get("documents", [])) >= 3, f"{slug} has fewer than 3 documents"
        assert data.get("country") == "lebanon", f"{slug} country != lebanon"
