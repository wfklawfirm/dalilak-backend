#!/usr/bin/env python3
"""
Qdrant Source Metadata Enrichment Script
=========================================
STATUS: CODE-ONLY — DO NOT RUN ON PRODUCTION until:
  1. Qdrant snapshot backup taken and verified (see scripts/qdrant_backup.sh)
  2. Dry-run tested on a local/staging Qdrant instance
  3. Owner has reviewed the field mapping below

PURPOSE:
  Enrich existing dalilak_ai_v2 collection points with structured metadata fields:
    - chunk_id       : str  — unique chunk identifier
    - country        : str  — "lebanon" | "syria"
    - jurisdiction   : str  — e.g., "لبنان/عام" | "سوريا/دمشق"
    - source_tier    : int  — 1 (law) | 2 (official) | 3 (circular) | 4 (professional) | 5 (operational)
    - effective_date : str  — ISO date or ""
    - last_verified  : str  — ISO date
    - review_expiry  : str  — ISO date (last_verified + 1 year)

USAGE:
  # Step 1: set environment variables (never hard-code secrets)
  export QDRANT_URL="..."
  export QDRANT_API_KEY="..."
  
  # Step 2: dry run (read-only, prints proposed changes)
  python scripts/qdrant_enrich_metadata.py --dry-run --limit 10
  
  # Step 3: apply (only after backup + dry-run verification)
  python scripts/qdrant_enrich_metadata.py --limit 1000 --offset 0
"""

import argparse
import os
import sys
import json
from datetime import date, timedelta
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, SetPayload
except ImportError:
    print("ERROR: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)

COLLECTION_NAME = "dalilak_ai_v2"
DEFAULT_LAST_VERIFIED = "2024-01-01"
DEFAULT_REVIEW_WINDOW_DAYS = 365

# ─────────────────────────────────────────────────────────────────────────────
# Source tier inference from existing payload fields
# ─────────────────────────────────────────────────────────────────────────────
_TIER_1_KEYWORDS = ["قانون", "مرسوم اشتراعي", "دستور", "قانون رقم", "law no", "decree-law"]
_TIER_2_KEYWORDS = ["وزارة", "مديرية عامة", "وزير", "قرار وزاري", "منشور وزاري"]
_TIER_3_KEYWORDS = ["تعميم", "مذكرة", "circular", "notification"]
_TIER_4_KEYWORDS = ["نقابة", "هيئة", "مجلس", "bar association", "professional"]


def _infer_source_tier(payload: dict[str, Any]) -> int:
    """Heuristically determine source_tier from existing payload."""
    text = " ".join([
        str(payload.get("source", "")),
        str(payload.get("domain", "")),
        str(payload.get("title", "")),
        str(payload.get("text", ""))[:200],
    ]).lower()
    
    if any(kw in text for kw in _TIER_1_KEYWORDS):
        return 1
    if any(kw in text for kw in _TIER_2_KEYWORDS):
        return 2
    if any(kw in text for kw in _TIER_3_KEYWORDS):
        return 3
    if any(kw in text for kw in _TIER_4_KEYWORDS):
        return 4
    return 5  # default: operational


def _infer_country(payload: dict[str, Any]) -> str:
    """Infer country from existing domain/source fields."""
    domain = str(payload.get("domain", "")).lower()
    source = str(payload.get("source", "")).lower()
    text_sample = str(payload.get("text", ""))[:100].lower()
    combined = domain + source + text_sample
    
    sy_markers = ["سوري", "سورية", "دمشق", "حلب", "syria", "syrian"]
    if any(m in combined for m in sy_markers):
        return "syria"
    return "lebanon"  # default


def _infer_jurisdiction(payload: dict[str, Any], country: str) -> str:
    """Build jurisdiction string from country + domain hints."""
    domain = str(payload.get("domain", "")).strip()
    if country == "syria":
        return f"سوريا/{domain}" if domain else "سوريا/عام"
    return f"لبنان/{domain}" if domain else "لبنان/عام"


def _build_enrichment(point_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the new metadata fields to set on this point."""
    country = _infer_country(payload)
    tier = _infer_source_tier(payload)
    last_verified = DEFAULT_LAST_VERIFIED
    review_expiry = (
        date.fromisoformat(last_verified) + timedelta(days=DEFAULT_REVIEW_WINDOW_DAYS)
    ).isoformat()
    
    return {
        "chunk_id": str(point_id),
        "country": country,
        "jurisdiction": _infer_jurisdiction(payload, country),
        "source_tier": tier,
        "effective_date": payload.get("effective_date", ""),
        "last_verified": last_verified,
        "review_expiry": review_expiry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Qdrant point metadata")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    parser.add_argument("--limit", type=int, default=100, help="Number of points to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()
    
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    
    if not qdrant_url:
        print("ERROR: QDRANT_URL environment variable not set")
        sys.exit(1)
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None, timeout=30)
    
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Fetching {args.limit} points from {args.collection} (offset={args.offset})...")
    
    result = client.scroll(
        collection_name=args.collection,
        limit=args.limit,
        offset=args.offset,
        with_payload=True,
        with_vectors=False,
    )
    
    points, next_offset = result
    print(f"Retrieved {len(points)} points. Next offset: {next_offset}")
    
    already_enriched = 0
    to_enrich = []
    
    for point in points:
        if point.payload and "chunk_id" in point.payload:
            already_enriched += 1
            continue
        enrichment = _build_enrichment(point.id, point.payload or {})
        to_enrich.append((point.id, enrichment))
    
    print(f"Already enriched: {already_enriched} | To enrich: {len(to_enrich)}")
    
    if args.dry_run:
        print("\n[DRY-RUN] Sample of proposed changes:")
        for pid, fields in to_enrich[:5]:
            print(f"  Point {pid}: {json.dumps(fields, ensure_ascii=False, indent=4)}")
        print(f"\n[DRY-RUN] Would update {len(to_enrich)} points. Run without --dry-run to apply.")
        return
    
    # Apply in batches of 100
    BATCH = 100
    updated = 0
    for i in range(0, len(to_enrich), BATCH):
        batch = to_enrich[i:i + BATCH]
        ids = [b[0] for b in batch]
        # Collect all payload fields
        payloads = {}
        for pid, fields in batch:
            for k, v in fields.items():
                if k not in payloads:
                    payloads[k] = {}
        
        # Use set_payload per point (most compatible)
        for pid, fields in batch:
            client.set_payload(
                collection_name=args.collection,
                payload=fields,
                points=[pid],
            )
        updated += len(batch)
        print(f"Updated {updated}/{len(to_enrich)} points...")
    
    print(f"Done. {updated} points enriched.")
    if next_offset:
        print(f"Re-run with --offset {next_offset} to process the next batch.")


if __name__ == "__main__":
    main()
