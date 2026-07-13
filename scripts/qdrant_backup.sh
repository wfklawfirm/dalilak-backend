#!/usr/bin/env bash
# Qdrant Snapshot Backup Script
# ================================
# Creates a snapshot of the dalilak_ai_v2 collection via Qdrant REST API.
# Run this BEFORE any bulk metadata operations.
#
# USAGE:
#   export QDRANT_URL="https://your-qdrant-host"
#   export QDRANT_API_KEY="your-api-key"   # omit if no auth
#   bash scripts/qdrant_backup.sh
#
# OUTPUT:
#   Snapshot name printed to stdout.
#   Snapshot stored on Qdrant server (retrieve via GET /collections/.../snapshots)

set -euo pipefail

COLLECTION="${1:-dalilak_ai_v2}"
QDRANT_URL="${QDRANT_URL:?QDRANT_URL not set}"

HEADERS=(-H "Content-Type: application/json")
if [[ -n "${QDRANT_API_KEY:-}" ]]; then
    HEADERS+=(-H "api-key: ${QDRANT_API_KEY}")
fi

echo "Creating snapshot for collection: $COLLECTION"
RESPONSE=$(curl -s -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots" "${HEADERS[@]}")
echo "Response: $RESPONSE"

SNAPSHOT_NAME=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('name','ERROR'))")
echo "Snapshot name: $SNAPSHOT_NAME"

if [[ "$SNAPSHOT_NAME" == "ERROR" || -z "$SNAPSHOT_NAME" ]]; then
    echo "ERROR: Snapshot creation failed. Check QDRANT_URL and API key."
    exit 1
fi

echo "SUCCESS: Snapshot '$SNAPSHOT_NAME' created."
echo "List all snapshots: GET ${QDRANT_URL}/collections/${COLLECTION}/snapshots"
