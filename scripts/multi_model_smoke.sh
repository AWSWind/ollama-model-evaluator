#!/bin/bash
set -e
# Submit a multi-model run so we can verify the per-model breakdown.
PAYLOAD='{"models": ["qwen3.6:27b", "qwen3.5:35b-a3b"], "suites": ["reasoning-basics"], "repetitions": 1, "concurrency": 1, "tag_filter": []}'
echo "=== Submitting ==="
echo "payload: $PAYLOAD"
RUN=$(curl -sS -X POST http://127.0.0.1:8765/api/runs -H 'Content-Type: application/json' -d "$PAYLOAD")
RUN_ID=$(printf '%s' "$RUN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
echo "run_id: $RUN_ID"
echo "$RUN_ID" > /tmp/last_multi_run.id
