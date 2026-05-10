#!/bin/bash
# Submit a tiny smoke run against a few of the new suites on .224.
# Arguments:
#   $1 — comma-separated model list (default: qwen3.6:27b)
#   $2 — comma-separated suite list
set -e

MODELS="${1:-qwen3.6:27b}"
SUITES="${2:-factual-qa}"

PAYLOAD=$(cat <<JSON
{
  "models": $(printf '%s' "$MODELS" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip().split(",")))'),
  "suites": $(printf '%s' "$SUITES" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip().split(",")))'),
  "repetitions": 1,
  "concurrency": 1,
  "tag_filter": []
}
JSON
)

echo "=== Submitting ==="
echo "payload: $PAYLOAD"
RUN=$(curl -sS -X POST http://127.0.0.1:8765/api/runs -H 'Content-Type: application/json' -d "$PAYLOAD")
RUN_ID=$(printf '%s' "$RUN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
echo "run_id: $RUN_ID"

echo
echo "=== Polling ==="
for i in $(seq 1 180); do
  STATE=$(curl -sS "http://127.0.0.1:8765/api/runs/$RUN_ID" 2>/dev/null | python3 -c '
import sys,json
try:
  d=json.load(sys.stdin)
  print(f"{d.get(\"status\",\"?\")}|{len(d.get(\"results\",[]))}|{d.get(\"ended_at\",\"\")}")
except Exception as e:
  print(f"pending|0|")
' 2>/dev/null)
  IFS='|' read -r status done_cnt ended <<< "$STATE"
  printf "poll %3d  status=%-12s done=%s\n" "$i" "$status" "$done_cnt"
  case "$status" in
    completed|failed|error|aborted) break ;;
  esac
  sleep 10
done

echo
echo "=== Aggregates ==="
curl -sS "http://127.0.0.1:8765/api/runs/$RUN_ID" | python3 -c '
import sys,json
d=json.load(sys.stdin)
print("status:", d["status"])
print("suites run:", sorted({r["suite"] for r in d["results"]}))
print("per-suite:")
from collections import Counter
by_suite = {}
for r in d["results"]:
  by_suite.setdefault(r["suite"], {"pass":0,"fail":0,"error":0,"timeout":0})
  by_suite[r["suite"]][r["status"]] = by_suite[r["suite"]].get(r["status"],0) + 1
for s, counts in by_suite.items():
  print(f"  {s}: {counts}")
print("aggregates:")
for a in d.get("aggregates", []):
  print(f"  model={a[\"model\"]} passed={a[\"passed\"]} failed={a[\"failed\"]} mean_tps={a[\"mean_tokens_per_second\"]}")
'
