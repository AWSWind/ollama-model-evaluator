#!/bin/bash
# Poll a single run by ID until it reaches a terminal state.
# Usage: bash scripts/poll_run.sh <run_id>
RUN_ID="${1:?usage: poll_run.sh <run_id>}"

for i in $(seq 1 360); do
  OUT=$(curl -sS "http://127.0.0.1:8765/api/runs/$RUN_ID" 2>/dev/null)
  STATE=$(printf '%s' "$OUT" | python3 -c '
import sys,json
try:
  d=json.load(sys.stdin)
  from collections import Counter
  c = Counter(r["status"] for r in d.get("results", []))
  print(f"{d.get(\"status\",\"?\")}|{len(d.get(\"results\",[]))}|{dict(c)}")
except Exception:
  print("pending|0|{}")
' 2>/dev/null)
  IFS='|' read -r status done_cnt per <<< "$STATE"
  printf "poll %3d  status=%-12s done=%s  per=%s\n" "$i" "$status" "$done_cnt" "$per"
  case "$status" in
    completed|failed|error|aborted) break ;;
  esac
  sleep 10
done

echo
echo "=== Final summary ==="
curl -sS "http://127.0.0.1:8765/api/runs/$RUN_ID" | python3 -c '
import sys,json
d=json.load(sys.stdin)
print("status:", d.get("status"))
print("run_id:", d.get("run_id"))
print("ended_at:", d.get("ended_at"))
from collections import defaultdict
per_suite = defaultdict(lambda: {"pass":0,"fail":0,"error":0,"timeout":0})
for r in d.get("results", []):
  per_suite[r["suite"]][r["status"]] += 1
print("per-suite:")
for s, c in per_suite.items():
  print(f"  {s}: {c}")
print("aggregates:")
for a in d.get("aggregates", []):
  print(f"  {a[\"model\"]}: passed={a[\"passed\"]} failed={a[\"failed\"]} mean_tps={a[\"mean_tokens_per_second\"]:.2f}")
'
