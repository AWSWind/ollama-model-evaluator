#!/bin/bash
# Fetch a single run's progress snapshot.
# Usage: bash scripts/check_run.sh <run_id>
RUN_ID="${1:?usage: check_run.sh <run_id>}"

curl -sS "http://127.0.0.1:8765/api/runs/$RUN_ID" > /tmp/_run.json 2>/dev/null
python3 <<'PY'
import json
from collections import Counter
with open("/tmp/_run.json") as f:
    d = json.load(f)
status = d.get("status")
results = d.get("results", [])
c = Counter(r["status"] for r in results)
print(f"status     : {status}")
print(f"# results  : {len(results)}")
print(f"per-status : {dict(c)}")
print(f"started_at : {d.get('started_at')}")
print(f"ended_at   : {d.get('ended_at')}")
PY
