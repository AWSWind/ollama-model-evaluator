#!/bin/bash
echo "=== /api/suites ==="
curl -sS http://127.0.0.1:8765/api/suites

echo
echo
echo "=== /api/suites/truthfulqa-mc1 case count ==="
curl -sS http://127.0.0.1:8765/api/suites/truthfulqa-mc1 | python3 -c '
import sys,json
try:
    d = json.load(sys.stdin)
    print("ok — name:", d.get("name"), "cases:", len(d.get("test_cases", [])))
except Exception as e:
    print("error:", e)
'
