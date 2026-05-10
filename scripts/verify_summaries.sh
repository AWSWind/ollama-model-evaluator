#!/bin/bash
echo "=== /api/suites/summaries ==="
curl -sS http://127.0.0.1:8765/api/suites/summaries | python3 -m json.tool

echo
echo "=== timing (single request) ==="
curl -sS -o /tmp/summaries.json -w "time_total=%{time_total}s  size=%{size_download}B\n" http://127.0.0.1:8765/api/suites/summaries

echo
echo "=== /runs/new bundle ref ==="
curl -sS http://127.0.0.1:8765/runs/new | grep -oE '/assets/[^"]+'
