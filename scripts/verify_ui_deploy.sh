#!/bin/bash
echo "== / =="
curl -sI http://127.0.0.1:8765/ | head -3
echo
echo "== /runs/new =="
curl -sS http://127.0.0.1:8765/runs/new | grep -oE '/assets/[^"]+'
echo
echo "== asset head =="
BUNDLE=$(curl -sS http://127.0.0.1:8765/runs/new | grep -oE '/assets/[^"]+' | head -1)
curl -sI "http://127.0.0.1:8765$BUNDLE" | head -5
echo
echo "== api/suites =="
curl -sS http://127.0.0.1:8765/api/suites | python3 -m json.tool
