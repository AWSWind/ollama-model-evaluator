#!/usr/bin/env bash
# Debug helper: run scripts/start.sh --background and report state.
# Uploaded to /tmp/ on the remote and executed from there.
set -u
cd /home/azurewind/workspaces/AI-Model-Evaluation

pkill -f 'ollama_evaluator.cli.*serve' 2>/dev/null || true
sleep 1
rm -rf .run logs

./scripts/start.sh --background --port 8765
rc=$?
echo "START_EXIT=$rc"
echo "--- .run ---"
ls -la .run 2>&1 || true
echo "--- logs ---"
ls -la logs 2>&1 || true
echo "--- listening ---"
ss -tlnp 2>/dev/null | grep 8765 || echo "nothing on 8765"
echo "--- backend log tail ---"
if [ -f logs/backend.log ]; then tail -10 logs/backend.log; else echo "no log"; fi
