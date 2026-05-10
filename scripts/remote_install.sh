#!/usr/bin/env bash
# Remote install + build for the Ollama Model Evaluator on 192.168.1.224.
# Executed by kiro after rsyncing the project to
# /home/azurewind/workspaces/AI-Model-Evaluation.
set -euo pipefail

PROJECT_ROOT="/home/azurewind/workspaces/AI-Model-Evaluation"
cd "$PROJECT_ROOT"

echo "=== [1/4] Python venv ==="
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip --quiet

echo "=== [2/4] Backend Python deps (editable + dev extras) ==="
pip install --quiet -e "backend[dev]"
python -c "import fastapi, pydantic, hypothesis, httpx; print('backend deps ok:', fastapi.__version__, pydantic.VERSION, hypothesis.__version__, httpx.__version__)"

echo "=== [3/4] UI deps ==="
cd "$PROJECT_ROOT/ui"
if [ ! -d node_modules ]; then
  npm ci --silent --no-audit --no-fund || npm install --silent --no-audit --no-fund
else
  echo "ui/node_modules already present; skipping npm install"
fi
node --version
npm --version

echo "=== [4/4] Build UI dist ==="
npm run build

echo "=== DONE ==="
