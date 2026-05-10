#!/usr/bin/env bash
# Ollama Model Evaluator — one-button local install.
#
# Runs on Linux or macOS. Invoke from the repo root:
#
#     ./scripts/install.sh
#
# Flags:
#     --skip-tests        Skip the post-install smoke test (faster).
#     --skip-ui           Skip the UI build (Python backend only).
#     --no-venv           Install into the system Python instead of ./.venv.
#     --python PATH       Use a specific python3 binary.
#     --verbose           Print every command before running it.
#     -h, --help          Show this help and exit.
#
# What this script does:
#   1. Checks prerequisites (python3 >= 3.11, node >= 18, npm, git).
#   2. Creates a Python virtual environment under ./.venv (unless --no-venv).
#   3. Installs the backend in editable mode with dev extras.
#   4. Installs UI dependencies via npm and produces ui/dist/.
#   5. Regenerates the shared OpenAPI/JSON Schema artefacts.
#   6. Runs a short smoke test (`ollama-evaluator --help` and
#      `pytest tests/unit -q -x` — unless --skip-tests).
#   7. Prints next-step instructions.
#
# Idempotent: safe to run again to update an existing install. Re-running
# after a `git pull` picks up new Python + UI dependencies automatically.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ANSI colour helpers. Disabled automatically when stdout is not a TTY.
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN=''
fi

msg()     { printf '%s>>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
section() { printf '\n%s==%s %s%s%s\n' "$C_BLUE" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"; }
ok()      { printf '%sOK%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()    { printf '%s!!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()     { printf '%s**%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }
die()     { err "$*"; exit 1; }

usage() {
  # Dump the comment block above -- the canonical help lives there so we never
  # drift between the top-of-file docs and ``--help``.
  sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SKIP_TESTS=false
SKIP_UI=false
USE_VENV=true
PYTHON_BIN=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests) SKIP_TESTS=true ;;
    --skip-ui)    SKIP_UI=true ;;
    --no-venv)    USE_VENV=false ;;
    --python)     PYTHON_BIN="${2:-}"; shift ;;
    --verbose|-v) VERBOSE=true ;;
    -h|--help)    usage ;;
    *)            die "Unknown option: $1 (use --help to see valid options)" ;;
  esac
  shift
done

if [[ "$VERBOSE" = true ]]; then
  set -x
fi

# ---------------------------------------------------------------------------
# 1. Prerequisite checks
# ---------------------------------------------------------------------------

section "Checking prerequisites"

# Python 3.11+.
if [[ -n "$PYTHON_BIN" ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "Python binary not found at: $PYTHON_BIN"
else
  # Prefer ``python3`` when present; fall back to ``python`` for minimal Linux
  # distros that only ship the unversioned command.
  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

[[ -n "$PYTHON_BIN" ]] || die "Python 3.11+ is required. Install from https://www.python.org/downloads/"

# Grab the version in a format we can compare numerically.
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"
if (( PY_MAJOR < 3 )) || { (( PY_MAJOR == 3 )) && (( PY_MINOR < 11 )); }; then
  die "Python 3.11+ required; found $PY_VERSION at $(command -v "$PYTHON_BIN")"
fi
ok "Python $PY_VERSION at $(command -v "$PYTHON_BIN")"

# Node + npm (only when building the UI).
if [[ "$SKIP_UI" = false ]]; then
  command -v node >/dev/null 2>&1 || \
    die "Node.js is required to build the UI. Install v18+ from https://nodejs.org/ (or re-run with --skip-ui)."
  command -v npm  >/dev/null 2>&1 || \
    die "npm is required to build the UI. Usually ships with Node.js (or re-run with --skip-ui)."
  NODE_VERSION="$(node --version | sed 's/^v//')"
  NODE_MAJOR="${NODE_VERSION%%.*}"
  if (( NODE_MAJOR < 18 )); then
    die "Node.js 18+ required; found v$NODE_VERSION"
  fi
  ok "Node v$NODE_VERSION"
  ok "npm $(npm --version)"
fi

# ``git`` is not strictly required, but we nag if missing so users hit the
# "clone the repo" guide before wasting time debugging a missing ``.git/``.
if ! command -v git >/dev/null 2>&1; then
  warn "git not found; this is fine if you downloaded the repo as a tarball."
fi

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------

section "Setting up Python environment"

cd "$REPO_ROOT"

if [[ "$USE_VENV" = true ]]; then
  VENV_DIR="$REPO_ROOT/.venv"
  if [[ -d "$VENV_DIR" ]]; then
    ok "Using existing virtual environment at .venv/"
  else
    msg "Creating virtual environment at .venv/"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # Re-point ``PYTHON_BIN`` at the venv interpreter for the rest of the run.
  # Using the venv's ``python`` directly is the most portable way to invoke
  # it without having to ``source`` activation scripts (which are shell-
  # specific and do not propagate cleanly to ``set -e``).
  PYTHON_BIN="$VENV_DIR/bin/python"
  [[ -x "$PYTHON_BIN" ]] || die "Virtual environment creation failed: $PYTHON_BIN missing"
  ok "Virtual environment ready"
else
  warn "Using system Python (--no-venv). This is not recommended outside of CI containers."
fi

msg "Upgrading pip inside the environment"
"$PYTHON_BIN" -m pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Backend install
# ---------------------------------------------------------------------------

section "Installing backend (editable + dev extras)"

# Editable install so ``pip install`` after a ``git pull`` picks up new
# files without rebuilding wheels. ``[dev]`` pulls pytest, hypothesis,
# ruff, mypy, and pytest-cov.
"$PYTHON_BIN" -m pip install --quiet -e "$REPO_ROOT/backend[dev]"
ok "Backend installed"

# ---------------------------------------------------------------------------
# 4. UI install + build
# ---------------------------------------------------------------------------

if [[ "$SKIP_UI" = true ]]; then
  warn "Skipping UI build (--skip-ui). The web UI will not be reachable via ``cli serve``."
else
  section "Installing UI dependencies"
  cd "$REPO_ROOT/ui"
  # ``npm ci`` is faster and deterministic when ``package-lock.json`` is
  # present; fall back to ``npm install`` if the lockfile is missing (a
  # fresh tarball drop might not include it).
  if [[ -f package-lock.json ]]; then
    npm ci --silent --no-audit --no-fund
  else
    npm install --silent --no-audit --no-fund
  fi
  ok "UI dependencies installed ($(ls node_modules | wc -l) packages)"

  section "Building UI bundle"
  npm run build
  [[ -f "$REPO_ROOT/ui/dist/index.html" ]] \
    || die "UI build completed without emitting ui/dist/index.html"
  ok "UI bundle ready at ui/dist/"
  cd "$REPO_ROOT"
fi

# ---------------------------------------------------------------------------
# 5. Regenerate shared schemas
# ---------------------------------------------------------------------------

section "Regenerating shared schemas"

"$PYTHON_BIN" "$REPO_ROOT/backend/scripts/regen_schemas.py" >/dev/null
ok "shared/openapi.yaml, shared/evaluation-suite.schema.json, shared/run-report.schema.json refreshed"

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

if [[ "$SKIP_TESTS" = true ]]; then
  warn "Skipping smoke tests (--skip-tests)."
else
  section "Running smoke tests"

  msg "CLI help check"
  "$PYTHON_BIN" -m ollama_evaluator.cli --help >/dev/null
  ok "CLI imports and loads"

  msg "Unit tests"
  # ``-x`` fails fast; ``tests/unit`` is the fastest subset and covers all
  # scaffolding. Property + integration tests live in other folders and
  # are slower; full ``pytest`` is what you run before shipping.
  ( cd "$REPO_ROOT/backend" && "$PYTHON_BIN" -m pytest tests/unit -q -x )
  ok "Unit tests passed"
fi

# ---------------------------------------------------------------------------
# 7. Next steps
# ---------------------------------------------------------------------------

cat <<NEXT

${C_GREEN}${C_BOLD}Install complete.${C_RESET}

${C_BOLD}Activate the environment for every new shell session:${C_RESET}
  cd $REPO_ROOT
  source .venv/bin/activate

${C_BOLD}Next steps:${C_RESET}
  • Pull a small Ollama model:    ${C_DIM}ollama pull llama3:8b${C_RESET}
  • Edit ${C_DIM}examples/config.qwen.yaml${C_RESET} and point ${C_DIM}models:${C_RESET} at your model.
  • Validate the example suite:   ${C_DIM}python -m ollama_evaluator.cli validate-suite examples/suites/reasoning-basics.yaml${C_RESET}
  • List Ollama models:           ${C_DIM}python -m ollama_evaluator.cli list-models${C_RESET}
  • Run an evaluation:            ${C_DIM}python -m ollama_evaluator.cli --config examples/config.qwen.yaml run${C_RESET}
  • Start the web UI + API:       ${C_DIM}OLLAMA_EVAL_UI_DIR=\$PWD/ui/dist python -m ollama_evaluator.cli --config examples/config.qwen.yaml serve${C_RESET}

  User manual: ${C_DIM}docs/USER_MANUAL.md${C_RESET}

NEXT
