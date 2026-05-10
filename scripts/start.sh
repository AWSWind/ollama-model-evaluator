#!/usr/bin/env bash
# Ollama Model Evaluator - one-button project launcher.
#
# Brings up the full stack:
#   1. Ollama server (if not already running and the CLI is installed).
#   2. The backend (FastAPI on port 8765 by default).
#   3. The React UI bundle mounted by the backend at ``/``.
#
# Usage from the repo root:
#
#     ./scripts/start.sh                    # production mode, foreground
#     ./scripts/start.sh --background       # detach after services are ready
#     ./scripts/start.sh --port 9000        # use a non-default backend port
#     ./scripts/start.sh --dev              # also run Vite dev server on :5173
#     ./scripts/start.sh --skip-ollama      # skip Ollama liveness check / start
#     ./scripts/start.sh --no-install       # never run install.sh, even if deps missing
#     ./scripts/start.sh --host 127.0.0.1   # bind backend to localhost only
#     ./scripts/start.sh --config PATH      # config file for the backend
#
# Default behaviour (no flags):
#   * Ensures ``.venv/`` and ``ui/dist/`` exist, running ``install.sh --skip-tests``
#     when either is missing.
#   * Verifies Ollama is reachable at ``http://localhost:11434``; starts it in
#     the background if the CLI is installed but not running.
#   * Starts the backend in the background and tails its log until Ctrl-C.
#   * Ctrl-C stops the backend (and, if started by this script, Ollama) and
#     tears down the Vite dev server in --dev mode.
#
# PID files land under ``.run/``; logs under ``logs/``. Both directories are
# created on first start and reused on subsequent runs.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ANSI colours only when stdout is a TTY. Plain ASCII prefixes so the output
# stays readable even when colour support is missing (typical cron, CI, and
# PowerShell-wrapped scenarios).
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''; C_YELLOW=''
  C_BLUE=''; C_CYAN=''
fi

msg()     { printf '%s>>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
section() { printf '\n%s==%s %s%s%s\n' "$C_BLUE" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"; }
ok()      { printf '%sOK%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()    { printf '%s!!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()     { printf '%s**%s %s\n' "$C_RED"    "$C_RESET" "$*" >&2; }
die()     { err "$*"; exit 1; }

usage() {
  # The canonical help is the comment block at the top of this file; dumping
  # it here means we never drift between ``--help`` and the actual docs.
  sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

CONFIG="examples/config.qwen.yaml"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8765"
OLLAMA_PORT="11434"
UI_DEV_PORT="5173"

RUN_DEV=false          # If true, also start the Vite dev server.
DETACH=false           # If true, return as soon as services are ready.
SKIP_OLLAMA=false      # If true, do not check/start the Ollama server.
AUTO_INSTALL=true      # If true and deps missing, automatically run install.sh.
LOG_LEVEL="info"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="${2:?}"; shift ;;
    --host)        BACKEND_HOST="${2:?}"; shift ;;
    --port)        BACKEND_PORT="${2:?}"; shift ;;
    --ollama-port) OLLAMA_PORT="${2:?}"; shift ;;
    --dev)         RUN_DEV=true ;;
    --background|--detach) DETACH=true ;;
    --foreground)  DETACH=false ;;
    --skip-ollama) SKIP_OLLAMA=true ;;
    --no-install)  AUTO_INSTALL=false ;;
    --log-level)   LOG_LEVEL="${2:?}"; shift ;;
    -h|--help)     usage ;;
    *)             die "Unknown option: $1 (use --help to see valid options)" ;;
  esac
  shift
done

RUN_DIR="$REPO_ROOT/.run"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

BACKEND_PID_FILE="$RUN_DIR/backend.pid"
OLLAMA_PID_FILE="$RUN_DIR/ollama.pid"     # Only written if we started it.
VITE_PID_FILE="$RUN_DIR/vite.pid"

BACKEND_LOG="$LOG_DIR/backend.log"
OLLAMA_LOG="$LOG_DIR/ollama.log"
VITE_LOG="$LOG_DIR/vite.log"

# ---------------------------------------------------------------------------
# Process-management helpers
# ---------------------------------------------------------------------------

# ``pid_alive`` returns 0 when the PID in the given file refers to a running
# process (``kill -0`` is a no-op permission check that fails on dead PIDs).
# Handles the "empty file" case cleanly.
pid_alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

# ``port_in_use`` returns 0 when something is listening on the given TCP port
# on 127.0.0.1. Uses ``ss`` when available (fastest), ``lsof`` as a fallback,
# and a raw ``/dev/tcp`` probe as a last resort for minimal containers.
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | awk '{print $4}' | grep -E "(^|:)${port}$" -q
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1
  else
    # ``/dev/tcp`` is bash-specific and does a real TCP connect. Suppress
    # its error output so this never produces noise when the port is free.
    ( exec 3<>/dev/tcp/127.0.0.1/"$port" ) 2>/dev/null
  fi
}

# ``wait_for_http`` polls a URL until it returns a 2xx or 3xx, or the deadline
# passes. Used to decide when a service is actually ready to accept traffic.
wait_for_http() {
  local url="$1" timeout_s="${2:-30}" label="${3:-service}"
  local deadline=$(( $(date +%s) + timeout_s ))
  while (( $(date +%s) < deadline )); do
    # ``--max-time`` caps the per-attempt wait so a hung service does not
    # blow past our overall timeout. ``-o /dev/null`` discards the body.
    if curl -fsSL --max-time 2 -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  err "$label did not become ready at $url within ${timeout_s}s"
  return 1
}

# ``kill_pidfile`` signals the pid in ``$1`` with ``$2`` (default TERM), waits
# up to five seconds for it to die, and KILLs if it does not. Always deletes
# the pidfile at the end so restart cycles don't leak state.
kill_pidfile() {
  local pidfile="$1" sig="${2:-TERM}"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -"$sig" "$pid" 2>/dev/null || true
    # Small grace period so the process can shut down cleanly.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    # Escalate to SIGKILL if the process is still up after the grace period.
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}

# ---------------------------------------------------------------------------
# 1. Dependency preflight
# ---------------------------------------------------------------------------

section "Preflight"

# The installer is the single source of truth for setup; the start script
# merely detects what's missing and delegates. This keeps the two scripts in
# lockstep without duplicating install logic.
missing=()
if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  missing+=("Python venv (.venv/)")
fi
if [[ ! -f "$REPO_ROOT/ui/dist/index.html" ]]; then
  missing+=("UI bundle (ui/dist/)")
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  warn "Missing dependencies: ${missing[*]}"
  if [[ "$AUTO_INSTALL" = true ]]; then
    msg "Running scripts/install.sh --skip-tests to set them up"
    bash "$SCRIPT_DIR/install.sh" --skip-tests
  else
    die "--no-install was given; run scripts/install.sh manually first"
  fi
fi
ok "Python venv ready"
ok "UI bundle ready"

PY="$REPO_ROOT/.venv/bin/python"

# ---------------------------------------------------------------------------
# 2. Ollama
# ---------------------------------------------------------------------------

# Track whether this script started Ollama so we only tear it down on exit in
# that case. A pre-existing ``ollama serve`` (daemon, systemd, etc.) belongs
# to whoever started it and must survive our shutdown.
STARTED_OLLAMA=false

if [[ "$SKIP_OLLAMA" = true ]]; then
  warn "Skipping Ollama check (--skip-ollama)"
else
  section "Checking Ollama"
  if curl -fsSL --max-time 2 "http://localhost:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
    ok "Ollama already running on :$OLLAMA_PORT"
  else
    if command -v ollama >/dev/null 2>&1; then
      msg "Ollama not reachable; starting 'ollama serve' in the background"
      if command -v setsid >/dev/null 2>&1; then
        setsid ollama serve </dev/null >"$OLLAMA_LOG" 2>&1 &
      else
        nohup ollama serve </dev/null >"$OLLAMA_LOG" 2>&1 &
        disown 2>/dev/null || true
      fi
      echo $! >"$OLLAMA_PID_FILE"
      STARTED_OLLAMA=true
      # Ollama's own startup takes a moment before the HTTP port is open.
      # ``wait_for_http`` polls with a 30s ceiling to avoid hanging forever
      # on a misconfigured GPU or a half-installed server.
      if ! wait_for_http "http://localhost:$OLLAMA_PORT/api/version" 30 "Ollama"; then
        err "Ollama failed to start. Last 20 lines of $OLLAMA_LOG:"
        tail -20 "$OLLAMA_LOG" || true
        die "Aborting"
      fi
      ok "Ollama up on :$OLLAMA_PORT (pid $(cat "$OLLAMA_PID_FILE"))"
    else
      err "Ollama is not running on :$OLLAMA_PORT and the 'ollama' CLI is not installed."
      err "Install from https://ollama.com/download, or pass --skip-ollama to continue without it."
      die "Aborting"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 3. Backend
# ---------------------------------------------------------------------------

section "Starting backend"

# Refuse to start a second backend on top of a live one. Doing so would
# silently hand the port to the existing process; the new launch would 'run'
# but the server the user hits is stale and the PID file we write is wrong.
if pid_alive "$BACKEND_PID_FILE"; then
  existing="$(cat "$BACKEND_PID_FILE")"
  die "Backend already running with PID $existing. Run scripts/stop.sh first."
fi
if port_in_use "$BACKEND_PORT"; then
  die "Port $BACKEND_PORT is already bound. Choose another with --port or free it first."
fi

[[ -f "$REPO_ROOT/$CONFIG" ]] || die "Config file not found: $CONFIG"

# ``OLLAMA_EVAL_UI_DIR`` tells the backend where to look for the compiled UI.
# Pointing it at our ``ui/dist/`` means one server, one URL for the user.
export OLLAMA_EVAL_UI_DIR="$REPO_ROOT/ui/dist"

# Spawn with ``setsid`` so the backend lives in its own session and
# process group. Plain ``nohup ... &`` is not enough on every distro:
# when this script is invoked through ``ssh ... "./start.sh"``, the
# ssh client sends SIGHUP to the whole foreground process group when
# the session closes, which reaps the nohup'd children. ``setsid``
# moves the new process into a fresh session so that HUP never
# reaches it. Falls back to plain ``nohup`` on systems without
# ``setsid`` (rare, but macOS pre-12 is the usual suspect).
if command -v setsid >/dev/null 2>&1; then
  setsid "$PY" -m ollama_evaluator.cli \
      --config "$CONFIG" --log-level "$LOG_LEVEL" \
      serve --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
      </dev/null >"$BACKEND_LOG" 2>&1 &
else
  nohup "$PY" -m ollama_evaluator.cli \
      --config "$CONFIG" --log-level "$LOG_LEVEL" \
      serve --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
      </dev/null >"$BACKEND_LOG" 2>&1 &
  disown 2>/dev/null || true
fi
echo $! >"$BACKEND_PID_FILE"

if ! wait_for_http "http://localhost:$BACKEND_PORT/api/health" 30 "Backend"; then
  err "Backend failed to start. Last 30 lines of $BACKEND_LOG:"
  tail -30 "$BACKEND_LOG" || true
  kill_pidfile "$BACKEND_PID_FILE"
  die "Aborting"
fi
ok "Backend up on http://$BACKEND_HOST:$BACKEND_PORT (pid $(cat "$BACKEND_PID_FILE"))"

# ---------------------------------------------------------------------------
# 4. UI dev server (optional, --dev)
# ---------------------------------------------------------------------------

if [[ "$RUN_DEV" = true ]]; then
  section "Starting Vite dev server"
  if pid_alive "$VITE_PID_FILE"; then
    die "Vite dev server already running with PID $(cat "$VITE_PID_FILE"). Run scripts/stop.sh first."
  fi
  if port_in_use "$UI_DEV_PORT"; then
    die "Port $UI_DEV_PORT is already bound. Set UI_DEV_PORT in scripts/start.sh or free the port."
  fi
  # The Vite dev server auto-reloads on file changes and proxies ``/api`` and
  # ``/openapi.json`` to the backend (see ui/vite.config.ts). Run it from
  # inside ``ui/`` so npm picks up the right package.json. ``setsid`` puts
  # it in its own session for the same reason as the backend.
  if command -v setsid >/dev/null 2>&1; then
    ( cd "$REPO_ROOT/ui" && setsid npm run dev -- --host 0.0.0.0 --port "$UI_DEV_PORT" </dev/null >"$VITE_LOG" 2>&1 ) &
  else
    ( cd "$REPO_ROOT/ui" && nohup npm run dev -- --host 0.0.0.0 --port "$UI_DEV_PORT" </dev/null >"$VITE_LOG" 2>&1 ) &
  fi
  echo $! >"$VITE_PID_FILE"
  if ! wait_for_http "http://localhost:$UI_DEV_PORT/" 30 "Vite"; then
    warn "Vite did not respond at :$UI_DEV_PORT within 30s; see $VITE_LOG"
  else
    ok "Vite dev server up on http://localhost:$UI_DEV_PORT (pid $(cat "$VITE_PID_FILE"))"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Ready banner
# ---------------------------------------------------------------------------

HOST_FOR_USER="$BACKEND_HOST"
[[ "$HOST_FOR_USER" = "0.0.0.0" ]] && HOST_FOR_USER="localhost"

cat <<READY

${C_GREEN}${C_BOLD}All services ready.${C_RESET}

  ${C_BOLD}Web UI${C_RESET}           http://${HOST_FOR_USER}:${BACKEND_PORT}/
  ${C_BOLD}Health probe${C_RESET}     http://${HOST_FOR_USER}:${BACKEND_PORT}/api/health
  ${C_BOLD}REST API docs${C_RESET}    http://${HOST_FOR_USER}:${BACKEND_PORT}/openapi.json
$(if [[ "$RUN_DEV" = true ]]; then printf '  %sVite dev server%s  http://localhost:%s/\n' "$C_BOLD" "$C_RESET" "$UI_DEV_PORT"; fi)
  ${C_BOLD}Backend log${C_RESET}      $BACKEND_LOG
$(if [[ "$STARTED_OLLAMA" = true ]]; then printf '  %sOllama log%s      %s\n' "$C_BOLD" "$C_RESET" "$OLLAMA_LOG"; fi)

  PID files in ${C_DIM}$RUN_DIR${C_RESET}

${C_BOLD}Stop everything:${C_RESET}
  ./scripts/stop.sh
  (or Ctrl-C if this script is in the foreground)

READY

# ---------------------------------------------------------------------------
# 6. Foreground vs. detached
# ---------------------------------------------------------------------------

if [[ "$DETACH" = true ]]; then
  ok "Detached. Services continue running in the background."
  exit 0
fi

# Signal handling: on Ctrl-C or normal termination, stop the services we
# control and only those we started ourselves. Ollama gets signalled only
# when ``STARTED_OLLAMA=true``; a pre-existing Ollama keeps running.
cleanup() {
  printf '\n'
  msg "Shutting down..."
  kill_pidfile "$BACKEND_PID_FILE"
  if [[ "$RUN_DEV" = true ]]; then
    kill_pidfile "$VITE_PID_FILE"
  fi
  if [[ "$STARTED_OLLAMA" = true ]]; then
    kill_pidfile "$OLLAMA_PID_FILE"
  fi
  ok "Done"
}
trap cleanup EXIT INT TERM

msg "Tailing $BACKEND_LOG — press Ctrl-C to stop"
# ``tail --pid`` would be nicer but isn't available everywhere; instead we
# tail and let the ``trap`` catch the user's Ctrl-C to run cleanup.
tail -F "$BACKEND_LOG"
