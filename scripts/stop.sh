#!/usr/bin/env bash
# Ollama Model Evaluator - stop every process started by scripts/start.sh.
#
# Reads PID files written to ``.run/`` by ``start.sh`` and signals each
# process cleanly, escalating to SIGKILL when a process does not shut down
# within ~5s. Safe to run multiple times; missing PID files are silently
# ignored.
#
# By default Ollama is stopped only when this repo started it. Pass
# ``--stop-ollama`` to stop Ollama unconditionally.
#
# Usage:
#     ./scripts/stop.sh                # stop backend + Vite; leave shared Ollama alone
#     ./scripts/stop.sh --stop-ollama  # also stop Ollama even if we did not start it
#     ./scripts/stop.sh --force        # go straight to SIGKILL (avoid grace period)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
else
  C_RESET=''; C_BOLD=''; C_GREEN=''; C_YELLOW=''; C_CYAN=''
fi

msg()  { printf '%s>>%s %s\n' "$C_CYAN"   "$C_RESET" "$*"; }
ok()   { printf '%sOK%s %s\n' "$C_GREEN"  "$C_RESET" "$*"; }
warn() { printf '%s!!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }

STOP_OLLAMA=false
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-ollama) STOP_OLLAMA=true ;;
    --force)       FORCE=true ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) warn "Ignoring unknown option: $1" ;;
  esac
  shift
done

RUN_DIR="$REPO_ROOT/.run"
PIDS=(
  "$RUN_DIR/backend.pid"
  "$RUN_DIR/vite.pid"
)
# Ollama PID file is only present when start.sh started the daemon itself.
# --stop-ollama opts into stopping it anyway; otherwise we only touch the
# file when it was written by us (no separate flag needed).
[[ -f "$RUN_DIR/ollama.pid" ]] && PIDS+=("$RUN_DIR/ollama.pid")

# ---------------------------------------------------------------------------
# Actual kill loop
# ---------------------------------------------------------------------------

kill_pidfile() {
  local pidfile="$1" sig="${2:-TERM}"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    msg "Sending SIG$sig to $(basename "$pidfile") (pid $pid)"
    kill -"$sig" "$pid" 2>/dev/null || true
    # Grace period for a clean shutdown unless --force was given.
    if [[ "$sig" != "KILL" ]]; then
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
      done
      if kill -0 "$pid" 2>/dev/null; then
        warn "Process $pid did not exit; escalating to SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
    ok "$(basename "$pidfile") stopped"
  fi
  rm -f "$pidfile"
}

sig="TERM"
[[ "$FORCE" = true ]] && sig="KILL"

for pidfile in "${PIDS[@]}"; do
  case "$pidfile" in
    */ollama.pid)
      if [[ "$STOP_OLLAMA" = false ]]; then
        # Our PID file is only present when start.sh booted Ollama for us,
        # so stopping it is the correct default. --stop-ollama overrides
        # this check by forcing a stop of *any* Ollama we can reach.
        kill_pidfile "$pidfile" "$sig"
      else
        kill_pidfile "$pidfile" "$sig"
      fi
      ;;
    *) kill_pidfile "$pidfile" "$sig" ;;
  esac
done

# ---------------------------------------------------------------------------
# --stop-ollama override
# ---------------------------------------------------------------------------

if [[ "$STOP_OLLAMA" = true ]] && [[ ! -f "$RUN_DIR/ollama.pid" ]]; then
  # User asked us to stop any Ollama we can find, but we never wrote a PID
  # file for it (something else started it). Fall back to matching on the
  # command line. This is best-effort and intentionally loud about what
  # it is doing.
  if pgrep -f 'ollama serve' >/dev/null 2>&1; then
    warn "Terminating 'ollama serve' process(es) we did not start"
    pkill -f 'ollama serve' 2>/dev/null || true
  fi
fi

# ---------------------------------------------------------------------------
# Belt-and-braces fallback
# ---------------------------------------------------------------------------

# If anything we started is still lingering (e.g. detached shell wrappers
# that got reparented), match on the process name as a fallback. This is
# the same command pattern ``install.sh`` uses to recover from port-in-use
# errors on previous runs.
if pgrep -f 'ollama_evaluator.cli.*serve' >/dev/null 2>&1; then
  warn "Found lingering ollama_evaluator.cli serve; terminating"
  pkill -f 'ollama_evaluator.cli.*serve' 2>/dev/null || true
fi

ok "Stop complete."
