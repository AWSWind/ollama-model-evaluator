#!/usr/bin/env bash
# Ollama Model Evaluator — one-button remote deployment.
#
# Runs on any machine with ssh/scp, pushes the repo to a remote host, and
# drives the remote's install.sh to produce a working install over there.
#
# Usage (from the repo root):
#
#     ./scripts/deploy-remote.sh user@host [target-dir]
#
# Examples:
#
#     # Default target dir is ~/ollama-model-evaluator on the remote
#     ./scripts/deploy-remote.sh azurewind@192.168.1.224
#
#     # Explicit target dir
#     ./scripts/deploy-remote.sh azurewind@192.168.1.224 /home/azurewind/workspaces/AI-Model-Evaluation
#
# Flags:
#     --key PATH          SSH private key to use.
#     --port N            SSH port (default 22).
#     --skip-tests        Pass --skip-tests to the remote install.sh.
#     --skip-ui           Pass --skip-ui to the remote install.sh.
#     --no-install        Only sync files; do not run install.sh.
#     --serve PORT        Start ``cli serve`` on the remote on PORT after
#                         install, in the background.
#     --config PATH       Override the config file the remote serve uses
#                         (relative to target-dir; default
#                         examples/config.qwen.yaml).
#     -h, --help          Show this help.
#
# What this script does:
#   1. Builds a clean tarball of the repo, excluding caches, node_modules,
#      and virtualenvs.
#   2. scp's the tarball to the remote.
#   3. Extracts it into target-dir on the remote, making the dir if needed.
#   4. Runs scripts/install.sh on the remote.
#   5. Optionally launches the server in the background via ``nohup``.
#
# Prerequisites on the control host: ssh, scp, tar.
# Prerequisites on the remote host: bash, python3>=3.11, node>=18, npm,
# Ollama (optional but required for actual evaluation runs).

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
  sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

SSH_TARGET=""
REMOTE_DIR=""
SSH_KEY=""
SSH_PORT="22"
INSTALL_FLAGS=()
DO_INSTALL=true
SERVE_PORT=""
SERVE_CONFIG="examples/config.qwen.yaml"

# First collect positional + flag args. ``--`` is the standard "stop parsing
# flags" sentinel; we honour it so callers can pass literal strings that
# start with a dash in the positional slots.
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)       usage ;;
    --key)           SSH_KEY="${2:-}"; shift ;;
    --port)          SSH_PORT="${2:-}"; shift ;;
    --skip-tests)    INSTALL_FLAGS+=("--skip-tests") ;;
    --skip-ui)       INSTALL_FLAGS+=("--skip-ui") ;;
    --no-install)    DO_INSTALL=false ;;
    --serve)         SERVE_PORT="${2:-}"; shift ;;
    --config)        SERVE_CONFIG="${2:-}"; shift ;;
    --)              shift; while [[ $# -gt 0 ]]; do POSITIONAL+=("$1"); shift; done ;;
    -*)              die "Unknown option: $1 (use --help to see valid options)" ;;
    *)               POSITIONAL+=("$1") ;;
  esac
  shift
done

[[ ${#POSITIONAL[@]} -ge 1 ]] || die "SSH target required. Example: $0 user@host"
SSH_TARGET="${POSITIONAL[0]}"
REMOTE_DIR="${POSITIONAL[1]:-~/ollama-model-evaluator}"

# ---------------------------------------------------------------------------
# Tooling detection on the control host
# ---------------------------------------------------------------------------

section "Preflight (control host)"

command -v ssh >/dev/null 2>&1 || die "ssh not found on this machine"
command -v scp >/dev/null 2>&1 || die "scp not found on this machine"
command -v tar >/dev/null 2>&1 || die "tar not found on this machine"
ok "ssh, scp, tar available"

# Build a single argument vector for ``ssh`` / ``scp`` invocations so every
# subsequent call inherits the same key + port.
SSH_OPTS=()
if [[ -n "$SSH_KEY" ]]; then
  [[ -f "$SSH_KEY" ]] || die "SSH key not found: $SSH_KEY"
  SSH_OPTS+=("-i" "$SSH_KEY")
fi
# Keep the ssh invocation non-interactive. ``StrictHostKeyChecking=accept-new``
# auto-trusts hosts the client has not seen before (writing the fingerprint
# to ``~/.ssh/known_hosts``) but still refuses changed keys; a fresh WSL or
# container that has never connected to the target otherwise aborts on the
# "The authenticity of host ... can't be established" prompt.
SSH_OPTS+=("-o" "BatchMode=yes" "-o" "StrictHostKeyChecking=accept-new" "-p" "$SSH_PORT")

SCP_OPTS=()
if [[ -n "$SSH_KEY" ]]; then
  SCP_OPTS+=("-i" "$SSH_KEY")
fi
SCP_OPTS+=("-o" "BatchMode=yes" "-o" "StrictHostKeyChecking=accept-new" "-P" "$SSH_PORT")

remote_run() {
  # Run a command on the remote with the accumulated SSH options.
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"
}

# Quick connectivity check.
msg "Verifying SSH access to $SSH_TARGET"
if ! remote_run "echo REMOTE_OK" >/dev/null 2>&1; then
  die "Cannot connect to $SSH_TARGET via ssh (check key auth, host, port, firewall)"
fi
ok "SSH access confirmed"

# ---------------------------------------------------------------------------
# Tarball creation
# ---------------------------------------------------------------------------

section "Building deployment tarball"

# ``mktemp`` gives us a unique path regardless of how many deploys run in
# parallel. The trap ensures we do not leave temp files around if the script
# exits early.
TARBALL="$(mktemp -t ollama-eval-XXXXXX.tgz)"
cleanup() {
  rm -f "$TARBALL" 2>/dev/null || true
}
trap cleanup EXIT

# ``--exclude`` ordering matters: we deliberately exclude every dev artefact
# that would bloat the tarball or leak ambient state from one machine to
# another. ``scripts/install.sh`` on the remote will regenerate everything
# we drop.
cd "$REPO_ROOT"
tar \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.hypothesis' \
  --exclude='.venv' \
  --exclude='node_modules' \
  --exclude='dist' \
  --exclude='.git' \
  -czf "$TARBALL" \
  backend ui shared examples scripts docs README.md

TARBALL_SIZE_KB="$(du -k "$TARBALL" | awk '{print $1}')"
ok "Tarball built: $TARBALL ($TARBALL_SIZE_KB KiB)"

# ---------------------------------------------------------------------------
# Upload + extract
# ---------------------------------------------------------------------------

section "Uploading to $SSH_TARGET:$REMOTE_DIR"

# ``mkdir -p`` + ``tar -xzf`` is the minimal extraction surface. Running
# them over ssh instead of bundling into a shell script keeps the remote
# side's responsibility obvious. The ``rm -f`` at the end deletes the
# transient tarball; ``install.sh`` will not use it.
REMOTE_TARBALL="/tmp/ollama-evaluator-deploy.$(date +%s).tgz"

msg "Uploading tarball"
scp "${SCP_OPTS[@]}" "$TARBALL" "$SSH_TARGET:$REMOTE_TARBALL" >/dev/null
ok "Uploaded"

msg "Extracting into $REMOTE_DIR"
remote_run "mkdir -p $REMOTE_DIR && \
            tar -xzf $REMOTE_TARBALL -C $REMOTE_DIR && \
            rm -f $REMOTE_TARBALL"
ok "Project synced"

# ---------------------------------------------------------------------------
# Remote install
# ---------------------------------------------------------------------------

if [[ "$DO_INSTALL" = true ]]; then
  section "Running install.sh on the remote"

  # The flag list is expanded through the shell on the remote side, so we
  # join the array with spaces and rely on the remote bash to re-split.
  # Every flag we forward is a known short option without embedded spaces.
  REMOTE_FLAGS="${INSTALL_FLAGS[*]:-}"

  # ``set -o pipefail`` on the remote means the whole pipeline fails if
  # install.sh fails, even though we pipe through ``tee`` for live output.
  remote_run "set -o pipefail; \
              cd $REMOTE_DIR && \
              chmod +x scripts/install.sh && \
              ./scripts/install.sh $REMOTE_FLAGS"

  ok "Remote install finished"
fi

# ---------------------------------------------------------------------------
# Optional: start the server
# ---------------------------------------------------------------------------

if [[ -n "$SERVE_PORT" ]]; then
  section "Starting server on remote (port $SERVE_PORT)"

  # Kill any previous evaluator server so we do not hit "address already
  # in use". ``|| true`` swallows the non-zero exit when no process matches.
  remote_run "pkill -f 'ollama_evaluator.cli.*serve' 2>/dev/null || true; sleep 1"

  # ``nohup`` + ``&`` detaches the process so the ssh channel can close.
  # ``OLLAMA_EVAL_UI_DIR`` makes the server mount the built UI at ``/``;
  # ``serve.log`` captures stdout+stderr under the project root.
  remote_run "cd $REMOTE_DIR && \
              nohup bash -c 'source .venv/bin/activate && \
                             OLLAMA_EVAL_UI_DIR=\$PWD/ui/dist \
                             python -m ollama_evaluator.cli --config $SERVE_CONFIG \
                               serve --host 0.0.0.0 --port $SERVE_PORT' \
                > serve.log 2>&1 &"
  sleep 2
  if remote_run "ss -tlnp 2>/dev/null | grep -q ':$SERVE_PORT '"; then
    ok "Server listening on $SSH_TARGET:$SERVE_PORT"
  else
    warn "Server did not bind port $SERVE_PORT within 2s; tail of serve.log:"
    remote_run "tail -20 $REMOTE_DIR/serve.log" || true
  fi

  HOSTNAME_ONLY="${SSH_TARGET#*@}"
  cat <<URL

${C_BOLD}Server ready.${C_RESET} Try from your browser:
  ${C_DIM}http://$HOSTNAME_ONLY:$SERVE_PORT/${C_RESET}        ${C_DIM}# Web UI${C_RESET}
  ${C_DIM}http://$HOSTNAME_ONLY:$SERVE_PORT/api/health${C_RESET}  ${C_DIM}# Liveness probe${C_RESET}
  ${C_DIM}http://$HOSTNAME_ONLY:$SERVE_PORT/api/models${C_RESET}  ${C_DIM}# Ollama model list${C_RESET}

${C_BOLD}Stop the remote server:${C_RESET}
  ${C_DIM}ssh ${SSH_KEY:+-i $SSH_KEY }${SSH_TARGET#*@/} "pkill -f 'ollama_evaluator.cli.*serve'"${C_RESET}
URL
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

cat <<DONE

${C_GREEN}${C_BOLD}Deployment complete.${C_RESET}
  Remote path: $SSH_TARGET:$REMOTE_DIR

${C_BOLD}Try a quick run over ssh:${C_RESET}
  ${C_DIM}ssh ${SSH_KEY:+-i $SSH_KEY }${SSH_TARGET} "cd $REMOTE_DIR && source .venv/bin/activate && python -m ollama_evaluator.cli list-models"${C_RESET}

DONE
