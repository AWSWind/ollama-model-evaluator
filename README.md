# Ollama Model Evaluator

Local evaluation harness for LLMs hosted by a running [Ollama](https://ollama.com) server.

Score quality, measure speed, and compare models side-by-side — all offline.

## Start the whole project with one command

### Linux/macOS
```bash
./scripts/start.sh
```

### Windows
```powershell
.\scripts\start.ps1
```
…or double-click `start.bat` in Explorer.

The launcher automatically:
1. Runs `install.sh` / `install.ps1` if the Python venv or UI bundle is missing.
2. Starts the Ollama daemon if it isn't already running.
3. Starts the backend (FastAPI on port 8765) with the built UI mounted at `/`.
4. Health-checks every service before handing back.
5. Tails the backend log; Ctrl-C cleanly stops everything we started.

Then open [`http://localhost:8765/`](http://localhost:8765/) in a browser.

Common variations:
```bash
./scripts/start.sh --background        # detach after services are ready
./scripts/start.sh --port 9000         # use a different backend port
./scripts/start.sh --dev               # also run Vite on :5173 with HMR
./scripts/start.sh --skip-ollama       # skip Ollama entirely (UI only)
./scripts/stop.sh                      # stop everything later
```

Windows equivalents:
```powershell
.\scripts\start.ps1 -Detach
.\scripts\start.ps1 -BackendPort 9000
.\scripts\start.ps1 -Dev
.\scripts\start.ps1 -SkipOllama
.\scripts\stop.ps1
```

## One-button install (if you only want to install, not run)

```bash
./scripts/install.sh
```

Builds the Python venv, installs the backend, builds the UI bundle, and runs a
short smoke test. Re-run it after `git pull` to pick up new dependencies.

Flags:
- `--skip-tests` — skip the post-install unit-test run.
- `--skip-ui` — backend only (no Node needed).
- `--no-venv` — install into system Python.
- `--python PATH` — use a specific interpreter.

### One-button install (Windows)

```powershell
.\scripts\install.ps1
```

Accepts `-SkipTests`, `-SkipUI`, `-NoVenv`, `-Python <path>`.

### One-button remote deploy

Push the repo to a remote server and install there in a single command.

From Linux/macOS:
```bash
./scripts/deploy-remote.sh user@host [/target/dir]
./scripts/deploy-remote.sh user@host /home/you/eval --skip-tests --serve 8765
```

From Windows:
```powershell
.\scripts\deploy-remote.ps1 -Target user@host -RemoteDir /home/you/eval
.\scripts\deploy-remote.ps1 -Target user@host -SkipTests -ServePort 8765
```

What the remote deploy does:
1. Tars the repo (excluding caches, `node_modules`, `.venv`, `.git`).
2. `scp`'s the tarball to the remote.
3. Extracts into `RemoteDir` (default `~/ollama-model-evaluator`).
4. Runs `scripts/install.sh` on the remote.
5. Optionally starts the web server in the background.

Prerequisites:
- Local: `ssh`, `scp`, `tar`.
- Remote: `bash`, `python3 >= 3.11`, `node >= 18`, `npm`. Ollama is only needed
  to actually run evaluations.

## After install

The fastest path after install is the one-button launcher:

```bash
./scripts/start.sh              # Linux / macOS
.\scripts\start.ps1             # Windows
```

Or keep running things manually:

```bash
cd <repo>
source .venv/bin/activate                  # Windows: .\.venv\Scripts\Activate.ps1

# Pull a small Ollama model if you have not already
ollama pull llama3:8b

# Edit examples/config.qwen.yaml — point `models:` at yours
python -m ollama_evaluator.cli validate-suite examples/suites/reasoning-basics.yaml
python -m ollama_evaluator.cli list-models
python -m ollama_evaluator.cli --config examples/config.qwen.yaml run

# Start the HTTP + WebSocket + UI server
OLLAMA_EVAL_UI_DIR=$PWD/ui/dist \
  python -m ollama_evaluator.cli --config examples/config.qwen.yaml serve
```

Then open [`http://localhost:8765/`](http://localhost:8765/) in your browser.

See **[docs/USER_MANUAL.md](docs/USER_MANUAL.md)** for a beginner-friendly
walkthrough.

## Make targets

```bash
make install              # One-button local install
make install-skip-tests   # Faster re-install
make test                 # Full backend test suite
make test-unit            # Unit tests only
make ui-build             # Rebuild ui/dist/
make ui-test              # Run vitest
make run                  # Run the example evaluation
make serve                # Start the server on port 8765
make deploy TARGET=user@host [REMOTE_DIR=/path] [SERVE_PORT=8765]
make clean                # Wipe caches (venv + node_modules preserved)
make help                 # See every target
```

## Repository layout

```
.
├── backend/         # Python backend (FastAPI + CLI)
├── ui/              # Vite + React + TypeScript UI
├── shared/          # OpenAPI + JSON Schemas shared by both
├── examples/        # Sample config + suite you can run today
├── scripts/         # install.sh / install.ps1 / deploy-remote.sh / .ps1
├── docs/            # USER_MANUAL.md
└── .kiro/specs/     # Requirements, design, task list
```

## Documentation

- **[docs/USER_MANUAL.md](docs/USER_MANUAL.md)** — hands-on guide for new users.
- **[.kiro/specs/ollama-model-evaluator/requirements.md](.kiro/specs/ollama-model-evaluator/requirements.md)** — functional requirements.
- **[.kiro/specs/ollama-model-evaluator/design.md](.kiro/specs/ollama-model-evaluator/design.md)** — system design and architecture.
- **[.kiro/specs/ollama-model-evaluator/tasks.md](.kiro/specs/ollama-model-evaluator/tasks.md)** — implementation plan / task list.

## Testing

```bash
make test                        # All 447 backend tests (~45s)
make test-unit                   # Unit tests only (fastest)
make test-property               # Hypothesis property tests
make test-integration            # FastAPI + fake Ollama integration tests
make ui-test                     # UI tests via Vitest
```

## Troubleshooting

See the [troubleshooting section of the user manual](docs/USER_MANUAL.md#11-troubleshooting).

Most common issues:
- **"ollama_unreachable"** — check `curl http://localhost:11434/api/version`.
- **Empty responses** — reasoning models need `max_tokens: 512+`. Already set
  in `examples/suites/reasoning-basics.yaml`.
- **Schema drift** — run `python backend/scripts/regen_schemas.py`.

## License

See `LICENSE` (if present) or the top of individual source files.
