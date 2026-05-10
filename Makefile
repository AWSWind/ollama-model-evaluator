# Convenience entry points for the Ollama Model Evaluator.
#
# The one-button installer lives in ``scripts/install.sh`` (Linux/macOS) and
# ``scripts/install.ps1`` (Windows). This Makefile is a thin wrapper so
# developers who live in ``make`` can run the same commands without having
# to remember the exact invocations.

PYTHON ?= python3
VENV   ?= .venv
VENV_PY := $(VENV)/bin/python

# ``make`` treats any target name that matches a file on disk as up to date
# unless we declare it phony. Every target below is command-only, so list
# them here so ``make install`` always runs the install.
.PHONY: help install install-skip-tests install-skip-ui test test-unit \
        test-property test-integration ui-build ui-test lint typecheck \
        serve validate list-models run clean deploy start start-bg start-dev \
        stop restart

## ----------------------------------------------------------------------------
## Default: print the list of targets with a one-liner each
## ----------------------------------------------------------------------------

help:
	@echo "Ollama Model Evaluator — make targets"
	@echo ""
	@echo "  make install              One-button local install (deps + UI + smoke test)"
	@echo "  make install-skip-tests   Same but skip the post-install smoke test"
	@echo "  make install-skip-ui      Backend only, skip Node/Vite build"
	@echo ""
	@echo "  make test                 Full backend test suite"
	@echo "  make test-unit            Unit tests only (fastest)"
	@echo "  make test-property        Hypothesis property tests"
	@echo "  make test-integration     Integration tests (fake Ollama + FastAPI)"
	@echo ""
	@echo "  make ui-build             Build ui/dist/ bundle"
	@echo "  make ui-test              Vitest run"
	@echo ""
	@echo "  make lint                 Ruff"
	@echo "  make typecheck            mypy"
	@echo ""
	@echo "  make validate             Validate examples/suites/reasoning-basics.yaml"
	@echo "  make list-models          Ask Ollama what models it has"
	@echo "  make run                  Run the example evaluation"
	@echo "  make serve                Start the HTTP + WebSocket + UI server"
	@echo ""
	@echo "  make start                One-button launch: Ollama + backend + UI (foreground)"
	@echo "  make start-bg             Same but detach after ready"
	@echo "  make start-dev            Also run the Vite dev server on :5173"
	@echo "  make stop                 Stop everything started by \`make start\`"
	@echo "  make restart              Stop then start-bg"
	@echo ""
	@echo "  make deploy TARGET=user@host [REMOTE_DIR=/path] [SERVE_PORT=8765]"
	@echo "                             Push + install on a remote host via SSH"
	@echo ""
	@echo "  make clean                Remove build artefacts and caches"

## ----------------------------------------------------------------------------
## One-button install
## ----------------------------------------------------------------------------

install:
	@bash scripts/install.sh

install-skip-tests:
	@bash scripts/install.sh --skip-tests

install-skip-ui:
	@bash scripts/install.sh --skip-ui

## ----------------------------------------------------------------------------
## Backend tests
## ----------------------------------------------------------------------------

test:
	$(VENV_PY) -m pytest backend -q

test-unit:
	$(VENV_PY) -m pytest backend/tests/unit -q

test-property:
	$(VENV_PY) -m pytest backend/tests/property -q

test-integration:
	$(VENV_PY) -m pytest backend/tests/integration -q

## ----------------------------------------------------------------------------
## UI
## ----------------------------------------------------------------------------

ui-build:
	cd ui && npm run build

ui-test:
	cd ui && npm test

## ----------------------------------------------------------------------------
## Quality
## ----------------------------------------------------------------------------

lint:
	$(VENV_PY) -m ruff check backend/src backend/tests

typecheck:
	$(VENV_PY) -m mypy backend/src

## ----------------------------------------------------------------------------
## Running the tool
## ----------------------------------------------------------------------------

CONFIG ?= examples/config.qwen.yaml
SUITE  ?= examples/suites/reasoning-basics.yaml
PORT   ?= 8765

validate:
	$(VENV_PY) -m ollama_evaluator.cli validate-suite $(SUITE)

list-models:
	$(VENV_PY) -m ollama_evaluator.cli list-models

run:
	$(VENV_PY) -m ollama_evaluator.cli --config $(CONFIG) run

serve:
	OLLAMA_EVAL_UI_DIR=$$PWD/ui/dist $(VENV_PY) -m ollama_evaluator.cli \
	    --config $(CONFIG) serve --host 0.0.0.0 --port $(PORT)

## ----------------------------------------------------------------------------
## One-button launcher
## ----------------------------------------------------------------------------

start:
	@bash scripts/start.sh --port $(PORT) $(if $(DEV),--dev,) $(if $(BACKGROUND),--background,)

start-bg:
	@bash scripts/start.sh --port $(PORT) --background

start-dev:
	@bash scripts/start.sh --port $(PORT) --dev

stop:
	@bash scripts/stop.sh

restart: stop
	@sleep 1
	@bash scripts/start.sh --port $(PORT) --background

## ----------------------------------------------------------------------------
## Remote deploy
## ----------------------------------------------------------------------------

REMOTE_DIR ?=
SERVE_PORT ?=
SSH_KEY    ?=

deploy:
	@if [ -z "$(TARGET)" ]; then \
		echo "Usage: make deploy TARGET=user@host [REMOTE_DIR=/path] [SERVE_PORT=8765] [SSH_KEY=~/.ssh/id]"; \
		exit 2; \
	fi
	@bash scripts/deploy-remote.sh \
	    $(if $(SSH_KEY),--key $(SSH_KEY),) \
	    $(if $(SERVE_PORT),--serve $(SERVE_PORT),) \
	    $(TARGET) $(REMOTE_DIR)

## ----------------------------------------------------------------------------
## Housekeeping
## ----------------------------------------------------------------------------

clean:
	@find backend -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	@find backend -type d -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf backend/.pytest_cache backend/.mypy_cache backend/.ruff_cache backend/.hypothesis
	@rm -rf ui/dist
	@echo "Cleaned caches and build artefacts (venv + node_modules preserved)."
