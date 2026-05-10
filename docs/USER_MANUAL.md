# Ollama Model Evaluator — User Manual

A friendly guide for developers and curious users who want to evaluate large
language models (LLMs) running locally via [Ollama](https://ollama.com).

**Last updated**: May 2026 · **Version**: 0.1.0

---

## Table of contents

1. [What this tool does](#1-what-this-tool-does)
2. [Concepts in plain English](#2-concepts-in-plain-english)
3. [Installation prerequisites](#3-installation-prerequisites)
4. [Your first evaluation in 5 minutes](#4-your-first-evaluation-in-5-minutes)
5. [Using the web UI](#5-using-the-web-ui)
6. [Using the command line](#6-using-the-command-line)
7. [Writing your own test suites](#7-writing-your-own-test-suites)
8. [Using public benchmarks (MMLU, GSM8K, etc.)](#8-using-public-benchmarks)
9. [Running remotely via SSH](#9-running-remotely-via-ssh)
10. [Reading the reports](#10-reading-the-reports)
11. [Troubleshooting](#11-troubleshooting)
12. [Command reference](#12-command-reference)
13. [Glossary](#13-glossary)

---

## 1. What this tool does

You have one or more LLMs running locally through Ollama (for example
`qwen3.6:27b` or `llama3:8b`). You want to know:

- How fast is each model?
- How accurate is it on questions you care about?
- Which model should I pick for my task?
- Has a newer model improved on the old one?

This tool runs a set of questions through each model, scores the answers,
measures the speed, and gives you a report — both a spreadsheet-friendly JSON
file and a human-readable Markdown summary.

You can drive it three ways:

1. **A web UI** in your browser (easiest to start with).
2. **A command-line tool** (best for scripting and CI pipelines).
3. **An HTTP API** (if you're building your own dashboard on top).

Everything runs on your own hardware. Nothing is sent to a cloud service.

### Example question the tool answers

> "Is `llama3:8b` faster than `qwen3.6:27b` on basic arithmetic, and which one
> gets more answers right?"

The tool runs the same arithmetic questions on both models, captures the
response, checks if the response contains the right answer, measures how many
tokens per second each model produced, and gives you a side-by-side comparison.

---

## 2. Concepts in plain English

Before you touch anything, here are the words used throughout this manual.

### Ollama

A program that runs LLMs on your computer. It listens on port `11434` and
exposes a simple HTTP API. You pull models (like downloading apps) and then
ask them questions. Install from [ollama.com/download](https://ollama.com/download).

### Model

A specific LLM identified by a name and tag, like `llama3:8b` or
`qwen3.6:27b`. The part after the colon usually indicates size or variant.

### Test case

One question (called a "prompt") plus the expected answer (or a rule for
what counts as a correct answer) plus a name. Example:

```
id: arithmetic-simple
prompt: "What is 2 + 2?"
expected_output: "4"
metric: contains "4"
```

### Evaluation suite

A named collection of test cases. You typically group related questions
together — a "math" suite, a "code" suite, a "safety" suite.

### Metric

A rule that scores a response. Built-in ones:

- **exact-match** — response must equal the expected output letter-for-letter.
- **regex-match** — response must match a pattern (like "any 4 in the text").
- **contains** — response must contain certain substrings.
- **json-schema-valid** — response must be valid JSON matching a shape.
- **length-range** — response must be within a certain length.
- **llm-as-judge** — another LLM scores the response against a rubric.
- **response-capture** — always passes; records the raw response for later grading.

### Run

One execution of the tool. A Run takes the configuration ("evaluate these
models against these suites") and produces a Run Report.

### Run report

The output of a Run. Contains every response, every metric score, timings,
and totals. Written as both `report.json` and `report.md`.

### Repetition

How many times each test case is executed per model. Useful for measuring
variance in non-deterministic models. Default is 1.

### Concurrency

How many questions can be in flight at the same time. Default is 1
(sequential) to keep GPU memory usage predictable.

### Backend / UI

The backend is the Python program that orchestrates everything. The UI is a
React web app that talks to the backend. When you run `serve`, the backend
starts a local web server and also serves the UI bundle so you can open it
in a browser.

---

## 3. Installation prerequisites

You need four things:

| What | Version | Why |
|---|---|---|
| Python | 3.11 or newer | Backend language |
| Node.js | 18 or newer | Builds the web UI bundle |
| Ollama | 0.5 or newer | Serves the LLMs being evaluated |
| A small Ollama model | any | Something to evaluate |

### Installing Ollama

On Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On macOS or Windows, download from [ollama.com/download](https://ollama.com/download).

After install, pull at least one model:
```bash
ollama pull llama3:8b   # 4.7 GB, general-purpose
# or, if you have more RAM / GPU memory
ollama pull qwen3.6:27b # ~17 GB, reasoning model
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/version
# → {"version":"0.20.3"}
```

### Installing this tool

You have two options:

#### Option A — One-button installer (recommended)

The repo ships with a one-command install script that handles Python venv,
backend install, UI build, schema regeneration, and a smoke test.

**Linux/macOS:**
```bash
./scripts/install.sh
```

**Windows PowerShell:**
```powershell
.\scripts\install.ps1
```

Both installers are idempotent — run them again after `git pull` to pick up
dependency changes. Pass `--skip-tests` (bash) or `-SkipTests` (PowerShell) to
skip the unit-test run and speed up re-installs.

#### Option B — Manual install

This tool is not yet on PyPI or npm. You clone the repository and install
from source.

```bash
# 1. Get the code (example path; put it wherever you like)
git clone <your-repo-url> AI-Model-Evaluation
cd AI-Model-Evaluation

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate           # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -e "backend[dev]"       # Installs the backend editable

# 3. UI dependencies and bundle
cd ui
npm install                         # Downloads ~200 packages
npm run build                       # Produces ui/dist/
cd ..
```

Verify either install:
```bash
python -m ollama_evaluator.cli --help
# You should see a help screen listing subcommands
```

---

## 4. Your first evaluation in 5 minutes

The repository ships with an `examples/` directory containing a ready-to-run
suite. There are two ways to get going: the one-button launcher that starts
every service for you, or the step-by-step commands below.

### The one-button path (simplest)

```bash
./scripts/start.sh              # Linux / macOS
```
```powershell
.\scripts\start.ps1             # Windows PowerShell
```

Or just double-click `start.bat` on Windows.

What it does, in order:
1. If `.venv/` or `ui/dist/` are missing, runs `install.sh` / `install.ps1`
   for you.
2. If Ollama isn't already running on port 11434 and you have the `ollama` CLI
   installed, starts `ollama serve` in the background.
3. Starts the FastAPI backend on port 8765, with the built UI mounted at `/`.
4. Waits until every service responds to a health check before handing back.
5. Tails the backend log; Ctrl-C stops every service it started (but *not* an
   Ollama it found already running — that one keeps going so other apps on
   your machine aren't disturbed).

After a few seconds you'll see:
```
All services ready.

  Web UI           http://localhost:8765/
  Health probe     http://localhost:8765/api/health
  REST API docs    http://localhost:8765/openapi.json
```

Open the UI URL in your browser.

Stop everything later with:
```bash
./scripts/stop.sh      # or Ctrl-C if start.sh is still in the foreground
```
```powershell
.\scripts\stop.ps1
```

Useful flags:
- `--background` / `-Detach` — return immediately instead of tailing the log.
- `--port 9000` / `-BackendPort 9000` — pick a different backend port.
- `--dev` / `-Dev` — also start the Vite dev server with hot-reload.
- `--skip-ollama` / `-SkipOllama` — skip the Ollama check/start.
- `--host 127.0.0.1` — bind only to localhost (default is 0.0.0.0 = LAN).

### The step-by-step path (for learning what each piece does)

### Step 1 — Check the example

Open `examples/suites/reasoning-basics.yaml`. It has three questions:

```yaml
name: reasoning-basics
defaults:
  temperature: 0.0
  max_tokens: 512
test_cases:
  - id: arithmetic-simple
    prompt: "What is 2 + 2? Answer with just the number."
    expected_output: "4"
    metrics:
      - name: regex-match
        params:
          pattern: "\\b4\\b"          # "look for a standalone 4 anywhere"
  - id: capital-of-france
    prompt: "What is the capital of France? Answer with just the city name."
    expected_output: "Paris"
    metrics:
      - name: contains
        params:
          substrings: ["Paris"]
          mode: any
  - id: python-syntax
    prompt: "Write a Python one-liner that returns the square of 5. Respond with code only."
    metrics:
      - name: regex-match
        params:
          pattern: "25"
```

### Step 2 — Adjust the config

Open `examples/config.qwen.yaml`:

```yaml
ollama_base_url: http://localhost:11434
suites_dir: examples/suites
output_dir: examples/runs
run:
  models:
    - qwen3.6:27b                    # ← change this to your model
  suites:
    - reasoning-basics
  repetitions: 1
  concurrency: 1
  ollama_timeout_s: 300.0
```

Change `models:` to whatever you pulled, e.g. `- llama3:8b`.

### Step 3 — Validate the suite file (optional, no Ollama needed)

```bash
python -m ollama_evaluator.cli validate-suite examples/suites/reasoning-basics.yaml
# → OK: reasoning-basics (3 test cases)
```

If the YAML has typos, this step tells you the exact field and line.

### Step 4 — List available models

```bash
python -m ollama_evaluator.cli list-models
# → llama3:8b	sha256:abc...	8B
```

This confirms the backend can reach Ollama.

### Step 5 — Run the evaluation

```bash
python -m ollama_evaluator.cli --config examples/config.qwen.yaml run
```

You will see HTTP log lines, then:

```
run f3c9a1...: 3 executions, passed=2, failed=1, error=0, timeout=0
```

The process exits with code 0 if every test passed, 1 if any failed, 2 on
setup problems like "Ollama not reachable". This matters for CI pipelines.

### Step 6 — Read the report

```bash
cat examples/runs/f3c9a1.../report.md
```

You'll see per-model and per-suite tables. For a deeper look:

```bash
python3 scripts/remote_show_report.py
```

This prints every response plus timings.

---

## 5. Using the web UI

If you prefer a browser experience, start the server:

```bash
OLLAMA_EVAL_UI_DIR=$(pwd)/ui/dist \
  python -m ollama_evaluator.cli --config examples/config.qwen.yaml serve \
    --host 0.0.0.0 --port 8765
```

Flag explanations:
- `--host 0.0.0.0` — listen on every network interface (use `127.0.0.1` to bind
  only to the local machine).
- `--port 8765` — TCP port to listen on. Change if something else uses this.
- `OLLAMA_EVAL_UI_DIR` — tells the backend where the built UI bundle lives
  so `/` serves the React app.

Leave this command running. Open a browser to `http://<host>:8765/`.

### UI pages

- **New Run** (`/runs/new`). Select models from a dropdown, pick one or more
  suites, set repetitions and concurrency, click *Submit Run*. You are
  immediately navigated to the Run Detail page.

- **Run Detail** (`/runs/:id`). Live progress bar, counters (passed / failed /
  error / timeout), and a table that fills in as each test case completes.
  A red banner appears if the stream disconnects. There is a *Cancel* button
  while the Run is in progress. When the Run finishes, a download link appears
  for the JSON and Markdown reports.

- **History** (`/history`). Every past Run. Filter by model, suite, status,
  or date. Check two rows and click *Compare* to open a diff view.

- **Compare** (`/compare?a=…&b=…`). Side-by-side metric scores and performance
  numbers for two Runs.

Stop the server with `Ctrl-C`.

---

## 6. Using the command line

For everyday use, the CLI is faster than clicking through the UI.

### Common commands

```bash
# Activate the venv every new terminal session
cd AI-Model-Evaluation
source .venv/bin/activate

# See help
python -m ollama_evaluator.cli --help
python -m ollama_evaluator.cli run --help

# Validate a suite file offline (does not need Ollama)
python -m ollama_evaluator.cli validate-suite path/to/suite.yaml

# List what Ollama has available
python -m ollama_evaluator.cli list-models

# Execute a Run
python -m ollama_evaluator.cli --config examples/config.qwen.yaml run

# Compare two Runs by id (ids come from the output of `run`)
python -m ollama_evaluator.cli --config examples/config.qwen.yaml \
  compare f3c9a1...  d712fe...

# Start the HTTP + WebSocket server
python -m ollama_evaluator.cli --config examples/config.qwen.yaml serve
```

### Global flags

| Flag | Default | Purpose |
|---|---|---|
| `--config <path>` | required | Config file path |
| `--output-dir <path>` | from config | Override output directory |
| `--log-level {debug,info,warn,error}` | `info` | Verbosity |
| `--dataset-mode {local,remote}` | `local` | Local cache vs HuggingFace Hub |
| `--hf-cache-dir <path>` | HF default | Where HuggingFace caches datasets |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Every test case passed |
| `1` | At least one test case failed |
| `2` | Preflight error (Ollama unreachable, missing model, etc.) |

This matters when scripting. In a CI pipeline you can do:

```bash
python -m ollama_evaluator.cli --config ci.yaml run
if [ $? -ne 0 ]; then
  echo "Evaluation regressed, failing build"
  exit 1
fi
```

---

## 7. Writing your own test suites

A suite is a YAML (or JSON) file in your `suites_dir`. Minimum shape:

```yaml
name: my-first-suite
test_cases:
  - id: hello-world
    prompt: "Say hello"
    metrics:
      - name: contains
        params:
          substrings: ["hello", "Hello"]
          mode: any
```

The `id` must be unique within the suite. `prompt` and `metrics` are required.
Everything else is optional.

### Fuller example

```yaml
name: math-deep
version: "1.0"
description: "Math problems of increasing difficulty"
defaults:
  temperature: 0.0         # Lower = more deterministic
  max_tokens: 256
  stop_sequences: []
test_cases:
  - id: add-tiny
    prompt: "What is 7 plus 5?"
    expected_output: "12"
    tags: [math, easy]
    metrics:
      - name: regex-match
        params:
          pattern: "\\b12\\b"
  - id: mul-tiny
    prompt: "What is 7 times 5? Answer with just the number."
    expected_output: "35"
    tags: [math, easy]
    metrics:
      - name: regex-match
        params:
          pattern: "\\b35\\b"
  - id: word-problem
    prompt: |
      A train leaves station A at 2pm going 60mph.
      Another train leaves station B at 3pm going 80mph.
      They are 340 miles apart. When do they meet?
      Answer in the form "HH:MM pm".
    expected_output: "5:00 pm"
    tags: [math, hard, word-problem]
    temperature: 0.2        # Per-case override
    metrics:
      - name: contains
        params:
          substrings: ["5:00 pm", "5 pm", "17:00"]
          mode: any
```

### Optional per-test-case fields

- `system_prompt` — string prepended to the request as a system message.
- `expected_output` — reference answer, used by `exact-match`, `regex-match`.
- `reference_data` — arbitrary structured data (used by external graders).
- `tags` — list of strings, for filtering at run-time (`tag_filter:`).
- `temperature`, `max_tokens`, `stop_sequences` — per-case generation overrides.

### Tag filtering

In your config:

```yaml
run:
  suites: [math-deep]
  tag_filter: [easy]   # Only test cases whose tags include "easy"
```

An empty `tag_filter: []` means "run every test case".

### Tips for writing good suites

- **Keep prompts short and specific.** Ambiguous prompts make metrics
  unreliable.
- **Use `temperature: 0.0` unless you want randomness.** For reproducible
  scoring, determinism is your friend.
- **Write forgiving metrics.** `contains` or `regex-match` usually works
  better than `exact-match` because models often add whitespace or
  explanations you don't care about.
- **Give reasoning models enough `max_tokens`.** Qwen and similar models
  emit a chain-of-thought; if you cap them at 64 tokens they'll never get to
  the actual answer. 512–1024 is a safe starting point.
- **Version your suites with Git.** Commit the YAML so you can reproduce any
  historical result.

---

## 8. Using public benchmarks

The tool ships with adapters for five well-known benchmarks and a generic
HuggingFace loader.

### Supported benchmarks

| Benchmark | Dataset | Metric | Notes |
|---|---|---|---|
| MMLU | `cais/mmlu` | regex-match (letter A/B/C/D) | Per-subject suites |
| HellaSwag | `Rowan/hellaswag` | regex-match | Continuation selection |
| TruthfulQA | `truthful_qa:multiple_choice` | regex-match | MC1 form only |
| GSM8K | `openai/gsm8k` | regex-match + numeric equality | Grade-school math |
| HumanEval | `openai_humaneval` | response-capture | v1 does not execute code |

### Converting a benchmark to a suite file

Let's say you want MMLU's "abstract algebra" subject:

```bash
python -m ollama_evaluator.cli convert mmlu \
  --source ./cache/mmlu \
  --output ./examples/suites \
  --subjects abstract_algebra
```

This reads the dataset from the local cache and writes a normal YAML suite
under `./examples/suites/mmlu-abstract_algebra.yaml`. From then on it behaves
like any hand-written suite.

### Arbitrary HuggingFace datasets

You can target any HF dataset with a field map. Create `squad-fields.yaml`:

```yaml
prompt: "question"
expected_output: "answers.text[0]"
tags_from: ["category"]
```

Then:

```bash
python -m ollama_evaluator.cli convert hf \
  --hf-ref "squad:plain_text:validation" \
  --field-map squad-fields.yaml \
  --name squad-small \
  --limit 100 \
  --output ./examples/suites
```

The `--limit` sub-samples deterministically (with optional `--seed`).

### Local vs remote dataset mode

Two modes:

- **`local`** (default, recommended). Reads pre-cached JSONL/Parquet from
  disk. Zero network at run-time. Run the `convert` command once; every
  future evaluation uses the cached file.
- **`remote`**. Streams rows from HuggingFace Hub at run-time. Requires
  internet. First access pulls and caches the dataset. Slower first run,
  always up-to-date.

Set globally in the config (`dataset_mode: local`) or per-run with
`--dataset-mode`.

---

## 9. Running remotely via SSH

If your workstation can't run large models (no GPU, not enough RAM), run the
tool on a server with the hardware and drive it over SSH.

### Option A — One-button remote deploy (recommended)

The repo ships with a script that tars the project, pushes it to the remote,
and runs `install.sh` there — all in a single command.

**From Linux/macOS:**
```bash
./scripts/deploy-remote.sh user@host [/target/dir]
```

**From Windows:**
```powershell
.\scripts\deploy-remote.ps1 -Target user@host -RemoteDir /home/you/eval
```

Useful flags:
- `--skip-tests` / `-SkipTests` — skip the remote unit-test run.
- `--skip-ui` / `-SkipUI` — backend only.
- `--serve PORT` / `-ServePort N` — start the server in the background after install.
- `--key PATH` / `-Key PATH` — use a specific SSH key.
- `--no-install` / `-NoInstall` — only sync files, don't run `install.sh`.

Example one-liner to deploy, install, and launch the web UI on port 8765:
```bash
./scripts/deploy-remote.sh --serve 8765 \
  --key ~/.ssh/id_ed25519 \
  user@host /home/user/ollama-evaluator
```

Then browse to `http://<host>:8765/`.

### Option B — Manual steps

If you prefer to understand each step, here is what the script does under the
hood.

#### One-time setup: key-based auth

Replace `user@server` with your values.

```bash
# On your local machine
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_eval -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_eval user@server     # enter password once
```

Verify:
```bash
ssh -i ~/.ssh/id_ed25519_eval user@server "echo OK"
# → OK
```

Windows PowerShell equivalent:
```powershell
# Generate
ssh-keygen -t ed25519 -f "$HOME\.ssh\id_ed25519_eval" -N '""'

# Install the public key (PuTTY's plink works well here)
# After the password prompt, future commands need no password.
$pub = Get-Content "$HOME\.ssh\id_ed25519_eval.pub" -Raw
ssh user@server "umask 077; mkdir -p ~/.ssh; echo '$($pub.Trim())' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
```

### Deploying the project

```bash
# Build a clean tarball (excludes caches, node_modules, dist)
tar --exclude='__pycache__' --exclude='*.egg-info' --exclude='.venv' \
    --exclude='node_modules' --exclude='dist' --exclude='.pytest_cache' \
    --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.hypothesis' \
    -czf /tmp/eval.tgz backend ui shared examples README.md

scp -i ~/.ssh/id_ed25519_eval /tmp/eval.tgz user@server:/tmp/
ssh -i ~/.ssh/id_ed25519_eval user@server \
  "mkdir -p ~/workspaces/AI-Model-Evaluation && \
   cd ~/workspaces/AI-Model-Evaluation && \
   tar -xzf /tmp/eval.tgz"
```

### Installing on the remote

```bash
ssh -i ~/.ssh/id_ed25519_eval user@server bash -s <<'EOF'
set -e
cd ~/workspaces/AI-Model-Evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e "backend[dev]"
cd ui
npm install
npm run build
EOF
```

### Running commands remotely

```bash
# Run tests
ssh -i ~/.ssh/id_ed25519_eval user@server \
  "cd ~/workspaces/AI-Model-Evaluation && source .venv/bin/activate && \
   cd backend && python -m pytest -q"

# Trigger an evaluation
ssh -i ~/.ssh/id_ed25519_eval user@server \
  "cd ~/workspaces/AI-Model-Evaluation && source .venv/bin/activate && \
   python -m ollama_evaluator.cli --config examples/config.qwen.yaml run"
```

### Serving the UI so you can use it in a browser

Start the server on the remote and expose it to your LAN:

```bash
ssh -i ~/.ssh/id_ed25519_eval user@server \
  "cd ~/workspaces/AI-Model-Evaluation && source .venv/bin/activate && \
   OLLAMA_EVAL_UI_DIR=\$PWD/ui/dist \
   nohup python -m ollama_evaluator.cli --config examples/config.qwen.yaml \
     serve --host 0.0.0.0 --port 8765 > serve.log 2>&1 &"
```

The `nohup ... &` keeps it running after you log out. Stop it with:

```bash
ssh user@server "pkill -f 'ollama_evaluator.cli.*serve'"
```

Then open `http://<server-ip>:8765/` in your browser.

### Syncing code changes

Rather than re-uploading the full tarball each time, push only what changed:

```bash
# Single file
scp -i ~/.ssh/id_ed25519_eval backend/src/ollama_evaluator/api/app.py \
  user@server:~/workspaces/AI-Model-Evaluation/backend/src/ollama_evaluator/api/app.py

# UI changed? Rebuild on the remote
ssh -i ~/.ssh/id_ed25519_eval user@server \
  "cd ~/workspaces/AI-Model-Evaluation/ui && npm run build"
```

---

## 10. Reading the reports

Every Run produces three artifacts under `<output_dir>/<run_id>/`:

- `report.json` — machine-readable. Every field from every test case.
- `report.md` — human-readable Markdown tables.
- `../history.db` — SQLite database indexing every past Run.

### The Markdown report

Open `examples/runs/<run_id>/report.md`:

```markdown
# Run report: f3c9a1...

- **Status**: completed
- **Backend version**: 0.1.0
- **Ollama version**: 0.20.3
- **Started at**: 2026-05-10T06:04:35+00:00
- **Ended at**: 2026-05-10T06:05:49+00:00

## Models
- **qwen3.6:27b** (digest=sha256:a50..., parameter_size=27.8B)

## Suites
- **reasoning-basics**

## Per-model results
| Model | Passed | Failed | Mean tokens/s | Mean total ms |
|---|---|---|---|---|
| qwen3.6:27b | 2 | 1 | 11.52 | 24666.00 |

## Per-suite results
...

## Error summary
None.
```

### The JSON report

Each test case is a record like:

```json
{
  "model": "qwen3.6:27b",
  "suite": "reasoning-basics",
  "test_case_id": "arithmetic-simple",
  "repetition": 1,
  "status": "pass",
  "response": "4",
  "error_message": null,
  "performance": {
    "ttft_ms": 278.9,
    "total_ms": 14066.0,
    "prompt_tokens": 15,
    "response_tokens": 162,
    "tokens_per_second": 11.52
  },
  "metrics": [
    {"name": "regex-match", "score": 1.0, "passed": true, "threshold": 1.0}
  ]
}
```

### Field meanings

- **status** — `pass`, `fail`, `error`, or `timeout`.
  - `pass` / `fail`: the metrics ran and scored; result is just the score.
  - `error`: something broke (network, HTTP 5xx, retry exhausted).
  - `timeout`: Ollama took longer than `ollama_timeout_s`.
- **ttft_ms** — milliseconds to first streamed token. A latency proxy.
- **total_ms** — wall-clock time for the full response.
- **response_tokens** — tokens the model emitted.
- **tokens_per_second** — derived: `response_tokens / (total_ms / 1000)`.
  Higher is faster.

### What counts as "passed"

A test case has `status = pass` **if and only if every metric reported
`passed = true`**. A single failing metric turns the whole case into `fail`.
Metrics that raised during scoring (e.g. malformed regex) are recorded with
`error` set, but do not fail the case — other metrics still apply.

### Comparing runs

```bash
python -m ollama_evaluator.cli --config examples/config.qwen.yaml \
  compare <old-run-id> <new-run-id>
```

Output is a JSON `ComparisonReport` with:

- **metric_diffs** — per `(model, metric)` present in both: mean score A, mean
  score B, signed difference.
- **performance_diffs** — per model: mean tokens/sec A/B, mean total_ms A/B,
  and the differences.

In the UI, go to `/history`, check two rows, click *Compare*.

---

## 11. Troubleshooting

### "ollama_unreachable"

The backend can't reach Ollama at the configured `ollama_base_url`.

- Is Ollama running? `curl http://localhost:11434/api/version`
- Is your config pointing at the right host? For remote Ollama, set
  `ollama_base_url: http://<host>:11434`.
- Firewall blocking 11434?

### "model_not_found"

The requested model isn't on the Ollama server.

```bash
# Check what's available
ollama list

# Pull the missing one
ollama pull qwen3.6:27b

# Or set `pull_missing_models: true` in your config
```

### All responses are empty

You're probably using a reasoning model with a small `max_tokens`. Reasoning
models like Qwen emit a "thinking" trace before the final answer. Increase
`max_tokens` to 512 or more in the suite's `defaults:`.

### `pytest` reports 2 schema failures

Run:
```bash
python backend/scripts/regen_schemas.py
```

This usually means the Pydantic version changed and the committed OpenAPI /
JSON Schema files need refreshing.

### "ValueError: no active connection" on `/api/runs`

You're running an old version. This was fixed by moving the store open into
the FastAPI lifespan. Pull the latest code and re-install.

### The UI loads but no models show up

- Is the browser hitting the right backend? Check the URL bar.
- Does `curl http://<host>:8765/api/models` return a list? If yes, it's a UI
  bug; open the browser's dev console to look for errors.

### Port already in use

```bash
# Who's on 8765?
ss -tlnp | grep 8765

# Kill the old server
pkill -f 'ollama_evaluator.cli.*serve'

# Or use a different port
python -m ollama_evaluator.cli --config X run --port 8766
```

### Runs are slow

- First run of a model is always slow because Ollama loads weights into
  VRAM. Subsequent runs are much faster.
- Reasoning models spend most of their time thinking, not generating. Check
  `tokens_per_second` in the report — if it's close to the raw model speed,
  you're not bottlenecked.
- Increase `concurrency` cautiously. Higher concurrency reduces wall-clock
  time but needs more VRAM and can cause out-of-memory errors. A safe
  starting point is 1–2 for 27B+ models, 4–8 for 8B.

### "Suite validation error"

The tool tells you the exact file, test-case id, and field. Fix the YAML and
run `validate-suite` to confirm.

### Reports are piling up

The `examples/runs/` directory grows one subdirectory per run. To clean up:

```bash
# List runs from the CLI
python -m ollama_evaluator.cli --config examples/config.qwen.yaml \
  compare --help    # CLI doesn't have a prune command yet
# For now, delete manually:
rm -rf examples/runs/<run-id>
```

Or use the REST API:
```bash
curl -X DELETE http://localhost:8765/api/runs/<run-id>
```

---

## 12. Command reference

```
Usage:
  python -m ollama_evaluator.cli [GLOBAL OPTIONS] COMMAND [ARGS]

Global options:
  --config PATH                 YAML or JSON config file
  --output-dir PATH             Override config.output_dir
  --log-level debug|info|warn|error
  --dataset-mode local|remote
  --hf-cache-dir PATH

Commands:
  list-models                   List models available on the Ollama server
  run                           Execute a Run using --config
  compare RUN_A RUN_B           Compare two historical Runs
  validate-suite FILE           Validate a suite file offline (no Ollama)
  serve [--host H] [--port P]   Start the HTTP + WebSocket + UI server
  convert mmlu|hellaswag|...    Materialise a public benchmark to YAML
  convert hf --hf-ref REF --field-map FILE --output DIR --name N [--limit N]

HTTP API endpoints (when `serve` is running):

  GET    /api/health                         Liveness
  GET    /api/models                         List models
  GET    /api/suites                         List suite names
  GET    /api/suites/{name}                  Full suite content
  POST   /api/runs                           Submit a Run (body = RunConfig)
  GET    /api/runs                           List persisted runs
  GET    /api/runs/{id}                      Run report
  GET    /api/runs/{id}/report.md            Markdown report
  DELETE /api/runs/{id}                      Delete a run
  POST   /api/runs/{id}/cancel               Cancel an in-progress Run
  GET    /api/compare?a=ID&b=ID              Comparison report
  GET    /openapi.json                       OpenAPI 3.1 document

WebSocket:
  WS     /api/runs/{id}/events               Live event stream
```

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Ollama** | Local LLM runtime. Listens on `http://localhost:11434`. |
| **LLM** | Large Language Model. |
| **TTFT** | Time-To-First-Token. The latency before any output arrives. |
| **Tokens/s** | Throughput in tokens per second. Higher = faster generation. |
| **System prompt** | Instructions prepended to the request that tell the model how to behave. |
| **Repetition** | A single execution of `(model, test case)`. Multiple repetitions measure variance. |
| **Concurrency** | How many generate calls run in parallel. |
| **Preflight** | Checks the tool does before starting a Run: reachable Ollama, model present, suites valid. |
| **Run id** | Unique identifier, like `f3c9a1b42e8b4...`. Used as the primary key in the history database. |
| **History store** | Local SQLite database at `<output_dir>/history.db`. Indexes every Run you've executed. |
| **Backend** | The Python program that orchestrates Runs and serves the API. |
| **UI** | The React web app. |
| **REST / WebSocket** | Two protocols the backend speaks. REST for request/response; WebSocket for the live event stream. |
| **OpenAPI** | Machine-readable API description at `/openapi.json`. Used to generate the UI's typed client. |

---

## Need help?

- For unexpected failures, run with `--log-level debug` and share the output.
- For missing features, open an issue or PR.
- Requirements, design, and implementation tasks live under
  `.kiro/specs/ollama-model-evaluator/` — a good place to see *why* the tool
  works the way it does.
