# Implementation Plan: Ollama Model Evaluator

## Overview

This plan breaks the Ollama Model Evaluator into incremental, end-to-end coding tasks. Work starts with the Python backend (data models → suite loading → Ollama client → metrics → adapters → runner → history → API) and then adds the React UI on top of the stable REST + WebSocket surface. Property-based test tasks are placed next to the implementation they verify, one sub-task per design Correctness Property (44 total). Each task links to specific sub-requirements from `requirements.md`; property tasks additionally cite the design property number.

Conventions:
- Sub-tasks marked with `*` (e.g. `- [ ]* 2.2`) are optional test tasks. Do not implement them unless explicitly requested.
- Sub-tasks without `*` are required implementation tasks.
- Each property-based test is implemented once, named `test_property_{n}_{short}.py` on the backend or `property_{n}_{short}.test.ts` on the UI, with `max_examples ≥ 100` (Hypothesis) / `numRuns ≥ 100` (fast-check) per the testing strategy in `design.md`.
- All code examples are Python 3.11+ for the backend and TypeScript/React for the UI, as fixed by the design.

## Tasks

- [x] 1. Scaffold the repository skeleton
  - Create the top-level layout defined in `design.md` §Repository layout: `backend/`, `ui/`, `shared/`.
  - In `backend/`, create `pyproject.toml` targeting Python 3.11+ with dependencies `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `httpx`, `ruamel.yaml`, `aiosqlite`, `typer`, `jsonschema`, `datasets` (HuggingFace), `hypothesis`, `pytest`, `pytest-asyncio`, `pytest-cov`, and dev tools (`ruff`, `mypy`).
  - Create the `src/ollama_evaluator/` package with empty `__init__.py` exposing `__version__ = "0.1.0"` plus empty subpackages `suites/`, `ollama/`, `metrics/`, `runner/`, `history/`, `api/`.
  - Create `backend/tests/{unit,property,integration}/` with `conftest.py` stubs and pytest markers `property` and `integration` registered in `pyproject.toml`.
  - In `ui/`, scaffold a Vite + React + TypeScript project with Vitest and fast-check installed; add `src/{api,routes,stream,state}/` empty directories.
  - In `shared/`, create placeholder `openapi.yaml`, `evaluation-suite.schema.json`, `run-report.schema.json` files (contents generated later).
  - _Requirements: 10.1, 13.1, 15.1_

- [x] 2. Define core Pydantic data models
  - [x] 2.1 Implement `TestCase`, `EvaluationSuite`, `GenerationDefaults`, `MetricConfig` in `suites/models.py`
    - Enforce unique test-case ids within a suite via a Pydantic validator; non-empty `prompt`; non-empty `metrics`.
    - Use `model_config = ConfigDict(extra="forbid")` on every model.
    - _Requirements: 3.3, 3.4_

  - [x] 2.2 Implement `ConfigFile` and `RunConfig` in `config.py`
    - Include `dataset_mode: Literal["local", "remote"] = "local"` and `hf_cache_dir: Path | None` on `ConfigFile`.
    - Include `repetitions`, `concurrency`, `pull_missing_models`, `retry_max_attempts`, `judge_model`, `tag_filter` on `RunConfig` with the defaults in `design.md` §Data Models.
    - _Requirements: 1.1, 1.4, 2.2, 2.4, 3.6, 5.2, 5.5, 10.4, 10.5, 10.6, 11.1, 17.3_

  - [x] 2.3 Implement `PerformanceMetrics`, `MetricResult`, `TestCaseResult`, `ModelAggregate`, `RunReport`, `ModelInfo`, `RunSummary`, `ErrorSummaryEntry` in `models.py`
    - `MetricResult.error` is `str | None`; `TestCaseResult.status` is the 4-value literal from the design.
    - _Requirements: 2.5, 6.1, 6.2, 6.3, 6.4, 6.5, 7.3, 7.4, 7.5, 7.6, 8.2, 8.4, 11.3_

  - [x] 2.4 Implement the `RunEvent` discriminated union
    - Define `BaseRunEvent`, `RunStartedEvent`, `RunProgressEvent`, `TestCaseCompletedEvent`, `RunCompletedEvent`, `RunAbortedEvent`, `RunFailedEvent` and the `Annotated[... , Field(discriminator="type")]` union.
    - _Requirements: 14.2, 14.3, 14.4, 14.5_

  - [x] 2.5 Write property test for `RunReport` round-trip
    - **Property 18: Run_Report round-trip** — `RunReport.model_validate_json(r.model_dump_json()) == r`.
    - Use Hypothesis strategies in `tests/property/generators.py` that build valid `RunReport`s including all status values and both missing and populated performance fields.
    - **Validates: Requirement 8.5**

- [x] 3. Implement suite loader, writer, and discovery
  - [x] 3.1 Implement `suites/loader.py` with `load_suite(path)`, `load_suite_from_string(text, fmt)`, and `discover_suites(dir)`
    - Detect `.yaml`/`.yml` via `ruamel.yaml` (round-trip mode), `.json` via `json`.
    - Wrap Pydantic `ValidationError` in `SuiteValidationError(path, test_case_id, missing_field, message, line)`.
    - _Requirements: 3.1, 3.2, 3.5, 4.1, 4.4_

  - [x] 3.2 Implement `suites/writer.py` with `dump_suite(suite, fmt)`
    - Canonicalise output: sorted keys, 2-space indent, block-style YAML, no comments, so round-trip equivalence is model-level not byte-level.
    - _Requirements: 4.2_

  - [x] 3.3 Write property test for Evaluation_Suite round-trip
    - **Property 1: Evaluation_Suite round-trip** — `load_suite_from_string(dump_suite(s, f), f) == s` for every valid `s` and `f ∈ {yaml, json}`.
    - **Validates: Requirements 3.2, 3.4, 4.1, 4.2, 4.3**

  - [x] 3.4 Write property test for Evaluation_Suite validation
    - **Property 2: Evaluation_Suite validation** — `load_suite` succeeds iff structural invariants hold and reports the offending test case id and field otherwise.
    - **Validates: Requirements 3.3, 3.5**

  - [x] 3.5 Write property test for suite discovery
    - **Property 3: Suite discovery** — `discover_suites(dir)` returns exactly the suites loaded from every `.yaml|.yml|.json` file in `dir`.
    - **Validates: Requirement 3.1**

- [x] 4. Implement the Ollama HTTP client
  - [x] 4.1 Implement `ollama/types.py` with `OllamaModelInfo`, `GenerateOptions`, `GenerateChunk`, `PullProgress` models matching Ollama's API shapes
    - _Requirements: 2.5, 6.3_

  - [x] 4.2 Implement `ollama/client.py` with `OllamaClient`
    - Methods: `version()`, `list_models()`, `pull_model(name)`, `generate(model, prompt, system, options)` as async-iterator of streamed chunks.
    - Configure `httpx.AsyncClient(base_url=..., timeout=httpx.Timeout(total=ollama_timeout_s))`.
    - Measure time-to-first-token as `monotonic()` delta between request dispatch and first streamed chunk.
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 2.1, 2.4, 2.5, 6.1_

  - [x] 4.3 Write unit tests for OllamaClient using a `FakeOllamaServer` ASGI fixture
    - Cover version, list-models, streamed generate with and without optional `prompt_eval_count`, timeout path.
    - _Requirements: 1.2, 1.4, 1.5, 2.1, 6.1, 6.3, 6.5_

- [x] 5. Implement the metric framework and built-in metrics
  - [x] 5.1 Implement `metrics/base.py` with the `Metric` protocol, `MetricResult`, `MetricContext`, and a `metrics/__init__.py` registry keyed by metric name
    - _Requirements: 7.3, 7.4_

  - [x] 5.2 Implement `metrics/builtin.py` with `exact-match`, `regex-match`, `contains`, `json-schema-valid`, `length-range`
    - Use the pass/score table in `design.md` §Metric framework.
    - _Requirements: 7.1, 7.3, 7.4_

  - [x] 5.3 Implement `metrics/judge.py` with the `llm-as-judge` metric
    - Parse `"Score: X/Y"` where `0 ≤ X ≤ Y ≤ 100`; on malformed judge output set `MetricResult.error` and `passed=False`.
    - Call the `OllamaClient` using the configured `judge_model`.
    - _Requirements: 7.2, 7.4, 7.5_

  - [x] 5.4 Implement `response-capture` metric in `metrics/builtin.py`
    - Always `passed=True`, `score=0.0`; store the raw response in `details.response`.
    - Reserve the metric name `humaneval-exec` in the registry with a "not implemented" guard.
    - _Requirements: 17.9_

  - [x] 5.5 Write property test for built-in metric correctness
    - **Property 12: Built-in metric correctness** — for every built-in `m ∈ {exact-match, regex-match, contains, json-schema-valid, length-range}`, `m.score` returns a numeric `score` and a boolean `passed` that obey the rules from the design.
    - **Validates: Requirements 7.1, 7.3, 7.4**

  - [x] 5.6 Write property test for llm-as-judge score normalisation
    - **Property 13: llm-as-judge score normalisation** — well-formed `"Score: X/Y"` outputs normalise to `X/Y`; malformed outputs set `MetricResult.error` and `passed=False`.
    - **Validates: Requirement 7.2**

- [x] 6. Implement the aggregation module
  - [x] 6.1 Implement `runner/aggregate.py` with per-metric mean/stddev and per-model `ModelAggregate` construction
    - Use `statistics.fmean` and `statistics.pstdev`.
    - _Requirements: 7.6, 8.2_

  - [x] 6.2 Write property test for repetition aggregates
    - **Property 15: Repetition aggregates** — `mean == fmean([s_1 … s_R])` and `stddev == pstdev([s_1 … s_R])` for every triple.
    - **Validates: Requirement 7.6**

  - [x] 6.3 Write property test for metric error isolation
    - **Property 14: Metric error isolation** — `TestCaseResult.metrics` preserves input order; metrics in the error subset have `error != None`, others are scored normally.
    - **Validates: Requirement 7.5**

- [x] 7. Implement public-benchmark adapters
  - [x] 7.1 Implement `suites/adapter_base.py` with `AdapterOptions`, `HFRef`, and the adapter protocol (`ADAPTER_NAME`, `DEFAULT_HF_REF`, `rows_to_suite`, `materialise`)
    - _Requirements: 17.1, 17.3, 17.4_

  - [x] 7.2 Implement `suites/mmlu.py`
    - Prompt template, answer extraction, `regex-match` metric with `expected_output = answer_letter`, `id = f"mmlu/{subject}/{row_index}"`, subject tags per `design.md` §Dataset sources.
    - _Requirements: 17.1, 17.10_

  - [x] 7.3 Implement `suites/hellaswag.py`
    - Per the design table: prompt template, `expected_output = "ABCD"[int(label)]`, `id = f"hellaswag/{ind}"`.
    - _Requirements: 17.1, 17.10_

  - [x] 7.4 Implement `suites/truthfulqa.py`
    - MC1 form only in v1; `expected_output` = letter of the index where `mc1_targets.labels == 1`; LLM-as-judge variant documented but not default.
    - _Requirements: 17.1, 17.10_

  - [x] 7.5 Implement `suites/gsm8k.py`
    - Regex `(?i)(?:final answer:\s*|####\s*)(-?\d[\d,]*(?:\.\d+)?)` with numeric-equality post-processing; `expected_output` from the gold `#### N` block.
    - _Requirements: 17.1, 17.10_

  - [x] 7.6 Implement `suites/humaneval.py`
    - Use `response-capture` metric only; store `reference_data = {"test": test, "entry_point": entry_point}` so an external grader can score later.
    - _Requirements: 17.1, 17.9_

  - [x] 7.7 Write property test that every adapter emits a valid Evaluation_Suite
    - **Property 42: Adapter output is a valid Evaluation_Suite** — for `a ∈ {mmlu, hellaswag, truthfulqa, gsm8k, humaneval}` and any well-formed rows, `a.rows_to_suite(R, o)` satisfies Property 2 validation and Property 1 round-trip in both `yaml` and `json`.
    - **Validates: Requirements 3.3, 4.1, 4.3, 17.1, 17.2, 17.8**

- [x] 8. Implement the generic HuggingFace loader
  - [x] 8.1 Implement `suites/huggingface.py` with `HFRef`, `HFFieldMap`, `HFSuiteSpec`, `stream_rows`, and `materialise_hf`
    - Field-map resolution supports dotted paths and bracketed list indices (`answers.text[0]`).
    - Raise `FieldMapError(row_index, field, reason)` on missing, `None`, or wrong-typed declared fields.
    - `stream_rows(ref, mode, cache_dir)` reads local JSONL/Parquet in `local` mode and delegates to the `datasets` library in `remote` mode.
    - `materialise_hf` is pure given `(spec, rows)` so tests exercise it without network I/O.
    - Honour optional `limit` and `seed` for deterministic sub-sampling.
    - _Requirements: 17.2, 17.3, 17.4, 17.7_

  - [x] 8.2 Wire per-adapter `materialise(mode, opts, cache_dir)` to dispatch on mode using `stream_rows` in `remote` mode and a local reader in `local` mode
    - Ensures the same `rows_to_suite` transform is used for both modes (Property 43 consistency invariant).
    - _Requirements: 17.3, 17.4, 17.10_

  - [x] 8.3 Write property test for local/remote mode equivalence
    - **Property 43: Local/remote mode equivalence** — for every adapter/spec `a` and populated cache, `a.materialise("local", ...)` and `a.materialise("remote", ...)` produce suites with equal name, equal ordered ids, and equal per-field `TestCase` contents.
    - Use a `FakeHFHub` fixture to drive `remote` mode and pre-converted fixture files for `local` mode.
    - **Validates: Requirement 17.10**

  - [x] 8.4 Write property test for HuggingFace field-map totality and injectivity
    - **Property 44: HuggingFace field-map totality and injectivity** — for any `HFSuiteSpec` whose field map is applicable to every row, `materialise_hf` produces `min(|R|, limit or |R|)` `TestCase`s, is deterministic under a fixed `seed`, and raises `FieldMapError` without a partial suite on any unresolved field.
    - **Validates: Requirements 3.3, 3.5, 17.2, 17.7**

- [x] 9. Implement tag and name filtering at run-expansion time
  - [x] 9.1 Implement `runner/selection.py` that takes `(suites, RunConfig)` and returns the list of `(model, TestCase, repetition)` planned executions
    - Apply `config.suites` name filter and `config.tag_filter` set intersection per the design.
    - _Requirements: 3.6, 5.1, 5.2_

  - [x] 9.2 Write property test for tag and name filtering
    - **Property 4: Tag and name filtering** — selected test cases equal `{tc | s.name ∈ config.suites and (tag_filter == [] or tc.tags ∩ tag_filter ≠ ∅)}`.
    - **Validates: Requirement 3.6**

  - [x] 9.3 Write property test for execution count and coverage
    - **Property 6: Execution count and coverage** — a successful Run dispatches exactly `|M| · |K| · R` generate calls, with exactly one call per tuple.
    - **Validates: Requirement 5.1**

  - [x] 9.4 Write property test for generation parameter resolution
    - **Property 7: Generation parameter resolution** — `GenerateOptions.field == (tc.field if tc.field is not None else defaults.field)` for every generation field.
    - **Validates: Requirements 5.3, 5.4**

- [x] 10. Checkpoint — models, loaders, metrics, adapters
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement the per-run event bus and run state
  - [x] 11.1 Implement `runner/run_state.py` with `RunState`, `RunEventBus`, and `append_event`
    - Append-only in-memory list plus `asyncio.Condition`; appends persist to the SQLite `run_events` table in the same transaction as state changes (store integration wired in task 13).
    - _Requirements: 14.5, 14.6, 14.7_

  - [x] 11.2 Implement the 2-second `run-progress` ticker as an `asyncio.create_task`
    - Cancelled on terminal transition; never emits after the terminal event.
    - _Requirements: 14.4_

- [x] 12. Implement the scheduler and test-case runner
  - [x] 12.1 Implement `runner/scheduler.py` with `RunScheduler.execute()`
    - Expand `(model, test_case, repetition)` into the pending queue.
    - Gate dispatch with `asyncio.Semaphore(concurrency)`.
    - For each execution: stream-generate from Ollama, measure `ttft_ms`, `total_ms`, `prompt_tokens`, `response_tokens`, compute `tokens_per_second` (or `None` when inputs are missing/zero).
    - Score every configured metric; wrap each metric call in `try/except` so a metric error becomes `MetricResult.error` without affecting other metrics.
    - Append `test-case-completed` to the bus and persist the `TestCaseResult`.
    - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.3, 7.5_

  - [x] 12.2 Implement the retry policy
    - Retry `httpx.ConnectError`, `httpx.ReadError`, HTTP 502/503/504 up to `retry_max_attempts` times with `1s · 2^k ± 20%` jitter.
    - Do not retry timeouts (record `status="timeout"`) or HTTP 4xx (record `status="error"`).
    - _Requirements: 1.5, 5.6, 11.1, 11.2_

  - [x] 12.3 Implement cooperative cancellation and signal handling
    - `RunState.cancel_requested` set by `POST /api/runs/{id}/cancel` or SIGINT/SIGTERM stops dequeue; in-flight awaited up to 30s then marked `error` with message `"cancelled"`.
    - Install signal handlers only in the `serve` CLI subcommand entry point.
    - _Requirements: 11.4_

  - [x] 12.4 Implement preflight
    - Verify Ollama reachable (`ollama_unreachable` on failure → `run-failed`).
    - Verify all requested models present; if `pull_missing_models=true`, call `OllamaClient.pull_model` for each missing one; else fail with `model_not_found` naming the missing models.
    - Materialise every remote-mode adapter-backed suite before emitting `run-started`; on failure emit `run-failed` with `error_code=dataset_fetch_failed` (or `field_map_invalid` for HF field-map issues).
    - Record per-evaluated-model `tag`, `digest`, and `parameter_size` on the RunReport.
    - _Requirements: 1.2, 1.3, 2.3, 2.4, 2.5, 17.5, 17.6, 17.7_

  - [x] 12.5 Write property test for concurrency bound
    - **Property 8: Concurrency bound** — max in-flight generate calls `≤ concurrency`.
    - **Validates: Requirement 5.5**

  - [x] 12.6 Write property test for scheduler failure isolation
    - **Property 9: Scheduler failure isolation** — failing subset `F` appears as `timeout|error`; rest as `pass|fail`; total results == planned count.
    - **Validates: Requirements 1.5, 5.6**

  - [x] 12.7 Write property test for tokens-per-second arithmetic
    - **Property 10: Tokens-per-second arithmetic** — `tokens_per_second == response_tokens / (total_ms / 1000)` or `None` when inputs are missing/zero.
    - **Validates: Requirement 6.4**

  - [x] 12.8 Write property test for optional performance fields
    - **Property 11: Optional performance fields** — missing Ollama metadata yields `None` fields without failing the test case.
    - **Validates: Requirements 6.3, 6.5**

  - [x] 12.9 Write property test for retry limit and terminal error
    - **Property 21: Retry limit and terminal error** — total attempts `== min(n + 1, retry_max_attempts + 1)`; last error message preserved; run continues past the failing execution.
    - **Validates: Requirements 11.1, 11.2**

  - [x] 12.10 Write property test for missing-model preflight
    - **Property 5: Missing-model preflight** — with `pull_missing_models=false`, run aborts iff `R \ A ≠ ∅`; `run-failed` event lists exactly `R \ A`.
    - **Validates: Requirement 2.3**

- [x] 13. Implement the History_Store and run lifecycle persistence
  - [x] 13.1 Implement `history/schema.sql` with the five tables from `design.md` §History_Store
    - Enable `PRAGMA journal_mode=WAL` on connection open.
    - _Requirements: 12.1_

  - [x] 13.2 Implement `history/store.py` with `HistoryStore` using `aiosqlite`
    - `create_run`, `update_run_status`, `append_event`, `list_events`, `write_test_case_result`, `write_report`, `get_run`, `list_runs(filter)`, `delete_run`.
    - Wrap per-row Pydantic validation in `try/except` so corrupted rows are skipped with a warning log naming the `run_id`.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 14.5_

  - [x] 13.3 Ensure the scheduler calls `write_report` and `append_event(terminal_event)` before `update_run_status(terminal_status)`
    - _Requirements: 12.2_

  - [x] 13.4 Write property test for history persistence across restart
    - **Property 23: History persistence across restart** — reopening the store returns `RunReport` equal to the one written.
    - **Validates: Requirement 12.1**

  - [x] 13.5 Write property test for terminal-event-before-status ordering
    - **Property 24: Terminal-event-before-status ordering** — the order `append_event(terminal) → write_report → update_run_status(terminal)` is preserved.
    - **Validates: Requirement 12.2**

  - [x] 13.6 Write property test for unique run identifiers
    - **Property 25: Unique run identifiers** — `n` sequential `create_run()` calls return pairwise-distinct ids.
    - **Validates: Requirement 12.3**

  - [x] 13.7 Write property test for history filter semantics
    - **Property 26: History filter semantics** — `list_runs(filter)` returns exactly the runs satisfying every non-`None` filter field.
    - **Validates: Requirement 12.4**

  - [x] 13.8 Write property test for history delete
    - **Property 27: History delete** — after `delete_run(id)`, `get_run` returns not-found and `list_runs` omits the id; others unaffected.
    - **Validates: Requirement 12.5**

  - [x] 13.9 Write property test for history skipping corrupted records
    - **Property 28: History skips corrupted records** — `list_runs` returns `N − |C|` records and logs a warning per corrupted row.
    - **Validates: Requirement 12.6**

- [x] 14. Implement Run_Report writing (JSON + Markdown)
  - [x] 14.1 Implement `runner/reports.py` with `write_artifacts(run_id, report, output_dir)`
    - Write `runs/{run_id}/report.json` and `runs/{run_id}/report.md` atomically (write-temp-then-rename).
    - Include in the report: `run_id`, `backend_version`, `ollama_version`, `started_at`, `ended_at`, `config`, `models`, `results`, `aggregates`, `error_summary`.
    - Render Markdown with per-Model and per-Evaluation_Suite tables containing column headers `Model`, `Passed`, `Failed`, `Mean tokens/s`, `Mean total ms`.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 11.3_

  - [x] 14.2 Write property test for Run_Report completeness
    - **Property 16: Run_Report completeness** — every successful Run produces a report with all required fields including per-model `tag`, `digest`, `parameter_size` and per-`(model, test_case_id, repetition)` results.
    - **Validates: Requirements 2.5, 8.2, 8.4**

  - [x] 14.3 Write property test for Markdown report contents
    - **Property 17: Markdown report contents** — Markdown rendering contains every model name, every suite name, and the listed headers.
    - **Validates: Requirement 8.3**

  - [x] 14.4 Write property test for error summary completeness
    - **Property 22: Error summary completeness** — `error_summary` contains exactly one entry per `error|timeout` result with `model`, `test_case_id`, `repetition`, `error_message`.
    - **Validates: Requirement 11.3**

- [x] 15. Implement the compare module
  - [x] 15.1 Implement `compare.py` with `compare(a: RunReport, b: RunReport) -> ComparisonReport`
    - Key `metric_diffs` by `(model, metric)` present in both; `performance_diffs` by `model` present in both.
    - Raise `NoCommonDimensionsError` iff both keyspaces are empty.
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 15.2 Write property test for comparison over intersection
    - **Property 19: Comparison over intersection** — `metric_diffs` keyed exactly by the intersection with `diff == mean_b - mean_a`; `NoCommonDimensionsError` iff both intersections empty.
    - **Validates: Requirements 9.2, 9.3, 9.4**

- [x] 16. Checkpoint — runner, history, reports, compare
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Implement the FastAPI REST surface
  - [x] 17.1 Implement `api/errors.py` with the `error_code` enum and the `{error_code, message, field}` envelope
    - Enumerate `ollama_unreachable`, `model_not_found`, `suite_not_found`, `run_not_found`, `validation_failed`, `suite_invalid`, `no_common_dimensions`, `dataset_fetch_failed`, `field_map_invalid`, `run_timeout`, `run_error`, `metric_error`.
    - _Requirements: 13.5, 13.6_

  - [x] 17.2 Implement `api/app.py` and `api/rest.py` with all endpoints from `design.md` §REST API
    - `GET /api/health`, `GET /api/models`, `GET /api/suites`, `GET /api/suites/{name}`, `POST /api/runs`, `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/report.md`, `DELETE /api/runs/{id}`, `POST /api/runs/{id}/cancel`, `GET /api/compare`, `GET /openapi.json`.
    - Install a global exception handler that maps Pydantic validation errors to `400 {error_code: "validation_failed", field, message}` with `field` = dotted path of the first failing field in document order.
    - _Requirements: 2.1, 9.1, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 16.1, 16.3, 16.5_

  - [x] 17.3 Wire `POST /api/runs` to create a `RunState`, persist it as `pending`, and hand it to a single-run scheduler worker
    - Only one run is `running` at a time; subsequent runs queue as `pending`.
    - _Requirements: 13.3_

  - [x] 17.4 Generate `shared/openapi.yaml` and `shared/*.schema.json` from the Pydantic models
    - Add a `scripts/regen_schemas.py` that writes the files; wire it into a CI verification that asserts the committed copies match the generated ones.
    - _Requirements: 13.7_

  - [x] 17.5 Write property test for 404 error envelope
    - **Property 29: 404 error envelope** — every endpoint that references a missing `run_id`, `suite_name`, or `model_name` returns `404 {error_code ∈ {run_not_found, suite_not_found, model_not_found}, message}`.
    - **Validates: Requirement 13.5**

  - [x] 17.6 Write property test for 400 field identification
    - **Property 30: 400 field identification** — invalid request bodies return `400 {error_code: "validation_failed", field, message}` with `field` = dotted path of first failing field.
    - **Validates: Requirement 13.6**

- [x] 18. Implement the WebSocket event stream
  - [x] 18.1 Implement `api/events_ws.py` at `GET /api/runs/{run_id}/events`
    - On subscribe, validate `run_id` (close with code 4404 and reason `"run_not_found"` if missing).
    - Replay persisted events in `seq` order, then forward newly-appended events.
    - Send the terminal event and close with code 1000.
    - Send WebSocket ping every 15s; missing pong for 30s closes the connection.
    - _Requirements: 13.5, 14.1, 14.5, 14.6, 14.7_

  - [x] 18.2 Emit `run-started`, `test-case-completed` (with `TestCaseResult`), and `run-progress` events per the design
    - _Requirements: 14.2, 14.3, 14.4_

  - [x] 18.3 Write property test for event log bookends
    - **Property 31: Event log bookends** — the event log begins with exactly one `run-started` and ends with exactly one terminal event; no events follow.
    - **Validates: Requirements 14.2, 14.5**

  - [x] 18.4 Write property test for test-case-completed bijection
    - **Property 32: test-case-completed bijection** — `test-case-completed` events correspond bijectively to executed `(model, test_case_id, repetition)` tuples and carry the full `TestCaseResult`.
    - **Validates: Requirement 14.3**

  - [x] 18.5 Write property test for progress cadence
    - **Property 33: Progress cadence** — consecutive `run-progress` events separated by ≤ `2s + ε`; last progress event strictly precedes the terminal event.
    - **Validates: Requirement 14.4**

  - [x] 18.6 Write property test for multi-subscriber replay and isolation
    - **Property 34: Multi-subscriber replay and isolation** — every subscriber that remains connected through the terminal event receives the canonical sequence in order, with no gaps or duplicates, regardless of other subscribers.
    - **Validates: Requirements 14.6, 14.7**

- [x] 19. Implement the CLI
  - [x] 19.1 Implement `cli.py` with Typer and subcommands `list-models`, `run`, `compare`, `validate-suite`, `serve`
    - Global flags: `--config`, `--output-dir`, `--log-level {debug,info,warn,error}` (default `info`), `--dataset-mode {local,remote}`, `--hf-cache-dir`.
    - `serve` starts Uvicorn and blocks until SIGINT/SIGTERM.
    - Exit code 0 iff every `TestCaseResult.status == pass`; non-zero otherwise; exit code 2 on preflight errors (`ollama_unreachable`, `model_not_found`, `dataset_fetch_failed`, `field_map_invalid`).
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 17.3_

  - [x] 19.2 Implement the `convert` subcommand family
    - `convert mmlu`, `convert hellaswag`, `convert truthfulqa`, `convert gsm8k`, `convert humaneval`, `convert hf --hf-ref --field-map --output`.
    - Each writes standard Evaluation_Suite YAML under `--output` so the files are indistinguishable from hand-authored ones.
    - _Requirements: 17.8_

  - [x] 19.3 Write property test for CLI exit code ↔ failure presence
    - **Property 20: CLI exit code ↔ failure presence** — exit code is 0 iff every test case status is `pass`.
    - **Validates: Requirements 10.2, 10.3**

- [x] 20. Checkpoint — Backend is feature-complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. Scaffold the UI and generate the typed API client
  - [x] 21.1 Configure Vite, React Router, TanStack Query, Zustand in `ui/`
    - Set up a base layout with navigation to `/runs/new`, `/               history`, and a run detail placeholder.
    - _Requirements: 15.1, 15.2_

  - [x] 21.2 Generate the TypeScript API client from `shared/openapi.yaml`
    - Use `openapi-typescript` or equivalent; place generated types in `ui/src/api/types.ts`.
    - Wrap endpoints in a small `apiClient` module that returns typed `Promise`s.
    - _Requirements: 13.7, 15.2_

  - [x] 21.3 Implement the `errorMessages` record mapping `error_code → user-visible message`
    - _Requirements: 15.4_

- [x] 22. Implement the New-Run route
  - [x] 22.1 Implement `ui/src/routes/NewRun.tsx`
    - Fetch models and suites on load via TanStack Query.
    - Form fields: models multi-select, suites multi-select, `repetitions`, `concurrency`, optional `tag_filter`, submit button.
    - On submit, `POST /api/runs`; on success navigate to `/runs/:id`; on `400 validation_failed`, render the returned `message` next to the input bound to `field`.
    - _Requirements: 15.2, 15.3, 15.4, 15.5_

  - [x] 22.2 Write property test for UI form → RunConfig body
    - **Property 35: UI form → RunConfig body** — for every valid selection, `POST /api/runs` body has fields equal to selections with no additional fields.
    - **Validates: Requirement 15.3**

  - [x] 22.3 Write property test for UI 400-error field mapping
    - **Property 36: UI 400-error field mapping** — `{error_code: "validation_failed", field, message}` renders `message` as a sibling of the input bound to `field`.
    - **Validates: Requirement 15.4**

- [x] 23. Implement the event-stream client and RunDetail route
  - [x] 23.1 Implement `ui/src/stream/runEvents.ts` with `RunEventStream`
    - WebSocket connect with reconnect schedule `[1s, 2s, 4s, 8s, 16s]` (max 5 attempts).
    - After 5 failures, switch to polling `GET /api/runs/:id` at 5s intervals and mark the status as `polling`.
    - Expose `onEvent` and `onStatus` callbacks.
    - _Requirements: 15.8, 15.9_

  - [x] 23.2 Implement a pure reducer `runEventState.ts` that folds `RunEvent`s into derived state
    - State includes `status`, `percent_complete`, counters for `{passed, failed, error, timeout}`, per-execution table rows, terminal error message.
    - Must be replay-deterministic: applying the same event sequence twice yields equal state.
    - _Requirements: 15.6, 15.7, 16.3, 16.6_

  - [x] 23.3 Implement `ui/src/routes/RunDetail.tsx`
    - Wire `useRunEvents(runId)` to the reducer; render status, progress percentage, counters, live-updating execution table with metric scores and performance metrics.
    - Show a visible "disconnected" indicator while the stream is reconnecting; show "polling" when in fallback.
    - Cancel button that calls `POST /api/runs/:id/cancel` after confirmation and updates the displayed status.
    - On completed runs, show full `RunReport` (per-test-case inputs, outputs, metric scores, performance metrics) and download links for `runs/{id}/report.json` and `runs/{id}/report.md`.
    - For `error`/`aborted` runs, display the terminal event's error message.
    - _Requirements: 15.5, 15.6, 15.7, 15.8, 15.10, 16.3, 16.5, 16.6_

  - [x] 23.4 Write property test for UI event-stream state reducer
    - **Property 37: UI event-stream state reducer** — for any well-formed event sequence, the reducer fold produces the reference derived state, and this holds under drop-and-replay of any prefix.
    - **Validates: Requirements 15.6, 15.7, 16.3, 16.6**

  - [x] 23.5 Write property test for UI reconnect schedule and polling fallback
    - **Property 38: UI reconnect schedule and polling fallback** — at most 5 reconnects with delays `[1s, 2s, 4s, 8s, 16s]`; after 5 failures, polling at 5s with status `"polling"`.
    - Use fast-check with a fake `WebSocket` and fake timers.
    - **Validates: Requirements 15.8, 15.9**

- [x] 24. Implement the History and Compare routes
  - [x] 24.1 Implement `ui/src/routes/History.tsx`
    - Fetch `GET /api/runs` with filters for `model`, `suite`, `status`, `since`, `until`.
    - Each row shows `started_at`, `status`, model names, suite names, aggregate pass rate.
    - Checkbox selection of exactly two rows enables a "Compare" action that navigates to `/compare?a=…&b=…`.
    - Each row has download links for the run's JSON and Markdown reports.
    - Filter selections are encoded into the URL so they round-trip with the Backend's filter parser.
    - _Requirements: 16.1, 16.2, 16.4, 16.5_

  - [x] 24.2 Implement `ui/src/routes/Compare.tsx`
    - Fetch `GET /api/compare?a=&b=`.
    - Render exactly `|metric_diffs|` rows in the metric table and `|performance_diffs|` rows in the performance table with `model`, `metric` (where applicable), both means, and signed difference.
    - On `400 no_common_dimensions`, display the returned message.
    - _Requirements: 16.4_

  - [x] 24.3 Write property test for UI history row completeness
    - **Property 39: UI history row completeness** — each rendered history row contains `started_at`, `status`, model names, suite names, aggregate pass rate.
    - **Validates: Requirement 16.1**

  - [x] 24.4 Write property test for UI filter query round-trip
    - **Property 40: UI filter query round-trip** — filter selections encode into a query string that decodes back to the same filter on the Backend.
    - **Validates: Requirement 16.2**

  - [x] 24.5 Write property test for UI comparison table rendering
    - **Property 41: UI comparison table rendering** — the view renders exactly `|metric_diffs|` and `|performance_diffs|` rows with the expected columns.
    - **Validates: Requirement 16.4**

- [x] 25. Wire the Backend to serve the built UI
  - Add a `dist/` build step (`npm run build`) and mount `ui/dist/` as static files under FastAPI when `serve` is invoked.
  - Ensure deep-link routes (`/runs/:id`, `/history`, `/compare`) fall back to `index.html`.
  - _Requirements: 10.7, 15.1_

- [x] 26. Integration tests across the full stack
  - [x] 26.1 Integration test: happy-path Run lifecycle over REST + WebSocket
    - Uses the `FakeOllamaServer` ASGI fixture and FastAPI's `TestClient` + `websockets` client; asserts `run-started`, multiple `test-case-completed`, `run-progress`, and `run-completed` events arrive in order; asserts `GET /api/runs/{id}` returns a valid `RunReport`.
    - _Requirements: 13.1, 13.2, 13.3, 14.1, 14.2, 14.3, 14.5_

  - [x] 26.2 Integration test: remote-mode dataset preflight abort
    - Uses a `FakeHFHub` that raises on fetch; asserts preflight emits `run-failed` with `error_code=dataset_fetch_failed` before any `test-case-completed` event.
    - _Requirements: 17.3, 17.5, 17.6_

  - [x] 26.3 Integration test: HF field-map validation
    - Feeds a row with a missing declared field; asserts `run-failed` with `error_code=field_map_invalid` naming the row index and field path.
    - _Requirements: 17.7_

  - [x] 26.4 Integration test: signal-driven graceful shutdown
    - Sends SIGINT during a slow run; asserts dispatch stops, in-flight executions drain within 30s, a partial `RunReport` is written, and the WebSocket delivers `run-aborted` before closing with code 1000.
    - _Requirements: 11.4_

  - [x] 26.5 Integration test: OpenAPI document validity
    - `GET /openapi.json` parses as OpenAPI 3.1 and matches the committed `shared/openapi.yaml`.
    - _Requirements: 13.7_

  - [x] 26.6 Integration test: CLI `run` exit codes
    - Drive the CLI against the fake server; assert exit 0 when all pass, non-zero when any fail, exit 2 when Ollama is unreachable.
    - _Requirements: 10.2, 10.3, 1.3_

- [x] 27. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; every required behaviour has non-test implementation tasks covering it.
- Each task references specific sub-requirements for traceability, and every property-based test cites a numbered Correctness Property from `design.md`.
- Property tests are placed next to the implementation they verify so regressions are caught as early as possible.
- Checkpoints at tasks 10, 16, 20, and 27 mark natural integration boundaries for running the full test suite.
- The 44 property tests from the design map to these task ids: P1→3.3, P2→3.4, P3→3.5, P4→9.2, P5→12.10, P6→9.3, P7→9.4, P8→12.5, P9→12.6, P10→12.7, P11→12.8, P12→5.5, P13→5.6, P14→6.3, P15→6.2, P16→14.2, P17→14.3, P18→2.5, P19→15.2, P20→19.3, P21→12.9, P22→14.4, P23→13.4, P24→13.5, P25→13.6, P26→13.7, P27→13.8, P28→13.9, P29→17.5, P30→17.6, P31→18.3, P32→18.4, P33→18.5, P34→18.6, P35→22.2, P36→22.3, P37→23.4, P38→23.5, P39→24.3, P40→24.4, P41→24.5, P42→7.7, P43→8.3, P44→8.4.
