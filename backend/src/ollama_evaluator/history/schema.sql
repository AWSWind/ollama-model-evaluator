-- History_Store schema for the Ollama Model Evaluator (design §History_Store).
--
-- A single SQLite database file holds the Backend's persistent run history
-- (Requirement 12.1). The tables below are populated by
-- :class:`ollama_evaluator.history.store.HistoryStore` as a Run progresses:
--
-- * ``runs``               — one row per Run; primary index of the history.
-- * ``run_models``         — per-Run list of evaluated Models (Requirement 2.5).
-- * ``run_suites``         — per-Run list of Evaluation_Suite names referenced
--                            by the ``RunConfig.suites``.
-- * ``run_events``         — append-only ``RunEvent`` log (Requirement 14.5).
-- * ``test_case_results``  — one row per ``(model, test_case, repetition)``
--                            execution for efficient history filters on the
--                            per-execution dimension.
--
-- ``PRAGMA journal_mode=WAL`` is enabled on connection open so concurrent
-- readers (the REST API, the WebSocket replay path) are not blocked by the
-- writer (the scheduler). WAL is a per-connection pragma on SQLite but it
-- is persisted on the database file itself, so setting it once at schema
-- creation time is sufficient; the :class:`HistoryStore` also re-asserts it
-- on every connection for belt-and-braces.
--
-- ``idx_runs_status_started`` backs the history filter query
-- (Requirement 12.4, design §REST API / ``GET /api/runs``): the common case
-- filters by ``status`` and orders by ``started_at DESC`` so both columns
-- share one composite index.

PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  started_at TEXT,
  ended_at TEXT,
  backend_version TEXT NOT NULL,
  ollama_version TEXT,
  config_json TEXT NOT NULL,
  report_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_status_started ON runs(status, started_at);

CREATE TABLE IF NOT EXISTS run_models (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  model_digest TEXT,
  parameter_size TEXT,
  PRIMARY KEY (run_id, model_name)
);

CREATE TABLE IF NOT EXISTS run_suites (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  suite_name TEXT NOT NULL,
  PRIMARY KEY (run_id, suite_name)
);

CREATE TABLE IF NOT EXISTS run_events (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  ts TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS test_case_results (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  suite_name TEXT NOT NULL,
  test_case_id TEXT NOT NULL,
  repetition INTEGER NOT NULL,
  status TEXT NOT NULL,
  response_text TEXT,
  error_message TEXT,
  ttft_ms REAL,
  total_ms REAL,
  prompt_tokens INTEGER,
  response_tokens INTEGER,
  tokens_per_second REAL,
  metrics_json TEXT NOT NULL,
  PRIMARY KEY (run_id, model_name, suite_name, test_case_id, repetition)
);
