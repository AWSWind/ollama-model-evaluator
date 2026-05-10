"""SQLite-backed History_Store for the Ollama Model Evaluator.

The :class:`HistoryStore` is the Backend's persistent run history layer
(Requirement 12.1). It is the only module that writes to the SQLite
database; schema lives in ``schema.sql`` next to this file.

Design references: ``.kiro/specs/ollama-model-evaluator/design.md``
§History_Store and §REST API / ``GET /api/runs``. Requirements driving
behaviour in this module:

* 12.1 — persist Run_Reports, Run_Events, and suite metadata across
  Backend restarts.
* 12.2 — terminal Run_Events are persisted before the ``runs.status``
  column transitions to a terminal value. The scheduler enforces the
  call ordering; this module just exposes the three mutators.
* 12.3 — ``create_run`` mints a globally unique id and uses it as the
  primary key.
* 12.4 — ``list_runs(filter)`` supports filters on model, suite,
  status, and a time range.
* 12.5 — ``delete_run(run_id)`` removes a Run and its child rows.
* 12.6 — corrupted records are logged and skipped, not fatal.
* 14.5 — events are persisted in ``seq`` order per Run; the
  ``(run_id, seq)`` composite primary key prevents gaps/duplicates.

The store is async-aware via :mod:`aiosqlite` but exposes an
``async with HistoryStore.open(db_path, report_dir) as store`` context
manager so tests and the CLI can open/close the connection without
leaking handles.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from ..config import ConfigFile
from ..events import RunEvent, RunEventAdapter
from ..models import RunReport, TestCaseResult


log = logging.getLogger(__name__)


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _iso(dt: datetime | None) -> str | None:
    """ISO-8601 UTC rendering for nullable datetimes.

    ``datetime.isoformat()`` already emits timezone information when
    the input is aware; the helper just normalises the ``None`` path.
    """
    return dt.isoformat() if dt is not None else None


class RunListFilter(BaseModel):
    """Filter for :meth:`HistoryStore.list_runs` (Requirement 12.4).

    Every field is optional; the store builds a WHERE clause from the
    fields that are not ``None``. The semantics are AND across
    specified fields; unspecified fields impose no constraint
    (Property 26).

    ``since`` / ``until`` are inclusive lower and upper bounds on
    ``started_at``. ``model`` / ``suite`` match against the
    ``run_models.model_name`` / ``run_suites.suite_name`` columns via a
    join — a Run matches if *any* of its models/suites equal the
    filter value (Property 26).
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(
        default=None,
        description="Restrict to Runs that evaluated this model.",
    )
    suite: str | None = Field(
        default=None,
        description="Restrict to Runs that included this Evaluation_Suite name.",
    )
    status: str | None = Field(
        default=None,
        description="Restrict to Runs in this state.",
    )
    since: datetime | None = Field(
        default=None,
        description="Inclusive lower bound on ``started_at``.",
    )
    until: datetime | None = Field(
        default=None,
        description="Inclusive upper bound on ``started_at``.",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (write-temp, rename).

    Kept inside this module so :meth:`HistoryStore.write_report` does
    not have to import the reports module (which the scheduler owns
    via Task 14.1). Cross-device renames on Windows can fail if the
    temp file is on a different volume, so the temp file is created
    in the destination directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        # Best-effort cleanup — swallow any remove error so the
        # original exception is the one propagated.
        try:
            os.unlink(tmp_name)
        except OSError:  # pragma: no cover - defensive
            pass
        raise


class HistoryStore:
    """Async SQLite-backed history store.

    Use :meth:`open` as an async context manager::

        async with HistoryStore.open(db_path, report_dir) as store:
            run_id = await store.create_run(config)
            await store.write_report(report)
            await store.update_run_status(run_id, "completed", ended_at=...)

    The store owns exactly one :class:`aiosqlite.Connection` across
    its lifetime. Callers that need additional concurrency should
    open additional stores.
    """

    def __init__(
        self,
        connection: aiosqlite.Connection,
        report_dir: Path,
    ) -> None:
        self._conn = connection
        self._report_dir = report_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        db_path: str | Path,
        report_dir: str | Path,
    ) -> AsyncIterator["HistoryStore"]:
        """Open the store, applying the schema, and close on exit.

        ``db_path`` may be ``:memory:`` for tests. ``report_dir`` is
        created lazily if missing.
        """
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(str(db_path))
        try:
            # Foreign keys + WAL. WAL is persisted on the db file but
            # also needs re-asserting at connection level on some
            # SQLite builds for the reader-pref semantics to apply.
            await conn.execute("PRAGMA foreign_keys = ON;")
            try:
                await conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:  # pragma: no cover
                # In-memory DBs reject WAL on some platforms; fall
                # back silently — tests do not need WAL semantics.
                pass

            schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            await conn.executescript(schema_sql)
            await conn.commit()
            store = cls(conn, report_dir)
            yield store
        finally:
            await conn.close()

    @property
    def report_dir(self) -> Path:
        """Root directory for on-disk Run_Report artifacts."""
        return self._report_dir

    # ------------------------------------------------------------------
    # Run lifecycle (Requirements 12.1, 12.2, 12.3)
    # ------------------------------------------------------------------

    async def create_run(self, config: ConfigFile) -> str:
        """Insert a new ``pending`` Run and return its minted id.

        ``uuid.uuid4().hex`` is used so two sequential calls on the
        same store return pairwise-distinct ids (Property 25). The
        Run starts with ``status = 'pending'``; callers transition it
        via :meth:`update_run_status`.
        """
        run_id = uuid.uuid4().hex
        # ``ConfigFile`` is Pydantic; ``model_dump_json`` handles
        # pathlib.Path → str and is stable across re-loads.
        config_json = config.model_dump_json()

        await self._conn.execute(
            """
            INSERT INTO runs (
                run_id, status, started_at, ended_at,
                backend_version, ollama_version, config_json, report_path
            ) VALUES (?, 'pending', NULL, NULL, ?, NULL, ?, NULL)
            """,
            (run_id, _BACKEND_VERSION_PLACEHOLDER, config_json),
        )
        # Populate run_suites from the config so filters on suite
        # name work before a report is written.
        for suite_name in config.run.suites:
            await self._conn.execute(
                "INSERT OR IGNORE INTO run_suites (run_id, suite_name) VALUES (?, ?)",
                (run_id, suite_name),
            )
        # Same for models — even though digests are only known after
        # preflight, having the names here lets ``list_runs(filter)``
        # find the Run while it is still ``pending``.
        for model_name in config.run.models:
            await self._conn.execute(
                "INSERT OR IGNORE INTO run_models (run_id, model_name) VALUES (?, ?)",
                (run_id, model_name),
            )
        await self._conn.commit()
        return run_id

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        ended_at: datetime | None = None,
    ) -> None:
        """Set ``runs.status`` (and ``ended_at`` when terminal)."""
        await self._conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?",
            (status, _iso(ended_at), run_id),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Event log (Requirement 14.5)
    # ------------------------------------------------------------------

    async def append_event(self, event: RunEvent) -> None:
        """Persist ``event`` to ``run_events`` using its own ``seq``.

        The event's ``seq`` must already have been assigned by the
        :class:`RunEventBus`; this method does not mint its own so
        the on-disk and in-memory sequences stay identical.
        """
        payload = RunEventAdapter.dump_json(event).decode("utf-8")
        await self._conn.execute(
            """
            INSERT INTO run_events (run_id, seq, event_type, ts, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event.run_id, event.seq, event.type, _iso(event.ts), payload),
        )
        await self._conn.commit()

    async def list_events(self, run_id: str) -> list[RunEvent]:
        """Return all persisted events for ``run_id`` in ``seq`` order."""
        cur = await self._conn.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? ORDER BY seq ASC",
            (run_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
        events: list[RunEvent] = []
        for (payload,) in rows:
            try:
                events.append(RunEventAdapter.validate_json(payload))
            except Exception as exc:  # noqa: BLE001 — Pydantic / JSON.
                log.warning("skipping corrupted event for run %s: %s", run_id, exc)
        return events

    # ------------------------------------------------------------------
    # Per-execution results
    # ------------------------------------------------------------------

    async def write_test_case_result(
        self, run_id: str, result: TestCaseResult
    ) -> None:
        """Insert one row into ``test_case_results``."""
        metrics_json = json.dumps([m.model_dump(mode="json") for m in result.metrics])
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO test_case_results (
                run_id, model_name, suite_name, test_case_id, repetition,
                status, response_text, error_message,
                ttft_ms, total_ms, prompt_tokens, response_tokens, tokens_per_second,
                metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result.model,
                result.suite,
                result.test_case_id,
                result.repetition,
                result.status,
                result.response,
                result.error_message,
                result.performance.ttft_ms,
                result.performance.total_ms,
                result.performance.prompt_tokens,
                result.performance.response_tokens,
                result.performance.tokens_per_second,
                metrics_json,
            ),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Run_Report (JSON artifact + DB pointer)
    # ------------------------------------------------------------------

    async def write_report(self, report: RunReport) -> Path:
        """Write the JSON Run_Report and update ``runs`` accordingly.

        The canonical JSON artifact goes to
        ``<report_dir>/<run_id>/report.json`` and its filesystem path
        is recorded in the ``runs.report_path`` column so the REST
        API can serve the file directly (Requirement 8.1). The
        Markdown artifact is written by the scheduler's
        ``reports.write_artifacts`` helper (Task 14.1); the store
        does not touch ``report.md``.

        Besides the primary artifact write, the store also refreshes
        ``runs`` metadata (status, started_at, ended_at, versions,
        config JSON), replaces the ``run_models`` rows with the full
        :class:`ModelInfo` entries from the report, and upserts every
        :class:`TestCaseResult` in ``report.results``. This keeps
        history queries in sync with the serialised artifact even if
        the caller only ever invokes ``write_report`` at terminal
        time.
        """
        run_dir = self._report_dir / report.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "report.json"
        _atomic_write_text(report_path, report.model_dump_json(indent=2))

        # Refresh runs row.
        await self._conn.execute(
            """
            UPDATE runs
               SET status = ?,
                   started_at = ?,
                   ended_at = ?,
                   backend_version = ?,
                   ollama_version = ?,
                   config_json = ?,
                   report_path = ?
             WHERE run_id = ?
            """,
            (
                report.status,
                _iso(report.started_at),
                _iso(report.ended_at),
                report.backend_version,
                report.ollama_version,
                report.config.model_dump_json(),
                str(report_path),
                report.run_id,
            ),
        )
        # If the run was not previously inserted by ``create_run`` (e.g.
        # tests that write a report directly), insert it now.
        cur = await self._conn.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (report.run_id,)
        )
        exists = await cur.fetchone()
        await cur.close()
        if not exists:
            await self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, status, started_at, ended_at,
                    backend_version, ollama_version, config_json, report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.status,
                    _iso(report.started_at),
                    _iso(report.ended_at),
                    report.backend_version,
                    report.ollama_version,
                    report.config.model_dump_json(),
                    str(report_path),
                ),
            )

        # Replace per-Run model rows with the full ModelInfo set.
        await self._conn.execute(
            "DELETE FROM run_models WHERE run_id = ?", (report.run_id,)
        )
        for info in report.models:
            await self._conn.execute(
                """
                INSERT OR REPLACE INTO run_models (run_id, model_name, model_digest, parameter_size)
                VALUES (?, ?, ?, ?)
                """,
                (report.run_id, info.name, info.digest, info.parameter_size),
            )

        # Keep suites in sync with the embedded config.
        await self._conn.execute(
            "DELETE FROM run_suites WHERE run_id = ?", (report.run_id,)
        )
        for suite_name in report.config.run.suites:
            await self._conn.execute(
                "INSERT OR IGNORE INTO run_suites (run_id, suite_name) VALUES (?, ?)",
                (report.run_id, suite_name),
            )

        # Upsert every per-execution result for the history filters.
        for result in report.results:
            metrics_json = json.dumps(
                [m.model_dump(mode="json") for m in result.metrics]
            )
            await self._conn.execute(
                """
                INSERT OR REPLACE INTO test_case_results (
                    run_id, model_name, suite_name, test_case_id, repetition,
                    status, response_text, error_message,
                    ttft_ms, total_ms, prompt_tokens, response_tokens, tokens_per_second,
                    metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    result.model,
                    result.suite,
                    result.test_case_id,
                    result.repetition,
                    result.status,
                    result.response,
                    result.error_message,
                    result.performance.ttft_ms,
                    result.performance.total_ms,
                    result.performance.prompt_tokens,
                    result.performance.response_tokens,
                    result.performance.tokens_per_second,
                    metrics_json,
                ),
            )

        await self._conn.commit()
        return report_path

    # ------------------------------------------------------------------
    # Queries (Requirements 12.4, 12.5, 12.6)
    # ------------------------------------------------------------------

    async def get_run(self, run_id: str) -> RunReport | None:
        """Return the :class:`RunReport` for ``run_id`` or ``None``."""
        cur = await self._conn.execute(
            "SELECT report_path FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return None
        (report_path,) = row
        if not report_path:
            return None
        path = Path(report_path)
        if not path.exists():
            return None
        try:
            return RunReport.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — disk read + Pydantic validation.
            log.warning("skipping corrupted run %s: %s", run_id, exc)
            return None

    async def list_runs(
        self, filter: RunListFilter | None = None
    ) -> list[RunReport]:
        """Return persisted :class:`RunReport`s matching ``filter``.

        Corrupted rows — those whose ``config_json`` or on-disk
        ``report.json`` fail to parse as valid Pydantic models — are
        logged at WARNING level and skipped (Requirement 12.6).
        """
        f = filter or RunListFilter()
        # ``run_ids`` is built progressively by intersecting the per-
        # join filters so the final fetch issues a single SELECT.
        where_clauses: list[str] = []
        params: list[Any] = []

        if f.status is not None:
            where_clauses.append("r.status = ?")
            params.append(f.status)
        if f.since is not None:
            where_clauses.append("r.started_at >= ?")
            params.append(_iso(f.since))
        if f.until is not None:
            where_clauses.append("r.started_at <= ?")
            params.append(_iso(f.until))

        joins = ""
        if f.model is not None:
            joins += " JOIN run_models rm ON rm.run_id = r.run_id "
            where_clauses.append("rm.model_name = ?")
            params.append(f.model)
        if f.suite is not None:
            joins += " JOIN run_suites rs ON rs.run_id = r.run_id "
            where_clauses.append("rs.suite_name = ?")
            params.append(f.suite)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = (
            "SELECT DISTINCT r.run_id, r.config_json, r.report_path "
            "FROM runs r " + joins + " " + where_sql + " ORDER BY r.started_at ASC"
        )

        cur = await self._conn.execute(query, params)
        rows = await cur.fetchall()
        await cur.close()

        reports: list[RunReport] = []
        for run_id, config_json, report_path in rows:
            try:
                # Eager Pydantic validation of the embedded config.
                # Corrupted JSON blobs are skipped with a warning per
                # Requirement 12.6 / Property 28.
                ConfigFile.model_validate_json(config_json)
            except Exception as exc:  # noqa: BLE001 — tolerate any shape error.
                log.warning("skipping corrupted run %s: %s", run_id, exc)
                continue

            if report_path:
                path = Path(report_path)
                if path.exists():
                    try:
                        reports.append(
                            RunReport.model_validate_json(
                                path.read_text(encoding="utf-8")
                            )
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "skipping corrupted run %s: %s", run_id, exc
                        )
                        continue

        return reports

    async def delete_run(self, run_id: str) -> None:
        """Remove ``run_id`` from the store (Requirement 12.5).

        ``ON DELETE CASCADE`` on the child tables removes the related
        ``run_models`` / ``run_suites`` / ``run_events`` /
        ``test_case_results`` rows automatically; the on-disk report
        file is also removed on a best-effort basis.
        """
        cur = await self._conn.execute(
            "SELECT report_path FROM runs WHERE run_id = ?", (run_id,)
        )
        row = await cur.fetchone()
        await cur.close()

        await self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        await self._conn.commit()

        if row and row[0]:
            try:
                p = Path(row[0])
                if p.exists():
                    p.unlink()
                # Also attempt to remove the enclosing run directory
                # if it is now empty. Silent on failure — callers
                # that care about directory cleanup can inspect the
                # filesystem separately.
                parent = p.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass


# ``create_run`` and ``write_report`` both need a stable
# ``backend_version`` value for rows that are inserted before a full
# report is available. The module-level constant is updated from
# :mod:`ollama_evaluator.__init__` at import time so tests and the
# scheduler see the same string.
from .. import __version__ as _BACKEND_VERSION_PLACEHOLDER  # noqa: E402


__all__ = [
    "HistoryStore",
    "RunListFilter",
]
