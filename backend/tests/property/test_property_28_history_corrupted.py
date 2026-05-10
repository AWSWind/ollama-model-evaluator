"""Feature: ollama-model-evaluator, Property 28: History skips corrupted records.

For any :class:`HistoryStore` containing ``N`` records of which an
arbitrary subset ``C`` have corrupted ``payload_json``, ``list_runs()``
returns exactly the ``N − |C|`` non-corrupted records and emits a
warning log entry for each record in ``C``.

Validates: Requirement 12.6.

Approach: write a small set of healthy reports, then open the SQLite
database directly and overwrite the ``runs.config_json`` column of a
Hypothesis-selected subset with invalid JSON. Reopen the store and
call :meth:`list_runs`; assert the length equals ``N − |C|`` and that
``caplog`` captured one WARNING per corrupted row.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import ConfigFile, RunConfig
from ollama_evaluator.history.store import HistoryStore, RunListFilter
from ollama_evaluator.models import (
    ModelInfo,
    PerformanceMetrics,
    RunReport,
    TestCaseResult,
)


def _make_report(run_id: str, offset: int = 0) -> RunReport:
    started = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset)
    return RunReport(
        run_id=run_id,
        backend_version="0.1.0",
        ollama_version=None,
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        status="completed",
        config=ConfigFile(
            suites_dir=Path("suites"),
            run=RunConfig(models=["m"], suites=["s"]),
        ),
        models=[ModelInfo(name="m")],
        results=[
            TestCaseResult(
                model="m",
                suite="s",
                test_case_id="tc",
                repetition=1,
                status="pass",
                response="ok",
                error_message=None,
                performance=PerformanceMetrics(total_ms=1.0),
                metrics=[],
            )
        ],
        aggregates=[],
        error_summary=[],
    )


@given(
    total=st.integers(min_value=1, max_value=5),
    corrupt_indices=st.lists(
        st.integers(min_value=0, max_value=4),
        min_size=0,
        max_size=5,
        unique=True,
    ),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_list_runs_skips_corrupted_rows_and_warns(
    total: int,
    corrupt_indices: list[int],
    caplog,  # type: ignore[no-untyped-def]
) -> None:
    """**Validates: Requirement 12.6**

    Every non-corrupted row is returned, every corrupted row is
    skipped with a WARNING log, and total healthy returned ==
    ``total - |{ i ∈ corrupt_indices : i < total }|``.
    """
    # Only corrupt indices that actually exist in the written set.
    corrupt_set = {i for i in corrupt_indices if i < total}
    expected_count = total - len(corrupt_set)

    run_ids = [uuid.uuid4().hex for _ in range(total)]

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "history.db"
            report_dir = Path(td) / "runs"

            async with HistoryStore.open(db_path, report_dir) as store:
                for i, rid in enumerate(run_ids):
                    await store.write_report(_make_report(rid, offset=i))

            # Corrupt selected rows by writing gibberish into
            # ``runs.config_json``. Using sqlite3 synchronously here
            # is fine; no async store is open during the mutation.
            if corrupt_set:
                conn = sqlite3.connect(str(db_path))
                try:
                    for i in corrupt_set:
                        conn.execute(
                            "UPDATE runs SET config_json = ? WHERE run_id = ?",
                            ("{ not valid json", run_ids[i]),
                        )
                    conn.commit()
                finally:
                    conn.close()

            caplog.clear()
            caplog.set_level(logging.WARNING, logger="ollama_evaluator.history.store")
            async with HistoryStore.open(db_path, report_dir) as store:
                listed = await store.list_runs(RunListFilter())

            assert len(listed) == expected_count, (
                f"expected {expected_count} healthy, got {len(listed)}"
            )

            # At least one warning per corrupted row.
            warning_records = [
                r for r in caplog.records if r.levelno == logging.WARNING
            ]
            assert len(warning_records) >= len(corrupt_set), (
                f"expected ≥{len(corrupt_set)} warnings, got {len(warning_records)}"
            )
            # The warning message naming convention is "skipping
            # corrupted run <run_id>: ..." — check each corrupted id
            # appears somewhere in the captured messages.
            joined = "\n".join(r.getMessage() for r in warning_records)
            for i in corrupt_set:
                assert run_ids[i] in joined, (
                    f"no warning mentioned run_id={run_ids[i]}: {joined}"
                )

    asyncio.run(_run())
