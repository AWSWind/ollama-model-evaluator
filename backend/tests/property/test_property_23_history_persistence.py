"""Feature: ollama-model-evaluator, Property 23: History persistence across restart.

For any sequence of persisted :class:`RunReport` values, closing and
reopening the :class:`HistoryStore` against the same database file
and report directory returns, for every ``run_id``, a ``RunReport``
equal to the one written.

Validates: Requirement 12.1.

Approach: draw a short list of :class:`RunReport` instances with
distinct ``run_id`` values from :func:`run_reports`, write each one
into a fresh store under a temporary directory, close the store,
reopen it, and assert every report is retrievable and equal to the
original.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.history.store import HistoryStore
from ollama_evaluator.models import RunReport

from .generators import run_reports


@given(reports=st.lists(run_reports(), min_size=1, max_size=4))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_reports_round_trip_across_store_restart(reports: list[RunReport]) -> None:
    """**Validates: Requirement 12.1**

    Writing N reports, closing the store, reopening it, and reading
    each ``run_id`` back yields the same :class:`RunReport` objects.
    """
    # De-duplicate ``run_id`` values across the draw — the hypothesis
    # strategy is independent per element, so collisions are possible
    # and would make one of the two writes the "winner" on read.
    seen: set[str] = set()
    uniq_reports: list[RunReport] = []
    for r in reports:
        if r.run_id in seen:
            r = r.model_copy(update={"run_id": r.run_id + uuid.uuid4().hex[:8]})
        seen.add(r.run_id)
        uniq_reports.append(r)

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "history.db"
            report_dir = Path(td) / "runs"

            async with HistoryStore.open(db_path, report_dir) as store:
                for r in uniq_reports:
                    await store.write_report(r)

            # Reopen against the same paths.
            async with HistoryStore.open(db_path, report_dir) as store:
                for r in uniq_reports:
                    fetched = await store.get_run(r.run_id)
                    assert fetched == r, (
                        f"mismatch for run_id={r.run_id!r}: {fetched!r} != {r!r}"
                    )

    asyncio.run(_run())
