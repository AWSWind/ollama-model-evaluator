"""Feature: ollama-model-evaluator, Property 27: History delete.

For any persisted run ``r``, after ``delete_run(r.run_id)``,
``get_run(r.run_id)`` returns ``None`` and ``list_runs()`` does not
include ``r.run_id``; other runs are unaffected.

Validates: Requirement 12.5.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.history.store import HistoryStore, RunListFilter
from ollama_evaluator.models import RunReport

from .generators import run_reports


@given(
    reports=st.lists(run_reports(), min_size=2, max_size=5),
    delete_index=st.integers(min_value=0, max_value=10),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_delete_run_removes_target_only(
    reports: list[RunReport],
    delete_index: int,
) -> None:
    """**Validates: Requirement 12.5**

    Deleting one run removes it from ``get_run`` and ``list_runs``;
    the other runs remain intact.
    """
    # De-duplicate run_ids that Hypothesis might have generated
    # identically across the list.
    seen: set[str] = set()
    unique: list[RunReport] = []
    for r in reports:
        if r.run_id in seen:
            r = r.model_copy(update={"run_id": r.run_id + uuid.uuid4().hex[:8]})
        seen.add(r.run_id)
        unique.append(r)

    target = unique[delete_index % len(unique)]
    survivors = [r for r in unique if r.run_id != target.run_id]

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            async with HistoryStore.open(Path(td) / "db", Path(td) / "runs") as store:
                for r in unique:
                    await store.write_report(r)

                await store.delete_run(target.run_id)

                assert await store.get_run(target.run_id) is None
                remaining = {r.run_id for r in await store.list_runs(RunListFilter())}
                assert target.run_id not in remaining
                for s in survivors:
                    assert s.run_id in remaining, f"missing survivor {s.run_id}"
                    fetched = await store.get_run(s.run_id)
                    assert fetched == s

    asyncio.run(_run())
