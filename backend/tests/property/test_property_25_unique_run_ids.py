"""Feature: ollama-model-evaluator, Property 25: Unique run identifiers.

For any ``n ≥ 1`` sequential calls to :meth:`HistoryStore.create_run`,
the returned ``run_id`` values are pairwise distinct.

Validates: Requirement 12.3.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import ConfigFile, RunConfig
from ollama_evaluator.history.store import HistoryStore


def _config() -> ConfigFile:
    return ConfigFile(
        suites_dir=Path("suites"),
        run=RunConfig(models=["m"], suites=["s"]),
    )


@given(n=st.integers(min_value=1, max_value=50))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_sequential_create_run_calls_return_distinct_ids(n: int) -> None:
    """**Validates: Requirement 12.3**

    ``n`` sequential ``create_run()`` calls return ``n`` pairwise-distinct ids.
    """

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            async with HistoryStore.open(Path(td) / "db", Path(td) / "runs") as store:
                ids = [await store.create_run(_config()) for _ in range(n)]
                assert len(set(ids)) == n, f"collision in {ids}"

    asyncio.run(_run())
