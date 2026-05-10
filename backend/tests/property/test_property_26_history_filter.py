"""Feature: ollama-model-evaluator, Property 26: History filter semantics.

For any set of persisted runs ``R`` and any filter
``(model?, suite?, status?, since?, until?)``, ``list_runs(filter)``
returns exactly the runs in ``R`` that satisfy every non-``None``
field of the filter.

Validates: Requirement 12.4.

Approach: draw a small universe of (run, filter) pairs using a
bounded strategy. Compute the Python reference predicate — a run
matches iff every non-``None`` filter field agrees with a
corresponding row field — and assert the store returns that exact
set of ``run_id``s.
"""

from __future__ import annotations

import asyncio
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


_STATUSES = ("pending", "running", "completed", "aborted", "failed")
_MODELS = ("m-a", "m-b", "m-c")
_SUITES = ("s-a", "s-b", "s-c")


def _report(
    run_id: str,
    models: list[str],
    suites: list[str],
    status: str,
    started_at: datetime,
) -> RunReport:
    return RunReport(
        run_id=run_id,
        backend_version="0.1.0",
        ollama_version=None,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=1),
        status=status,  # type: ignore[arg-type]
        config=ConfigFile(
            suites_dir=Path("suites"),
            run=RunConfig(models=models, suites=suites),
        ),
        models=[ModelInfo(name=m) for m in models],
        results=[
            TestCaseResult(
                model=models[0],
                suite=suites[0],
                test_case_id="tc",
                repetition=1,
                status="pass",
                response=None,
                error_message=None,
                performance=PerformanceMetrics(total_ms=1.0),
                metrics=[],
            )
        ],
        aggregates=[],
        error_summary=[],
    )


_run_spec_strategy = st.fixed_dictionaries(
    {
        "models": st.lists(st.sampled_from(_MODELS), min_size=1, max_size=2, unique=True),
        "suites": st.lists(st.sampled_from(_SUITES), min_size=1, max_size=2, unique=True),
        "status": st.sampled_from(_STATUSES),
        "started_offset": st.integers(min_value=0, max_value=30),  # days
    }
)


def _matches(
    spec: dict,
    *,
    model: str | None,
    suite: str | None,
    status: str | None,
    since: datetime | None,
    until: datetime | None,
    started_at: datetime,
) -> bool:
    if model is not None and model not in spec["models"]:
        return False
    if suite is not None and suite not in spec["suites"]:
        return False
    if status is not None and spec["status"] != status:
        return False
    if since is not None and started_at < since:
        return False
    if until is not None and started_at > until:
        return False
    return True


@given(
    specs=st.lists(_run_spec_strategy, min_size=1, max_size=4),
    filter_model=st.one_of(st.none(), st.sampled_from(_MODELS)),
    filter_suite=st.one_of(st.none(), st.sampled_from(_SUITES)),
    filter_status=st.one_of(st.none(), st.sampled_from(_STATUSES)),
    filter_since_days=st.one_of(st.none(), st.integers(min_value=0, max_value=30)),
    filter_until_days=st.one_of(st.none(), st.integers(min_value=0, max_value=30)),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_list_runs_matches_reference_predicate(
    specs: list[dict],
    filter_model: str | None,
    filter_suite: str | None,
    filter_status: str | None,
    filter_since_days: int | None,
    filter_until_days: int | None,
) -> None:
    """**Validates: Requirement 12.4**

    ``list_runs(filter)`` returns exactly those runs matching every
    non-``None`` filter field.
    """

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            async with HistoryStore.open(Path(td) / "db", Path(td) / "runs") as store:
                # Build per-spec (run_id, report, started_at) triples so
                # we can reference each one in the expected set below.
                triples = []
                for spec in specs:
                    run_id = uuid.uuid4().hex
                    started_at = base + timedelta(days=spec["started_offset"])
                    report = _report(
                        run_id=run_id,
                        models=list(spec["models"]),
                        suites=list(spec["suites"]),
                        status=spec["status"],
                        started_at=started_at,
                    )
                    await store.write_report(report)
                    triples.append((spec, run_id, started_at))

                since = (
                    base + timedelta(days=filter_since_days)
                    if filter_since_days is not None
                    else None
                )
                until = (
                    base + timedelta(days=filter_until_days)
                    if filter_until_days is not None
                    else None
                )
                flt = RunListFilter(
                    model=filter_model,
                    suite=filter_suite,
                    status=filter_status,
                    since=since,
                    until=until,
                )

                expected_ids = {
                    run_id
                    for spec, run_id, started_at in triples
                    if _matches(
                        spec,
                        model=filter_model,
                        suite=filter_suite,
                        status=filter_status,
                        since=since,
                        until=until,
                        started_at=started_at,
                    )
                }

                actual = await store.list_runs(flt)
                actual_ids = {r.run_id for r in actual}
                assert actual_ids == expected_ids, (
                    f"filter={flt!r} expected={expected_ids} got={actual_ids}"
                )

    asyncio.run(_run())
