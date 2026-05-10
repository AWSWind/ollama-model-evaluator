"""Unit tests for :class:`ollama_evaluator.history.store.HistoryStore`.

Smoke-level coverage of the public API: create, append events, write
report, get_run, list_runs, delete_run. The property tests in
``tests/property/test_property_23…28*.py`` cover the generative
behaviour; these tests pin specific examples so regressions in the
SQL or atomic-write paths surface immediately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ollama_evaluator.config import ConfigFile, RunConfig
from ollama_evaluator.events import RunStartedEvent
from ollama_evaluator.history.store import HistoryStore, RunListFilter
from ollama_evaluator.models import (
    ModelInfo,
    PerformanceMetrics,
    RunReport,
    TestCaseResult,
)


def _config() -> ConfigFile:
    return ConfigFile(
        suites_dir=Path("suites"),
        run=RunConfig(models=["m"], suites=["s"]),
    )


def _report(run_id: str, status: str = "completed") -> RunReport:
    return RunReport(
        run_id=run_id,
        backend_version="0.1.0",
        ollama_version="0.2.0",
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        status=status,  # type: ignore[arg-type]
        config=_config(),
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


async def test_create_run_returns_distinct_ids(tmp_path: Path) -> None:
    async with HistoryStore.open(tmp_path / "db", tmp_path / "runs") as store:
        a = await store.create_run(_config())
        b = await store.create_run(_config())
        assert a != b


async def test_write_and_get_report(tmp_path: Path) -> None:
    async with HistoryStore.open(tmp_path / "db", tmp_path / "runs") as store:
        run_id = await store.create_run(_config())
        report = _report(run_id)
        await store.write_report(report)
        fetched = await store.get_run(run_id)
        assert fetched is not None
        assert fetched == report


async def test_append_and_list_events(tmp_path: Path) -> None:
    async with HistoryStore.open(tmp_path / "db", tmp_path / "runs") as store:
        run_id = await store.create_run(_config())
        ev = RunStartedEvent(
            run_id=run_id,
            seq=0,
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            planned_executions=3,
        )
        await store.append_event(ev)
        events = await store.list_events(run_id)
        assert len(events) == 1
        assert events[0].seq == 0


async def test_delete_run_removes_it(tmp_path: Path) -> None:
    async with HistoryStore.open(tmp_path / "db", tmp_path / "runs") as store:
        run_id = await store.create_run(_config())
        await store.write_report(_report(run_id))
        await store.delete_run(run_id)
        assert await store.get_run(run_id) is None


async def test_list_runs_applies_filter(tmp_path: Path) -> None:
    async with HistoryStore.open(tmp_path / "db", tmp_path / "runs") as store:
        a = await store.create_run(_config())
        b = await store.create_run(_config())
        await store.write_report(_report(a, status="completed"))
        await store.write_report(_report(b, status="failed"))

        all_runs = await store.list_runs(RunListFilter())
        assert {r.run_id for r in all_runs} == {a, b}

        only_completed = await store.list_runs(RunListFilter(status="completed"))
        assert {r.run_id for r in only_completed} == {a}
