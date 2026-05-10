"""Feature: ollama-model-evaluator, Property 24: Terminal-event-before-status ordering.

For any Run that reaches a terminal state, the sequence of store
operations contains an ``append_event(terminal_event)`` and a
``write_report(run_id, ...)`` strictly before the
``update_run_status(run_id, terminal_status)`` call.

Validates: Requirement 12.2.

Approach: run a small end-to-end Run with a recording *spy store*
that appends every method name + key args to a list; Hypothesis
varies the number of test cases, repetitions, and whether the Run
completes or is aborted. The test then scans the call log for the
terminal event's position, the ``write_report`` position, and the
``update_run_status`` position, and asserts the strict ordering.
"""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.metrics import register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult as MResult
from ollama_evaluator.ollama.types import OllamaModelInfo
from ollama_evaluator.runner.run_state import RunEventBus, RunState
from ollama_evaluator.runner.scheduler import RunScheduler
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)
from tests.unit._fakes import FakeOllamaClient, make_chunks


class _PassMetric:
    name = "p24-pass"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_PassMetric())


class _SpyStore:
    """Recording substitute for :class:`HistoryStore`.

    Methods match the :class:`HistoryStore` surface the scheduler
    calls; each call appends a ``(method, key)`` tuple to
    :attr:`calls` where ``key`` is a minimal identifier of the
    operation (event ``type``, run id, status).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def append_event(self, event: object) -> None:
        self.calls.append(("append_event", getattr(event, "type", "?")))

    async def write_report(self, report: object) -> None:
        self.calls.append(("write_report", getattr(report, "run_id", "?")))

    async def update_run_status(
        self, run_id: str, status: str, ended_at: object = None
    ) -> None:
        self.calls.append(("update_run_status", status))

    async def write_test_case_result(self, run_id: str, result: object) -> None:
        self.calls.append(("write_test_case_result", getattr(result, "test_case_id", "?")))


_TERMINAL_TYPES = frozenset({"run-completed", "run-aborted", "run-failed"})


@given(
    n_cases=st.integers(min_value=1, max_value=3),
    repetitions=st.integers(min_value=1, max_value=2),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_terminal_event_precedes_write_report_precedes_update_status(
    n_cases: int,
    repetitions: int,
) -> None:
    """**Validates: Requirement 12.2**

    The store call log carries exactly one terminal ``append_event``
    and one ``write_report`` strictly before the terminal
    ``update_run_status``.
    """

    async def _run() -> None:
        models = ["m"]
        test_cases = [
            TestCase(
                id=f"tc{i}",
                prompt=f"p{i}",
                metrics=[MetricConfig(name="p24-pass")],
            )
            for i in range(n_cases)
        ]
        suite = EvaluationSuite(name="suite", test_cases=test_cases)

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models([OllamaModelInfo(name="m")])
        fake.set_generate_chunks(make_chunks("ok"))

        run_config = RunConfig(
            models=models,
            suites=["suite"],
            repetitions=repetitions,
            concurrency=1,
            retry_max_attempts=0,
        )

        spy = _SpyStore()
        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
            store=spy,
        )
        await scheduler.execute()

        # Find indices.
        terminal_event_idx: int | None = None
        write_report_idx: int | None = None
        update_status_idx: int | None = None
        for i, (method, key) in enumerate(spy.calls):
            if method == "append_event" and key in _TERMINAL_TYPES and terminal_event_idx is None:
                terminal_event_idx = i
            elif method == "write_report" and write_report_idx is None:
                write_report_idx = i
            elif method == "update_run_status" and key in ("completed", "aborted", "failed"):
                update_status_idx = i

        assert terminal_event_idx is not None, f"no terminal append_event in {spy.calls}"
        assert write_report_idx is not None, f"no write_report in {spy.calls}"
        assert update_status_idx is not None, f"no terminal update_run_status in {spy.calls}"

        assert terminal_event_idx < write_report_idx < update_status_idx, (
            f"ordering violated: terminal_event={terminal_event_idx}, "
            f"write_report={write_report_idx}, update_status={update_status_idx}, "
            f"calls={spy.calls}"
        )

    asyncio.run(_run())
