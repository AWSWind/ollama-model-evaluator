"""Feature: ollama-model-evaluator, Property 31: Event log bookends.

For any event log captured from :meth:`RunScheduler.execute`, the log
begins with exactly one ``run-started`` event and ends with exactly
one terminal event (``run-completed``/``run-aborted``/``run-failed``);
no events follow the terminal event.

Validates: Requirements 14.2, 14.5.

Approach: vary the number of test cases and repetitions via Hypothesis,
run the scheduler against a :class:`FakeOllamaClient`, and inspect
``RunState.events`` after the run completes.
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
    name = "p31-pass"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_PassMetric())

_TERMINAL_TYPES = frozenset({"run-completed", "run-aborted", "run-failed"})


@given(
    n_cases=st.integers(min_value=1, max_value=3),
    repetitions=st.integers(min_value=1, max_value=2),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_event_log_starts_with_run_started_and_ends_with_terminal(
    n_cases: int, repetitions: int
) -> None:
    """**Validates: Requirements 14.2, 14.5**"""

    async def _run() -> list:
        models = ["m"]
        cases = [
            TestCase(id=f"tc{i}", prompt=f"p{i}", metrics=[MetricConfig(name="p31-pass")])
            for i in range(n_cases)
        ]
        suite = EvaluationSuite(name="suite", test_cases=cases)

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2")
        fake.set_models([OllamaModelInfo(name="m")])
        fake.set_generate_chunks(make_chunks("ok"))

        run_config = RunConfig(
            models=models,
            suites=["suite"],
            repetitions=repetitions,
            concurrency=1,
            retry_max_attempts=0,
        )

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )
        await scheduler.execute()
        return list(state.events)

    events = asyncio.run(_run())

    assert events, "run produced no events"
    assert events[0].type == "run-started", (
        f"first event must be run-started, got {events[0].type}"
    )
    # Exactly one run-started.
    started_count = sum(1 for e in events if e.type == "run-started")
    assert started_count == 1, f"expected exactly 1 run-started, got {started_count}"

    # Exactly one terminal event, positioned last.
    terminal_count = sum(1 for e in events if e.type in _TERMINAL_TYPES)
    assert terminal_count == 1, (
        f"expected exactly 1 terminal event, got {terminal_count}: "
        f"{[e.type for e in events]}"
    )
    assert events[-1].type in _TERMINAL_TYPES, (
        f"last event must be terminal, got {events[-1].type}"
    )
