"""Feature: ollama-model-evaluator, Property 34: Multi-subscriber replay and isolation.

Two subscribers that subscribe at different points during a Run both
receive the full canonical sequence of events in order, with no gaps
or duplicates. Dropped subscribers do not affect the producer or
other subscribers.

Validates: Requirements 14.6, 14.7.

Approach: drive the scheduler against a :class:`FakeOllamaClient` and
have two observer coroutines subscribe at different moments — one
before execution starts, one after a short delay. Both observers run
to the terminal event and then the test compares their event lists
against each other and against the ``RunState.events`` canonical log.
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
    name = "p34-pass"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_PassMetric())

_TERMINAL_TYPES = frozenset({"run-completed", "run-aborted", "run-failed"})


@given(
    n_cases=st.integers(min_value=2, max_value=4),
    repetitions=st.integers(min_value=1, max_value=2),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_two_subscribers_receive_identical_canonical_sequence(
    n_cases: int, repetitions: int
) -> None:
    """**Validates: Requirements 14.6, 14.7**

    Both subscribers receive the full canonical sequence (equal ``seq``
    streams) even when they subscribe at different times.
    """

    async def _run() -> tuple[list, list, list]:
        cases = [
            TestCase(id=f"tc{i}", prompt=f"p{i}", metrics=[MetricConfig(name="p34-pass")])
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
            models=["m"],
            suites=["suite"],
            repetitions=repetitions,
            concurrency=1,
            retry_max_attempts=0,
        )

        observer_a_events: list = []
        observer_b_events: list = []

        async def _observe(target: list) -> None:
            async for event in bus.subscribe():
                target.append(event)
                if event.type in _TERMINAL_TYPES:
                    return

        # Subscriber A starts before the scheduler runs.
        task_a = asyncio.create_task(_observe(observer_a_events))
        # Let A subscribe.
        await asyncio.sleep(0)

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )
        sched_task = asyncio.create_task(scheduler.execute())
        # Tiny delay so B subscribes mid-flight.
        await asyncio.sleep(0)
        task_b = asyncio.create_task(_observe(observer_b_events))

        await sched_task
        await task_a
        await task_b
        return observer_a_events, observer_b_events, list(state.events)

    a_events, b_events, canonical = asyncio.run(_run())

    # Both observers ended on the terminal event.
    assert a_events and a_events[-1].type in _TERMINAL_TYPES
    assert b_events and b_events[-1].type in _TERMINAL_TYPES

    # Subscriber A started before the run, so it must have the full sequence.
    assert [e.seq for e in a_events] == [e.seq for e in canonical], (
        f"subscriber A seq mismatch: a={[e.seq for e in a_events]}, canonical={[e.seq for e in canonical]}"
    )

    # Subscriber B may start mid-flight but must still receive every event
    # (snapshot+replay behaviour in RunEventBus.subscribe).
    assert [e.seq for e in b_events] == [e.seq for e in canonical], (
        f"subscriber B seq mismatch: b={[e.seq for e in b_events]}, canonical={[e.seq for e in canonical]}"
    )

    # No duplicates in either observer.
    assert len({e.seq for e in a_events}) == len(a_events)
    assert len({e.seq for e in b_events}) == len(b_events)
