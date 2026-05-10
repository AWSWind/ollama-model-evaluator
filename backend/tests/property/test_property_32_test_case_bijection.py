"""Feature: ollama-model-evaluator, Property 32: test-case-completed bijection.

``test-case-completed`` events correspond 1:1 to executed
``(model, test_case_id, repetition)`` tuples and carry the full
:class:`TestCaseResult` for the execution.

Validates: Requirement 14.3.

Approach: Hypothesis varies model/test-case counts and repetitions,
runs the scheduler, then compares the set of
``(model, test_case_id, repetition)`` tuples harvested from
``test-case-completed`` events against the plan expansion set.
"""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.metrics import register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult as MResult
from ollama_evaluator.models import TestCaseResult
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
    name = "p32-pass"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_PassMetric())


@given(
    n_models=st.integers(min_value=1, max_value=2),
    n_cases=st.integers(min_value=1, max_value=3),
    repetitions=st.integers(min_value=1, max_value=2),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_test_case_completed_events_map_to_plan(
    n_models: int, n_cases: int, repetitions: int
) -> None:
    """**Validates: Requirement 14.3**

    The set of ``(model, test_case_id, repetition)`` tuples harvested
    from ``test-case-completed`` events equals the plan, and each
    event carries a full :class:`TestCaseResult`.
    """

    async def _run() -> list:
        models = [f"m{i}" for i in range(n_models)]
        cases = [
            TestCase(id=f"tc{i}", prompt=f"p{i}", metrics=[MetricConfig(name="p32-pass")])
            for i in range(n_cases)
        ]
        suite = EvaluationSuite(name="suite", test_cases=cases)

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2")
        fake.set_models([OllamaModelInfo(name=m) for m in models])
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

    expected = {
        (m, c, r)
        for m in [f"m{i}" for i in range(n_models)]
        for c in [f"tc{i}" for i in range(n_cases)]
        for r in range(1, repetitions + 1)
    }

    observed = []
    for e in events:
        if e.type == "test-case-completed":
            assert isinstance(e.result, TestCaseResult), (
                "test-case-completed must carry a full TestCaseResult"
            )
            observed.append((e.result.model, e.result.test_case_id, e.result.repetition))

    assert len(observed) == len(expected), (
        f"bijection violated: len(observed)={len(observed)}, len(expected)={len(expected)}"
    )
    assert set(observed) == expected, (
        f"observed set {set(observed)} != expected {expected}"
    )
    # No duplicates (bijection — exactly one event per tuple).
    assert len(set(observed)) == len(observed), (
        f"duplicate tuples in observed: {observed}"
    )
