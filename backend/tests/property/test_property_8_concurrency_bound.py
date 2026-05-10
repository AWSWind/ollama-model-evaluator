"""Feature: ollama-model-evaluator, Property 8: Concurrency bound.

Property 8 (from ``design.md`` §Correctness Properties):

    *For any* ``RunConfig.concurrency = C``, the maximum number of
    simultaneously in-flight Ollama generate calls observed during
    the Run is ``≤ C``.

Validates: Requirement 5.5.

Approach: drive the scheduler against a :class:`FakeOllamaClient`
that tracks concurrent invocations via a counter. The fake yields
cooperatively so the semaphore can hold dispatches off. Hypothesis
varies ``concurrency``, the number of models, the number of test
cases, and ``repetitions`` across the Property's input space and
asserts that the peak concurrent count never exceeds
``concurrency``.
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


class _AlwaysPassMetric:
    name = "always-pass-p8"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_AlwaysPassMetric())


@given(
    concurrency=st.integers(min_value=1, max_value=6),
    n_models=st.integers(min_value=1, max_value=3),
    n_cases=st.integers(min_value=1, max_value=4),
    repetitions=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_max_inflight_calls_never_exceeds_concurrency(
    concurrency: int,
    n_models: int,
    n_cases: int,
    repetitions: int,
) -> None:
    """The peak concurrent count observed by the fake client never
    exceeds ``concurrency`` regardless of plan shape."""

    async def _run() -> None:
        models = [f"m{i}" for i in range(n_models)]
        test_cases = [
            TestCase(
                id=f"tc{i}",
                prompt=f"prompt-{i}",
                metrics=[MetricConfig(name="always-pass-p8")],
            )
            for i in range(n_cases)
        ]
        suite = EvaluationSuite(name="suite", test_cases=test_cases)
        run_config = RunConfig(
            models=models,
            suites=["suite"],
            repetitions=repetitions,
            concurrency=concurrency,
        )

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models([OllamaModelInfo(name=m) for m in models])

        # Return a small multi-chunk stream so the async iterator
        # stays in-flight for a couple of await points — otherwise a
        # single-chunk stream could finish inside ``run_one`` before
        # any other task is dispatched, and the counter would only
        # ever read 1 regardless of concurrency.
        def chunk_factory(model: str, *_: object) -> list[object]:
            return make_chunks(f"out-{model}", model=model)

        fake.set_generate_chunks(chunk_factory)  # type: ignore[arg-type]

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )

        await scheduler.execute()
        assert fake.concurrency_observed <= concurrency, (
            f"concurrency_observed={fake.concurrency_observed} > "
            f"concurrency={concurrency}"
        )
        # Also assert the total dispatch count matches the plan
        # (Property 6 sanity check — strictly tested elsewhere).
        assert len(fake.dispatch_log) == n_models * n_cases * repetitions

    asyncio.run(_run())
