"""Feature: ollama-model-evaluator, Property 9: Scheduler failure isolation.

Property 9 (from ``design.md`` §Correctness Properties):

    *For any* Run and any subset ``F`` of planned executions that
    fail with a timeout or Ollama 5xx/4xx error, every execution in
    ``F`` appears in ``Run_Report.results`` with status in
    ``{timeout, error}`` and every execution not in ``F`` appears
    with status in ``{pass, fail}``, and the total number of results
    equals the planned execution count.

Validates: Requirements 1.5, 5.6.

Approach: generate a plan and a failure policy keyed by the
per-call dispatch index. A :class:`FakeOllamaClient` consults the
policy on each call to decide whether to raise a
:class:`httpx.ReadTimeout` (→ ``timeout``), an
:class:`OllamaHTTPError` 400 (→ ``error``), or succeed. The test
recovers the failing indices from the fake's dispatch log and checks
that each one maps to the corresponding status in the report.
"""

from __future__ import annotations

import asyncio

import httpx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.metrics import register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult as MResult
from ollama_evaluator.ollama.errors import OllamaHTTPError
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
    name = "always-pass-p9"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_AlwaysPassMetric())


@given(
    n_cases=st.integers(min_value=2, max_value=4),
    repetitions=st.integers(min_value=1, max_value=2),
    failure_policy=st.lists(
        st.sampled_from(["ok", "timeout", "error"]),
        min_size=4,
        max_size=16,
    ),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_failures_isolated_per_execution(
    n_cases: int,
    repetitions: int,
    failure_policy: list[str],
) -> None:
    """Build a Run whose per-call outcomes follow ``failure_policy``
    and assert the property's three clauses on the resulting
    :class:`RunReport`."""

    async def _run() -> None:
        models = ["m"]
        test_cases = [
            TestCase(
                id=f"tc{i}",
                prompt=f"p{i}",
                metrics=[MetricConfig(name="always-pass-p9")],
            )
            for i in range(n_cases)
        ]
        suite = EvaluationSuite(name="suite", test_cases=test_cases)
        planned = len(models) * n_cases * repetitions

        # Repeat the policy so we have one entry per planned dispatch.
        # Hypothesis's list-min-size covers the small cases; we still
        # cycle for larger plans.
        per_call_policy: list[str] = [
            failure_policy[i % len(failure_policy)] for i in range(planned)
        ]

        # Map the policy labels to outcomes applied by the fake on
        # each generate invocation. We keep a shared counter in a
        # mutable closure so the factory picks the right slot for
        # each dispatch.
        call_counter = {"i": 0}

        def chunk_factory(model: str, prompt: str, *_: object) -> list[object]:
            idx = call_counter["i"]
            call_counter["i"] += 1
            outcome = per_call_policy[idx] if idx < len(per_call_policy) else "ok"
            if outcome == "timeout":
                raise httpx.ReadTimeout("simulated timeout", request=None)  # type: ignore[arg-type]
            if outcome == "error":
                raise OllamaHTTPError(status=400, url="/api/generate", body="bad")
            return make_chunks("ok", model=model)

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models([OllamaModelInfo(name="m")])
        fake.set_generate_chunks(chunk_factory)  # type: ignore[arg-type]

        run_config = RunConfig(
            models=models,
            suites=["suite"],
            repetitions=repetitions,
            # Use concurrency=1 so the dispatch order is deterministic
            # and the call_counter slot lines up with plan order.
            concurrency=1,
            # No retries so each dispatch gets exactly one outcome.
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
        report = await scheduler.execute()

        # Clause 3: total results == planned.
        assert len(report.results) == planned

        # Build the expected per-index status set.
        for i, (policy, result) in enumerate(zip(per_call_policy, report.results)):
            if policy == "timeout":
                assert result.status == "timeout", (
                    f"i={i} policy={policy} got={result.status}"
                )
            elif policy == "error":
                assert result.status == "error", (
                    f"i={i} policy={policy} got={result.status}"
                )
            else:
                assert result.status in ("pass", "fail"), (
                    f"i={i} policy={policy} got={result.status}"
                )

    asyncio.run(_run())
