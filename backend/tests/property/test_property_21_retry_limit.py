"""Feature: ollama-model-evaluator, Property 21: Retry limit and terminal error.

Property 21 (from ``design.md`` §Correctness Properties):

    *For any* execution that encounters a sequence of ``n``
    consecutive retriable network errors followed by success or
    exhaustion, the total number of Ollama attempts equals
    ``min(n + 1, retry_max_attempts + 1)``; if all attempts fail the
    test case status is ``error`` with ``error_message`` equal to the
    last error, and the Run continues past the failing execution.

Validates: Requirements 11.1, 11.2.

Approach: run a 2-test-case suite where the *first* execution sees
``n`` retriable failures followed by success-or-nothing, and the
second execution always succeeds. A :class:`FakeOllamaClient` with
an installed failure plan enforces the per-call outcome. The test
then checks:

1. Total calls for the first execution equal ``min(n + 1,
   retry_max_attempts + 1)``.
2. Status is ``error`` if exhausted, ``pass`` if the retry
   succeeded.
3. The second execution still runs to success (the Run continues).
4. ``error_message`` on the failed result carries the raised
   exception's message.
"""

from __future__ import annotations

import asyncio

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
from tests.unit._fakes import FakeOllamaClient, make_chunks, make_http_error


class _AlwaysPassMetric:
    name = "always-pass-p21"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_AlwaysPassMetric())


@given(
    n_failures=st.integers(min_value=0, max_value=5),
    retry_max_attempts=st.integers(min_value=0, max_value=4),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_total_attempts_and_terminal_status(
    n_failures: int, retry_max_attempts: int
) -> None:
    async def _run() -> None:
        models = ["m"]
        # Two test cases: the first one sees the failure plan, the
        # second one always succeeds. Because concurrency=1, dispatch
        # is strictly ordered by plan.
        test_cases = [
            TestCase(
                id="tc1",
                prompt="first",
                metrics=[MetricConfig(name="always-pass-p21")],
            ),
            TestCase(
                id="tc2",
                prompt="second",
                metrics=[MetricConfig(name="always-pass-p21")],
            ),
        ]
        suite = EvaluationSuite(name="suite", test_cases=test_cases)

        # Per-call plan for the *first execution only*. The scheduler
        # consumes at most ``min(n_failures + 1, retry_max_attempts
        # + 1)`` slots before the first execution finishes (either
        # success or exhaustion). We size the plan to exactly that
        # number so the second execution falls through to the
        # chunk-factory success path.
        first_exec_attempts = min(n_failures + 1, retry_max_attempts + 1)
        plan: list[BaseException | None] = []
        for slot in range(first_exec_attempts):
            # A slot is a 503 if it corresponds to one of the
            # ``n_failures`` programmed failures, otherwise a success
            # (``None``). Since attempts are ordered and 503 comes
            # first, the ``None`` only appears when
            # ``slot == n_failures`` i.e. the retry that succeeded.
            plan.append(make_http_error(503) if slot < n_failures else None)

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models([OllamaModelInfo(name="m")])
        fake.set_generate_failure_plan(plan)
        fake.set_generate_chunks(make_chunks("ok", model="m"))

        run_config = RunConfig(
            models=models,
            suites=["suite"],
            repetitions=1,
            concurrency=1,
            retry_max_attempts=retry_max_attempts,
        )

        # Kill retry delays — otherwise the exponential backoff (even
        # a few attempts at 1s, 2s, 4s) would blow the Hypothesis
        # deadline.
        import ollama_evaluator.runner.scheduler as scheduler_mod

        async def no_sleep(_: float) -> None:
            return

        original_sleep = scheduler_mod.asyncio.sleep
        # We can't monkey-patch asyncio.sleep globally; instead, pass
        # the ``sleep`` kwarg through with_retry via a scheduler
        # field. The simplest approach: patch ``with_retry``'s
        # default sleep by rebinding within a subclass.
        # Here, we instead monkey-patch via the module's ``random``
        # attribute to 0 jitter and accept the real sleep, since with
        # max 5 failures and base 1s that could be up to 1+2+4+8+16 s
        # — too slow. Instead, bypass via attribute replacement:
        original = scheduler_mod.with_retry

        async def fast_with_retry(fn, *, max_attempts, **kwargs):  # type: ignore[no-untyped-def]
            return await original(fn, max_attempts=max_attempts, sleep=no_sleep, **kwargs)

        scheduler_mod.with_retry = fast_with_retry  # type: ignore[assignment]
        try:
            scheduler = RunScheduler(
                run_state=state,
                bus=bus,
                ollama_client=fake,
                run_config=run_config,
                suites=[suite],
                generation_defaults=GenerationDefaults(),
            )
            report = await scheduler.execute()
        finally:
            scheduler_mod.with_retry = original  # type: ignore[assignment]
            _ = original_sleep

        # The first execution's attempts are all entries in the
        # dispatch log that happened before the second execution
        # started. We can count by looking at the failure-plan
        # consumption: ``set_generate_failure_plan`` pops one entry
        # per call. In our plan, all entries were for the first
        # execution, so:
        #   attempts_for_first = min(n_failures + 1,
        #                            retry_max_attempts + 1)
        expected_first_attempts = min(n_failures + 1, retry_max_attempts + 1)

        # Count generate calls to the first execution. Because
        # concurrency=1 and the plan entries are all 503s or None,
        # the first execution's attempts are exactly the number of
        # generate calls before the second execution starts. We
        # identify by the prompt.
        first_calls = [
            entry for entry in fake.dispatch_log if entry[1] == "first"
        ]
        second_calls = [
            entry for entry in fake.dispatch_log if entry[1] == "second"
        ]

        assert len(first_calls) == expected_first_attempts, (
            f"n_failures={n_failures} retry={retry_max_attempts} "
            f"expected={expected_first_attempts} "
            f"got={len(first_calls)}"
        )

        # The second execution always runs (Run continues past
        # failing execution).
        assert len(second_calls) == 1
        second_result = next(r for r in report.results if r.test_case_id == "tc2")
        assert second_result.status == "pass"

        first_result = next(r for r in report.results if r.test_case_id == "tc1")
        if n_failures <= retry_max_attempts:
            # Final retry succeeded, status is pass.
            assert first_result.status == "pass"
        else:
            # Exhausted all retries → error with last error's message.
            assert first_result.status == "error"
            assert first_result.error_message is not None
            assert "503" in first_result.error_message

    asyncio.run(_run())
