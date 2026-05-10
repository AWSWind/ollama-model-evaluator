"""Unit tests for :mod:`ollama_evaluator.runner.scheduler`.

Happy-path coverage for Task 12.1 + Task 12.2 + Task 12.4:

* A Run with 2 test cases × 2 repetitions against 1 model dispatches
  exactly 4 generate calls, emits ``run-started`` + 4
  ``test-case-completed`` + ``run-completed`` in order, and produces
  a :class:`RunReport` with 4 :class:`TestCaseResult`s all ``pass``.
* :func:`with_retry` retries the configured HTTP statuses and bails
  cleanly on a non-retriable 4xx.
* Preflight emits ``run-failed`` / ``ollama_unreachable`` when the
  Ollama version check raises.
* :func:`compute_tokens_per_second` matches Property 10's rule.

Property-level coverage (Properties 5, 8, 9, 10, 11, 21) lives in
``tests/property/``.
"""

from __future__ import annotations

import httpx
import pytest

from ollama_evaluator.config import RunConfig
from ollama_evaluator.events import (
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    TestCaseCompletedEvent,
)
from ollama_evaluator.metrics import register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult as MResult
from ollama_evaluator.models import PerformanceMetrics
from ollama_evaluator.ollama.errors import OllamaHTTPError
from ollama_evaluator.ollama.types import OllamaModelInfo
from ollama_evaluator.runner.run_state import RunEventBus, RunState
from ollama_evaluator.runner.scheduler import (
    RunScheduler,
    compute_tokens_per_second,
    with_retry,
)
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)
from tests.unit._fakes import FakeOllamaClient, make_chunks, make_http_error


# ---------------------------------------------------------------------------
# Always-pass metric for deterministic scheduler tests
# ---------------------------------------------------------------------------


class _AlwaysPassMetric:
    name = "always-pass"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


@pytest.fixture(autouse=True)
def _register_metric() -> None:
    register_metric(_AlwaysPassMetric())


# ---------------------------------------------------------------------------
# compute_tokens_per_second (Property 10-flavoured unit check)
# ---------------------------------------------------------------------------


class TestComputeTokensPerSecond:
    def test_normal_case(self) -> None:
        # 5 tokens in 500 ms → 10 tok/s.
        assert compute_tokens_per_second(5, 500.0) == 10.0

    def test_none_tokens_returns_none(self) -> None:
        assert compute_tokens_per_second(None, 500.0) is None

    def test_zero_total_ms_returns_none(self) -> None:
        assert compute_tokens_per_second(10, 0.0) is None

    def test_none_total_returns_none(self) -> None:
        assert compute_tokens_per_second(10, None) is None


# ---------------------------------------------------------------------------
# with_retry (Task 12.2)
# ---------------------------------------------------------------------------


class TestWithRetry:
    async def test_retries_on_503_then_succeeds(self) -> None:
        attempts: list[int] = []

        async def fn() -> int:
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise make_http_error(503, "unavailable")
            return 42

        # No real waiting.
        async def no_sleep(_: float) -> None:
            return

        result = await with_retry(fn, max_attempts=3, sleep=no_sleep)
        assert result == 42
        assert len(attempts) == 3

    async def test_raises_after_exhausting_retries(self) -> None:
        async def fn() -> int:
            raise make_http_error(503)

        async def no_sleep(_: float) -> None:
            return

        with pytest.raises(OllamaHTTPError):
            await with_retry(fn, max_attempts=2, sleep=no_sleep)

    async def test_does_not_retry_4xx(self) -> None:
        attempts: list[int] = []

        async def fn() -> int:
            attempts.append(1)
            raise make_http_error(400, "bad request")

        async def no_sleep(_: float) -> None:
            return

        with pytest.raises(OllamaHTTPError):
            await with_retry(fn, max_attempts=3, sleep=no_sleep)
        assert len(attempts) == 1

    async def test_does_not_retry_timeout(self) -> None:
        attempts: list[int] = []

        async def fn() -> int:
            attempts.append(1)
            raise httpx.ReadTimeout("timeout", request=None)  # type: ignore[arg-type]

        async def no_sleep(_: float) -> None:
            return

        with pytest.raises(httpx.TimeoutException):
            await with_retry(fn, max_attempts=3, sleep=no_sleep)
        assert len(attempts) == 1

    async def test_max_attempts_zero_means_no_retry(self) -> None:
        attempts: list[int] = []

        async def fn() -> int:
            attempts.append(1)
            raise make_http_error(503)

        async def no_sleep(_: float) -> None:
            return

        with pytest.raises(OllamaHTTPError):
            await with_retry(fn, max_attempts=0, sleep=no_sleep)
        assert len(attempts) == 1


# ---------------------------------------------------------------------------
# Happy-path scheduler (Task 12.1)
# ---------------------------------------------------------------------------


def _make_suite() -> EvaluationSuite:
    return EvaluationSuite(
        name="suite-a",
        test_cases=[
            TestCase(
                id="tc1",
                prompt="say hi",
                metrics=[MetricConfig(name="always-pass")],
            ),
            TestCase(
                id="tc2",
                prompt="say bye",
                metrics=[MetricConfig(name="always-pass")],
            ),
        ],
    )


class TestSchedulerHappyPath:
    async def test_dispatches_model_x_cases_x_reps_and_emits_events(self) -> None:
        suite = _make_suite()
        run_config = RunConfig(
            models=["llama3:8b"],
            suites=["suite-a"],
            repetitions=2,
            concurrency=2,
        )

        state = RunState(run_id="run-1")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models(
            [OllamaModelInfo(name="llama3:8b", digest="sha256:abc", parameter_size="8B")]
        )
        fake.set_generate_chunks(make_chunks("hello world", model="llama3:8b"))

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )

        report = await scheduler.execute()

        # 1 model × 2 test cases × 2 repetitions = 4 generate calls.
        assert len(fake.dispatch_log) == 4
        # Concurrency bound respected.
        assert fake.concurrency_observed <= 2

        # 4 per-execution results, all pass.
        assert len(report.results) == 4
        assert [r.status for r in report.results] == ["pass"] * 4

        # Every result carries timing + token counts from the fake's
        # default final chunk.
        for r in report.results:
            assert r.performance.prompt_tokens == 3
            assert r.performance.response_tokens == 5
            assert r.performance.ttft_ms is not None
            assert r.performance.tokens_per_second is not None

        # Event stream ordering: run-started, 4 test-case-completed,
        # run-completed.
        events = state.events
        assert isinstance(events[0], RunStartedEvent)
        assert events[0].planned_executions == 4
        tc_events = [e for e in events if isinstance(e, TestCaseCompletedEvent)]
        assert len(tc_events) == 4
        assert isinstance(events[-1], RunCompletedEvent)
        assert events[-1].summary.passed == 4

        # Per-model ModelInfo populated from the fake's tags.
        assert [m.name for m in report.models] == ["llama3:8b"]
        assert report.models[0].digest == "sha256:abc"
        assert report.models[0].parameter_size == "8B"

        # Suite name propagated to results.
        for r in report.results:
            assert r.suite == "suite-a"

        # Aggregates: 1 ModelAggregate for the single model, 4 passes.
        assert len(report.aggregates) == 1
        assert report.aggregates[0].passed == 4
        assert report.aggregates[0].failed == 0

        # Backend version and Ollama version captured.
        assert report.ollama_version == "0.2.0"
        assert report.backend_version

        # Terminal status consistent.
        assert report.status == "completed"


# ---------------------------------------------------------------------------
# Preflight failure paths (Task 12.4)
# ---------------------------------------------------------------------------


class TestPreflight:
    async def test_ollama_unreachable_emits_run_failed(self) -> None:
        suite = _make_suite()
        run_config = RunConfig(
            models=["llama3:8b"],
            suites=["suite-a"],
        )

        state = RunState(run_id="run-2")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version_raise(httpx.ConnectError("cannot connect", request=None))  # type: ignore[arg-type]

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )

        report = await scheduler.execute()

        failures = [e for e in state.events if isinstance(e, RunFailedEvent)]
        assert len(failures) == 1
        assert failures[0].error_code == "ollama_unreachable"
        assert report.status == "failed"
        # No generate calls were made.
        assert fake.dispatch_log == []

    async def test_model_not_found_emits_run_failed(self) -> None:
        suite = _make_suite()
        run_config = RunConfig(
            models=["missing:latest", "also-missing"],
            suites=["suite-a"],
            pull_missing_models=False,
        )

        state = RunState(run_id="run-3")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_models([OllamaModelInfo(name="other:model")])

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )

        report = await scheduler.execute()
        failures = [e for e in state.events if isinstance(e, RunFailedEvent)]
        assert len(failures) == 1
        assert failures[0].error_code == "model_not_found"
        assert "missing:latest" in failures[0].message
        assert "also-missing" in failures[0].message
        assert report.status == "failed"
        assert fake.dispatch_log == []

    async def test_pull_missing_models_attempts_pull(self) -> None:
        suite = _make_suite()
        run_config = RunConfig(
            models=["to-pull"],
            suites=["suite-a"],
            pull_missing_models=True,
            repetitions=1,
        )

        state = RunState(run_id="run-4")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")

        # First call to list_models returns empty; we need to simulate
        # the model becoming available after the pull. We do this by
        # wrapping list_models in a counter.
        pull_calls = {"count": 0}

        original_list_models = fake.list_models

        async def scripted_list_models() -> list[OllamaModelInfo]:
            pull_calls["count"] += 1
            if pull_calls["count"] == 1:
                return []
            return [OllamaModelInfo(name="to-pull", digest="sha256:x", parameter_size="1B")]

        fake.list_models = scripted_list_models  # type: ignore[assignment]

        fake.set_generate_chunks(make_chunks("ok", model="to-pull"))

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )

        report = await scheduler.execute()
        assert "to-pull" in fake.pulled_models
        assert report.status == "completed"
        _ = original_list_models  # silence unused
