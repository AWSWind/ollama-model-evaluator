"""Feature: ollama-model-evaluator, Property 11: Optional performance fields.

Property 11 (from ``design.md`` §Correctness Properties):

    *For any* Ollama final-chunk payload missing any subset of
    ``prompt_eval_count``, ``eval_count``, or ``total_duration``, the
    resulting ``PerformanceMetrics`` object is constructed
    successfully, the fields sourced from the missing keys are
    ``None``, and the test case status is not ``error`` solely
    because of the missing fields.

Validates: Requirements 6.3, 6.5.

Approach: drive the full scheduler loop with a :class:`FakeOllamaClient`
scripted to omit an arbitrary subset of the three performance fields
on the final chunk. Assert that the resulting :class:`TestCaseResult`
has ``status != "error"`` and that the missing fields map to ``None``
on :class:`PerformanceMetrics`.
"""

from __future__ import annotations

import asyncio

from datetime import datetime, timezone

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.metrics import register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult as MResult
from ollama_evaluator.ollama.types import GenerateChunk, OllamaModelInfo
from ollama_evaluator.runner.run_state import RunEventBus, RunState
from ollama_evaluator.runner.scheduler import RunScheduler
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)
from tests.unit._fakes import FakeOllamaClient


class _AlwaysPassMetric:
    """Always-pass metric used so ``status`` depends purely on the
    non-error path, not on scoring decisions."""

    name = "always-pass-p11"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_AlwaysPassMetric())


def _build_final_chunk(
    model: str,
    *,
    total_duration: int | None,
    prompt_eval_count: int | None,
    eval_count: int | None,
) -> GenerateChunk:
    """Build a done-chunk with explicitly selected ``None`` fields."""
    return GenerateChunk(
        model=model,
        created_at=datetime.now(tz=timezone.utc),
        response="",
        done=True,
        total_duration=total_duration,
        load_duration=None,
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration=None,
        eval_count=eval_count,
        eval_duration=None,
    )


def _build_partial_chunk(model: str, text: str) -> GenerateChunk:
    return GenerateChunk(
        model=model,
        created_at=datetime.now(tz=timezone.utc),
        response=text,
        done=False,
    )


@given(
    include_total=st.booleans(),
    include_prompt_tokens=st.booleans(),
    include_response_tokens=st.booleans(),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_missing_ollama_fields_yield_none_without_erroring(
    include_total: bool,
    include_prompt_tokens: bool,
    include_response_tokens: bool,
) -> None:
    """Omit an arbitrary subset of timing + token fields on the final
    chunk; the scheduler must record ``None`` for the missing ones
    and not mark the test case as ``error``."""

    async def _run() -> None:
        model = "llama3:8b"
        partial = _build_partial_chunk(model, "hi")
        final = _build_final_chunk(
            model,
            total_duration=1_000_000_000 if include_total else None,
            prompt_eval_count=3 if include_prompt_tokens else None,
            eval_count=5 if include_response_tokens else None,
        )

        suite = EvaluationSuite(
            name="s",
            test_cases=[
                TestCase(
                    id="tc",
                    prompt="say hi",
                    metrics=[MetricConfig(name="always-pass-p11")],
                )
            ],
        )
        run_config = RunConfig(models=[model], suites=["s"], repetitions=1)

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models([OllamaModelInfo(name=model)])
        fake.set_generate_chunks([partial, final])

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )

        report = await scheduler.execute()

        assert len(report.results) == 1
        result = report.results[0]

        # Status never goes to ``error`` solely because of missing
        # Ollama metadata.
        assert result.status != "error", (
            f"unexpected error on missing fields: {result.error_message}"
        )

        perf = result.performance
        if include_prompt_tokens:
            assert perf.prompt_tokens == 3
        else:
            assert perf.prompt_tokens is None
        if include_response_tokens:
            assert perf.response_tokens == 5
        else:
            assert perf.response_tokens is None
        # total_ms is derived; when Ollama omits total_duration the
        # scheduler still records a monotonic-clock delta, so
        # total_ms is never None. tokens_per_second depends on
        # response_tokens — Property 10 covers that relationship.

    asyncio.run(_run())
