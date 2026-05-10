"""Feature: ollama-model-evaluator, Property 5: Missing-model preflight.

Property 5 (from ``design.md`` §Correctness Properties):

    *For any* inventory of Ollama models ``A`` and any requested
    model set ``R``, starting a Run with ``pull_missing_models=false``
    aborts before any ``test-case-completed`` event is emitted if and
    only if ``R \\ A ≠ ∅``; when aborted, the emitted ``run-failed``
    event lists the missing models as exactly ``R \\ A``.

Validates: Requirement 2.3.

Approach: Hypothesis generates disjoint-ish draws of requested and
available model sets. For each draw we spin up a scheduler against a
:class:`FakeOllamaClient` that reports the chosen inventory and
assert:

* If ``R \\ A`` is non-empty, exactly one ``run-failed`` event is
  emitted with ``error_code == "model_not_found"`` and the message
  contains exactly the missing names (no extras, no duplicates).
* If ``R ⊆ A``, no ``run-failed`` event is emitted (``run-started``
  is emitted instead) and at least one ``test-case-completed`` event
  follows.
"""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.events import (
    RunFailedEvent,
    RunStartedEvent,
    TestCaseCompletedEvent,
)
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
    name = "always-pass-p5"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        return MResult(name=self.name, score=1.0, passed=True, threshold=1.0, details={})


register_metric(_AlwaysPassMetric())


_MODEL_NAMES = st.sampled_from(
    ["m1", "m2", "m3", "m4", "m5", "m6"]
)


@given(
    requested=st.lists(_MODEL_NAMES, min_size=1, max_size=4, unique=True),
    available=st.lists(_MODEL_NAMES, min_size=0, max_size=6, unique=True),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_preflight_aborts_iff_requested_minus_available_is_non_empty(
    requested: list[str],
    available: list[str],
) -> None:
    async def _run() -> None:
        missing = [m for m in requested if m not in set(available)]

        suite = EvaluationSuite(
            name="suite",
            test_cases=[
                TestCase(
                    id="tc1",
                    prompt="p",
                    metrics=[MetricConfig(name="always-pass-p5")],
                )
            ],
        )
        run_config = RunConfig(
            models=list(requested),
            suites=["suite"],
            repetitions=1,
            pull_missing_models=False,
        )

        state = RunState(run_id="r")
        bus = RunEventBus(state)
        fake = FakeOllamaClient()
        fake.set_version("0.2.0")
        fake.set_models([OllamaModelInfo(name=m) for m in available])
        fake.set_generate_chunks(make_chunks("ok", model="m"))

        scheduler = RunScheduler(
            run_state=state,
            bus=bus,
            ollama_client=fake,
            run_config=run_config,
            suites=[suite],
            generation_defaults=GenerationDefaults(),
        )
        report = await scheduler.execute()

        failed_events = [e for e in state.events if isinstance(e, RunFailedEvent)]
        started_events = [
            e for e in state.events if isinstance(e, RunStartedEvent)
        ]
        tc_events = [
            e for e in state.events if isinstance(e, TestCaseCompletedEvent)
        ]

        if missing:
            # Preflight aborts: exactly one run-failed, no
            # run-started, no test-case-completed.
            assert len(failed_events) == 1
            assert failed_events[0].error_code == "model_not_found"
            # Every missing model must appear in the message; no
            # extras beyond the missing set.
            for name in missing:
                assert name in failed_events[0].message
            # No non-requested-but-missing names leak into the message.
            non_missing_names = [m for m in requested if m not in missing]
            for name in non_missing_names:
                # A substring check here would false-positive if
                # names share prefixes; the test uses distinct short
                # ids so a plain ``in`` is fine.
                assert name not in failed_events[0].message
            assert started_events == []
            assert tc_events == []
            assert report.status == "failed"
            assert fake.dispatch_log == []
        else:
            # R ⊆ A: preflight passes, run-started emitted, at least
            # one test-case-completed follows.
            assert failed_events == []
            assert len(started_events) == 1
            assert len(tc_events) == len(requested)
            assert report.status == "completed"

    asyncio.run(_run())
