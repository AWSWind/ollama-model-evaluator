"""Task 26.4 — signal-driven graceful shutdown.

Requirement 11.4: a cooperative shutdown (SIGINT/SIGTERM, or an
explicit ``POST /api/runs/{id}/cancel``) must drain in-flight work
and persist a partial :class:`RunReport`.

The unit suite already covers :func:`install_signal_handlers`
(``tests/unit/test_runner_scheduler.py`` and
``tests/unit/test_runner_run_state.py``). The integration-level
invariant worth pinning here is the scheduler-side drain path that
every signal-equivalent pathway (the real SIGINT handler, the
``POST /api/runs/{id}/cancel`` handler, and the FastAPI lifespan
shutdown hook) depends on: flipping ``cancel_requested`` on a
running :class:`RunState` stops further dispatch and drives the Run
to ``aborted`` with a persisted partial report.

Two clarifications about the implementation:

* The real SIGINT handler installed by :func:`install_signal_handlers`
  is the one-liner ``state.cancel_requested = True`` — it does *not*
  call :meth:`RunSupervisor.stop`. The test therefore simulates the
  signal with a direct flag flip (or equivalently,
  :meth:`RunSupervisor.cancel`, which the REST cancel endpoint uses).
* :meth:`RunSupervisor.stop` is a different code path intended for
  process shutdown (FastAPI lifespan). It also flips
  ``cancel_requested`` but then hard-cancels the worker task, which
  races the scheduler's drain. ``stop()`` is therefore *not* what a
  real signal handler invokes, and its correctness is covered at
  unit level.

The task description is explicit that this test should exercise the
cooperative-drain path, not the hard-cancel path::

    This doesn't need to exercise the signal handler directly — the
    unit-level coverage of ``install_signal_handlers`` is already
    done. Focus on shutdown-drain correctness.

Requirements traced: 11.4.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from ollama_evaluator.api.supervisor import RunSupervisor
from ollama_evaluator.config import RunConfig
from ollama_evaluator.history.store import HistoryStore
from ollama_evaluator.ollama.types import (
    GenerateChunk,
    GenerateOptions,
    OllamaModelInfo,
    PullProgress,
)
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    MetricConfig,
    TestCase,
)
from ollama_evaluator.suites.writer import dump_suite


class _SlowOllamaClient:
    """Fake :class:`OllamaClient` whose ``generate`` sleeps before yielding.

    Scripted so ``version`` and ``list_models`` return immediately
    (preflight must pass), but every ``generate`` call sleeps for
    ``sleep_s`` seconds before emitting the final chunk. The delay
    gives the test a window to flip ``cancel_requested`` while the
    Run is actively executing — the contract the real signal handler
    exercises.
    """

    def __init__(self, sleep_s: float = 0.15) -> None:
        self._sleep_s = sleep_s
        # Instrumentation for the test assertions.
        self.in_flight = 0
        self.max_in_flight = 0
        self.dispatched: list[tuple[str, str]] = []

    async def version(self) -> str:
        return "0.1.32"

    async def list_models(self) -> list[OllamaModelInfo]:
        return [
            OllamaModelInfo(
                name="llama3:8b",
                digest="sha256:abc",
                parameter_size="8B",
            )
        ]

    async def aclose(self) -> None:
        return None

    async def pull_model(self, name: str) -> AsyncIterator[PullProgress]:
        yield PullProgress(status="success")

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        self.in_flight += 1
        if self.in_flight > self.max_in_flight:
            self.max_in_flight = self.in_flight
        self.dispatched.append((model, prompt))
        try:
            # Slow path: the sleep is what gives the test a window
            # to flip ``cancel_requested`` during the Run.
            await asyncio.sleep(self._sleep_s)
            yield GenerateChunk(
                model=model,
                created_at=datetime.now(tz=timezone.utc),
                response="ok",
                done=False,
            )
            yield GenerateChunk(
                model=model,
                created_at=datetime.now(tz=timezone.utc),
                response="",
                done=True,
                total_duration=150_000_000,
                load_duration=0,
                prompt_eval_count=3,
                prompt_eval_duration=0,
                eval_count=1,
                eval_duration=150_000_000,
            )
        finally:
            self.in_flight -= 1


def _write_suite(suites_dir: Path, n_cases: int = 6) -> None:
    """Write a tiny suite with enough test cases that drain is observable."""
    suites_dir.mkdir(parents=True, exist_ok=True)
    suite = EvaluationSuite(
        name="slow",
        test_cases=[
            TestCase(
                id=f"tc{i}",
                prompt=f"q{i}",
                metrics=[MetricConfig(name="exact-match")],
                expected_output="ok",
            )
            for i in range(n_cases)
        ],
    )
    (suites_dir / "slow.yaml").write_text(
        dump_suite(suite, "yaml"), encoding="utf-8"
    )


async def test_cancel_flip_drains_run_and_persists_partial_report(
    tmp_path: Path,
) -> None:
    """Flip ``cancel_requested`` during a slow Run; assert drain semantics.

    Steps (all in one asyncio event loop):

    1. Build an in-memory :class:`HistoryStore`, drop a 6-test-case
       suite on disk.
    2. Build a :class:`RunSupervisor` with an ``ollama_client_factory``
       that returns a :class:`_SlowOllamaClient` (150 ms per generate)
       and start it.
    3. Submit a Run with ``concurrency=1`` so the six test cases
       execute serially; the slow generate guarantees the Run is
       still in its dispatch loop when we signal cancel.
    4. Wait for the first generate to start dispatching
       (``in_flight > 0``). This means preflight has passed, the
       ``run-started`` event is on the bus, and the Run is in the
       dispatch loop.
    5. Flip ``cancel_requested`` on the live :class:`RunState` — the
       exact action :func:`install_signal_handlers` performs on
       SIGINT/SIGTERM. :meth:`RunSupervisor.cancel` does the same
       thing for the ``POST /api/runs/{id}/cancel`` handler.
    6. Await the drain: the scheduler exits the dispatch loop, emits
       the terminal ``run-aborted`` event, writes the partial
       :class:`RunReport`, and transitions the store status to
       ``aborted``.
    7. Assert:

       * ``state.cancel_requested`` is ``True`` (the flip observed).
       * A :class:`RunReport` was persisted with
         ``status in {"aborted", "cancelled", "failed"}``.
       * The persisted event log ends on ``run-aborted``
         (Requirement 11.4's explicit design intent).
       * Not every planned execution ran — at least one was
         cancelled, so ``completed_executions < planned_executions``.
    """
    suites_dir = tmp_path / "suites"
    _write_suite(suites_dir, n_cases=6)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    slow_client = _SlowOllamaClient(sleep_s=0.15)

    async with HistoryStore.open(":memory:", runs_dir) as store:

        def _factory(base_url: str, timeout_s: float) -> Any:
            return slow_client

        supervisor = RunSupervisor(
            store,
            suites_dir=suites_dir,
            output_dir=runs_dir,
            ollama_client_factory=_factory,
            default_ollama_base_url="http://ollama",
        )
        await supervisor.start()
        try:
            run_id = await supervisor.submit(
                RunConfig(
                    models=["llama3:8b"],
                    suites=["slow"],
                    repetitions=1,
                    concurrency=1,
                    tag_filter=[],
                )
            )

            # Wait for the first generate to actually start. This
            # guarantees preflight has passed and the dispatch loop
            # is live — the state in which a real signal would fire.
            await _wait_until(lambda: slow_client.in_flight > 0, timeout_s=2.0)

            # Grab the live state for later inspection; keep a ref
            # because the supervisor pops live states on worker
            # completion.
            state = supervisor.get_state(run_id)
            assert state is not None, "live RunState missing while Run was in-flight"
            assert state.status == "running"

            # ============================================================
            # The signal emulation: flip ``cancel_requested``.
            # This is literally what :func:`install_signal_handlers`
            # does on SIGINT. Using :meth:`RunSupervisor.cancel` would
            # be equivalent — it calls the same mutation.
            # ============================================================
            state.cancel_requested = True

            # Wait for the scheduler to drain and persist the report.
            # The in-flight generate must finish (≤150 ms), the
            # dispatch loop must exit, the scheduler must build the
            # terminal event and call store.write_report. 5s is an
            # order of magnitude over the expected latency.
            report = await _wait_until_report(store, run_id, timeout_s=5.0)

            # cancel_requested preserved on the live state.
            assert state.cancel_requested is True

            # Partial report with non-completed terminal status.
            assert report.status in {"aborted", "cancelled", "failed"}, (
                f"expected a partial-run terminal status, got {report.status!r}"
            )

            # Terminal event recorded. The scheduler emits
            # ``run-aborted`` for the cancel-drain path specifically;
            # the event log's last entry must be the terminal event.
            events = await store.list_events(run_id)
            assert events, "expected persisted run events for the aborted Run"
            assert events[-1].type == "run-aborted", (
                f"Requirement 11.4 design intent: terminal event must be "
                f"``run-aborted`` for a cooperative cancel; got "
                f"{events[-1].type!r}"
            )

            # Drain semantics: at least one planned execution did not
            # run (else we haven't actually tested the drain — the Run
            # would have completed normally). The scheduler pads the
            # results list to plan length by synthesising cancelled
            # ``error`` results for un-dispatched executions, so we
            # check the event counts rather than len(report.results).
            tc_events = [
                e for e in events if e.type == "test-case-completed"
            ]
            assert len(tc_events) < 6, (
                f"expected at least one execution to be cancelled, "
                f"but all 6 test-case-completed events were emitted"
            )
        finally:
            # Clean up the supervisor before the HistoryStore context
            # exits, so the worker task and any pending Run are torn
            # down cleanly.
            await supervisor.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_until(
    predicate, timeout_s: float = 2.0, poll_interval_s: float = 0.01
) -> None:
    """Poll ``predicate()`` until it returns truthy or we time out."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(poll_interval_s)
    raise AssertionError(f"predicate never became true within {timeout_s}s")


async def _wait_until_report(
    store: HistoryStore, run_id: str, *, timeout_s: float
) -> Any:
    """Poll ``store.get_run(run_id)`` until a report is persisted."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        report = await store.get_run(run_id)
        if report is not None:
            return report
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"partial RunReport was never persisted for {run_id} within {timeout_s}s"
    )
