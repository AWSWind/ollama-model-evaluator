"""Feature: ollama-model-evaluator, Property 33: Progress cadence.

Consecutive ``run-progress`` events are separated by no more than
``interval_s + ε`` wall-clock seconds, and the last ``run-progress``
event strictly precedes the terminal event (``seq`` ordering).

Validates: Requirement 14.4.

Approach: exercise the
:class:`~ollama_evaluator.runner.run_state.ProgressTicker` directly
with a short ``interval_s`` so the test stays fast, capture real
timestamps as the ticker fires events, and assert both the inter-event
cadence (≤ 2s budget scaled down to ``interval_s * 5``) and the
relative ordering against a synthetic terminal event appended after
``stop``.
"""

from __future__ import annotations

import asyncio
import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.events import RunCompletedEvent
from ollama_evaluator.models import RunSummary
from ollama_evaluator.runner.run_state import (
    ProgressCounters,
    ProgressTicker,
    RunEventBus,
    RunState,
)


@given(
    n_ticks=st.integers(min_value=2, max_value=4),
    interval_ms=st.integers(min_value=10, max_value=50),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_progress_events_respect_cadence_and_precede_terminal(
    n_ticks: int, interval_ms: int
) -> None:
    """**Validates: Requirement 14.4**"""

    interval_s = interval_ms / 1000.0
    # Budget: cadence must be <= interval_s + epsilon. Allow a generous
    # 5x factor for test-host scheduling jitter.
    tolerance = max(0.25, interval_s * 5.0)

    async def _run() -> tuple[list, list]:
        state = RunState(run_id="r")
        state.status = "running"
        bus = RunEventBus(state)
        counters = ProgressCounters(completed=0, in_progress=0, pending=1)
        ticker = ProgressTicker(bus, state, counters, interval_s=interval_s)

        timestamps: list[float] = []

        async def _observe() -> None:
            async for event in bus.subscribe():
                if event.type == "run-progress":
                    timestamps.append(time.monotonic())
                if event.type in ("run-completed", "run-aborted", "run-failed"):
                    return

        observer = asyncio.create_task(_observe())

        ticker.start()
        # Sleep long enough for n_ticks events to fire.
        await asyncio.sleep(interval_s * n_ticks + interval_s / 2)
        await ticker.stop()
        state.status = "completed"
        await bus.append_event(
            RunCompletedEvent(
                run_id="r",
                seq=0,
                ts=state.events[0].ts if state.events else None,  # type: ignore[arg-type]
                summary=RunSummary(
                    planned_executions=1,
                    completed_executions=0,
                    passed=0,
                    failed=0,
                    errored=0,
                    timed_out=0,
                ),
            )
        )
        await observer
        return timestamps, list(state.events)

    timestamps, events = asyncio.run(_run())

    # At least one progress event must have fired.
    assert timestamps, "no run-progress events were observed"

    # Consecutive ticks obey the cadence budget.
    for earlier, later in zip(timestamps, timestamps[1:]):
        delta = later - earlier
        assert delta <= tolerance, (
            f"cadence violated: {delta:.3f}s between events (budget {tolerance:.3f}s)"
        )

    # seq ordering: last run-progress strictly precedes the terminal event.
    last_progress_seq = None
    terminal_seq = None
    for e in events:
        if e.type == "run-progress":
            last_progress_seq = e.seq
        elif e.type in ("run-completed", "run-aborted", "run-failed"):
            terminal_seq = e.seq
            break

    assert last_progress_seq is not None, "no run-progress in state.events"
    assert terminal_seq is not None, "no terminal event in state.events"
    assert last_progress_seq < terminal_seq, (
        f"last run-progress seq={last_progress_seq} must precede terminal seq={terminal_seq}"
    )
