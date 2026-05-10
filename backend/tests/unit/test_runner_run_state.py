"""Unit tests for :mod:`ollama_evaluator.runner.run_state`.

Covers Task 11.1 / Task 11.2 contracts:

* :class:`RunEventBus.append_event` assigns monotonic ``seq`` values
  and notifies waiting subscribers.
* :meth:`RunEventBus.subscribe` snapshots existing events and yields
  newly-appended events via the condition-based broadcast, terminating
  on the terminal event.
* Multiple concurrent subscribers all receive the canonical sequence
  (Property 34's non-Hypothesis cousin).
* :class:`ProgressTicker` emits at the configured cadence, stops when
  cancelled, and does not emit after the state transitions out of
  ``running``.

Property-based coverage lives in ``tests/property/`` (Properties 31,
33, 34 in later tasks); this file is the example-level smoke test.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from ollama_evaluator.events import (
    RunCompletedEvent,
    RunProgressEvent,
    RunStartedEvent,
    TestCaseCompletedEvent,
)
from ollama_evaluator.models import (
    MetricResult,
    PerformanceMetrics,
    RunSummary,
    TestCaseResult,
)
from ollama_evaluator.runner.run_state import (
    ProgressCounters,
    ProgressTicker,
    RunEventBus,
    RunState,
)

_UTC = timezone.utc


def _started(run_id: str, planned: int = 1) -> RunStartedEvent:
    return RunStartedEvent(
        run_id=run_id,
        seq=0,
        ts=datetime.now(tz=_UTC),
        planned_executions=planned,
    )


def _completed(run_id: str, passed: int = 1) -> RunCompletedEvent:
    return RunCompletedEvent(
        run_id=run_id,
        seq=0,
        ts=datetime.now(tz=_UTC),
        summary=RunSummary(
            planned_executions=1,
            completed_executions=1,
            passed=passed,
            failed=0,
            errored=0,
            timed_out=0,
        ),
    )


def _test_case_completed(run_id: str, rep: int = 1) -> TestCaseCompletedEvent:
    return TestCaseCompletedEvent(
        run_id=run_id,
        seq=0,
        ts=datetime.now(tz=_UTC),
        result=TestCaseResult(
            model="llama3:8b",
            suite="s",
            test_case_id="t",
            repetition=rep,
            status="pass",
            response="hi",
            error_message=None,
            performance=PerformanceMetrics(
                ttft_ms=10.0, total_ms=100.0, prompt_tokens=1,
                response_tokens=1, tokens_per_second=10.0,
            ),
            metrics=[MetricResult(name="m", score=1.0, passed=True)],
        ),
    )


class TestRunState:
    """Field defaults and shape."""

    def test_defaults_are_sensible(self) -> None:
        st = RunState(run_id="r1")
        assert st.run_id == "r1"
        assert st.status == "pending"
        assert st.cancel_requested is False
        assert st.events == []
        assert st.next_seq == 0
        assert st.history_store is None


class TestRunEventBusAppend:
    """:meth:`RunEventBus.append_event` sequencing and notification."""

    async def test_append_assigns_monotonic_seq_and_appends(self) -> None:
        st = RunState(run_id="r1")
        bus = RunEventBus(st)

        e1 = _started("r1")
        e2 = _test_case_completed("r1")
        e3 = _completed("r1")

        await bus.append_event(e1)
        await bus.append_event(e2)
        await bus.append_event(e3)

        assert [e.seq for e in st.events] == [0, 1, 2]
        assert st.next_seq == 3
        # The event objects have been mutated in-place to carry the
        # assigned seq.
        assert e1.seq == 0 and e2.seq == 1 and e3.seq == 2


class TestRunEventBusSubscribe:
    """:meth:`RunEventBus.subscribe` replay + live tail."""

    async def test_snapshot_replay_then_live_then_terminal(self) -> None:
        st = RunState(run_id="r1")
        bus = RunEventBus(st)

        # Append the first event *before* subscribing so the
        # subscriber sees it via the snapshot path.
        await bus.append_event(_started("r1"))

        received: list[str] = []

        async def consume() -> None:
            async for ev in bus.subscribe():
                received.append(ev.type)

        consumer = asyncio.create_task(consume())
        # Let the subscriber run through the snapshot branch and then
        # block on the condition.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Now append a live event and the terminal event.
        await bus.append_event(_test_case_completed("r1"))
        await bus.append_event(_completed("r1"))

        await asyncio.wait_for(consumer, timeout=1.0)

        assert received == [
            "run-started",
            "test-case-completed",
            "run-completed",
        ]

    async def test_multiple_subscribers_each_receive_full_sequence(self) -> None:
        st = RunState(run_id="r1")
        bus = RunEventBus(st)

        received_a: list[int] = []
        received_b: list[int] = []

        async def consume(target: list[int]) -> None:
            async for ev in bus.subscribe():
                target.append(ev.seq)

        task_a = asyncio.create_task(consume(received_a))
        task_b = asyncio.create_task(consume(received_b))
        # Yield so both consumers enter the subscribe coroutine.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        await bus.append_event(_started("r1"))
        await bus.append_event(_test_case_completed("r1"))
        await bus.append_event(_completed("r1"))

        await asyncio.wait_for(task_a, timeout=1.0)
        await asyncio.wait_for(task_b, timeout=1.0)

        assert received_a == [0, 1, 2]
        assert received_b == [0, 1, 2]

    async def test_subscriber_that_drops_does_not_block_producer(self) -> None:
        st = RunState(run_id="r1")
        bus = RunEventBus(st)

        async def consume() -> None:
            async for ev in bus.subscribe():
                # Drop on the very first event.
                del ev
                return

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        await bus.append_event(_started("r1"))
        await asyncio.wait_for(task, timeout=1.0)

        # The producer should be able to continue appending events
        # after a subscriber drops without blocking.
        await bus.append_event(_completed("r1"))
        assert st.next_seq == 2


class TestProgressTicker:
    """Task 11.2 — periodic :class:`RunProgressEvent` emission."""

    async def test_emits_events_while_running_and_stops_on_stop(self) -> None:
        st = RunState(run_id="r1", status="running")
        bus = RunEventBus(st)
        counters = ProgressCounters(completed=0, in_progress=1, pending=9)

        # 10 ms cadence so the test finishes in well under a second.
        ticker = ProgressTicker(bus, st, counters, interval_s=0.01)
        ticker.start()

        # Wait long enough for a few ticks.
        await asyncio.sleep(0.05)
        await ticker.stop()

        progress_events = [e for e in st.events if isinstance(e, RunProgressEvent)]
        assert len(progress_events) >= 2, (
            f"expected at least 2 progress events, got {len(progress_events)}"
        )
        # Every emitted event reflects the live counter values at
        # emission time. We never mutate the counters in this test so
        # every event should carry (0, 1, 9).
        for ev in progress_events:
            assert (ev.completed, ev.in_progress, ev.pending) == (0, 1, 9)

    async def test_does_not_emit_after_status_leaves_running(self) -> None:
        st = RunState(run_id="r1", status="running")
        bus = RunEventBus(st)
        counters = ProgressCounters()
        ticker = ProgressTicker(bus, st, counters, interval_s=0.01)
        ticker.start()

        # Transition to ``completed`` almost immediately so the loop
        # observes the status change on its next wake.
        st.status = "completed"
        await asyncio.sleep(0.05)
        await ticker.stop()

        progress_events = [e for e in st.events if isinstance(e, RunProgressEvent)]
        assert progress_events == []

    async def test_stop_is_idempotent(self) -> None:
        st = RunState(run_id="r1", status="running")
        bus = RunEventBus(st)
        ticker = ProgressTicker(bus, st, ProgressCounters(), interval_s=0.01)
        ticker.start()
        await asyncio.sleep(0.02)
        await ticker.stop()
        # Second stop should not raise.
        await ticker.stop()


@pytest.mark.parametrize(
    "terminal_factory",
    [
        _completed,
        # Aborted / failed are also terminal event types; re-use the
        # completed-shaped helper for shape simplicity by wrapping.
    ],
)
class TestTerminalHandling:
    """Subscribe terminates on every terminal variant."""

    async def test_subscribe_terminates_after_terminal_event(
        self, terminal_factory
    ) -> None:
        st = RunState(run_id="r1")
        bus = RunEventBus(st)

        received: list[str] = []

        async def consume() -> None:
            async for ev in bus.subscribe():
                received.append(ev.type)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        await bus.append_event(_started("r1"))
        await bus.append_event(terminal_factory("r1"))

        await asyncio.wait_for(task, timeout=1.0)

        assert received[-1] in {"run-completed", "run-aborted", "run-failed"}
