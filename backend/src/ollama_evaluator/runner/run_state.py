"""Per-Run in-memory state, event bus, and progress ticker.

This module owns the building blocks the scheduler (Task 12.x) relies
on to thread the lifecycle of a single Run through the rest of the
system:

* :class:`RunState` — the mutable in-memory record of a Run's status,
  cancellation flag, event sequence counter, and an optional reference
  to the persistent :class:`HistoryStore`. The scheduler mutates this
  object, the signal handler (Task 12.3) flips
  ``cancel_requested`` on it, the ticker (below) reads it, and the
  WebSocket handler (Task 18) iterates its event list for replay.

* :class:`RunEventBus` — a thin wrapper that owns the append-only list
  of :class:`RunEvent`s and an :class:`asyncio.Condition` that
  subscribers can wait on. The bus is deliberately tiny: it assigns a
  monotonic ``seq``, appends to an in-memory list, and wakes
  subscribers. It does **not** talk to SQLite — persistence is the
  caller's concern and happens in the same transaction as any state
  change that accompanies the event (Req 14.5). Keeping the bus
  free of the store dependency means unit tests can exercise the
  subscribe/replay path without standing up a database.

* :class:`ProgressTicker` (+ :class:`ProgressCounters`) — the 2-second
  periodic producer of :class:`RunProgressEvent`s (Req 14.4,
  Property 33). Lives here next to :class:`RunEventBus` because the
  scheduler always pairs the two.

Design references:

* ``.kiro/specs/ollama-model-evaluator/design.md`` §Architecture /
  §Event bus and §Concurrency model. The consumer algorithm
  (snapshot → await → replay appended events → stop on terminal) is
  taken directly from the former; the 2-second periodic tick is
  taken from the latter.
* Requirements 14.4 (``run-progress`` cadence ≤ 2 s), 14.5 (exactly
  one terminal event per Run), 14.6 (subscribers receive full replay
  plus live events), 14.7 (a dropped subscriber does not affect
  others or the producer).

Threading model
---------------
All callers live inside a single asyncio event loop (the FastAPI
worker or the CLI ``run`` task). The bus uses ``asyncio.Condition``,
which only cooperates with that loop, so it is deliberately *not*
thread-safe. Property 34 (multi-subscriber replay and isolation) is
therefore a property about coroutines, not OS threads.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from ..events import RunEvent, RunProgressEvent

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from ..history.store import HistoryStore


# The three terminal event ``type`` values that signal the end of a Run.
# ``run-completed``, ``run-aborted``, and ``run-failed`` are all
# terminal; everything else is mid-Run. Kept as a frozenset so
# membership lookup is O(1) inside the subscribe loop.
_TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {"run-completed", "run-aborted", "run-failed"}
)


RunStatus = Literal["pending", "running", "completed", "aborted", "failed"]
"""The five Run state-machine values matching :class:`RunReport.status`."""


@dataclass
class RunState:
    """In-memory record of a single Run's mutable state.

    The dataclass carries only what the runtime path needs — the full
    :class:`RunReport` is built at terminal time and written through
    the :class:`HistoryStore`. Keeping the in-memory footprint small
    means many concurrent Runs (v1 actually caps at one, see the
    design's §Concurrency model, but the shape generalises) do not
    pin an outsized per-Run memory budget just to keep state.

    Attributes:
        run_id: Stable identifier for the Run. Must be non-empty;
            callers typically mint a UUID at ``POST /api/runs`` time.
        status: Current state-machine value. The scheduler transitions
            this through ``pending → running → (completed|aborted|failed)``
            (see the state diagram in ``design.md`` §Run lifecycle).
        cancel_requested: Set ``True`` by
            ``POST /api/runs/{id}/cancel`` or by the SIGINT/SIGTERM
            signal handler (Task 12.3). The scheduler observes this
            between dispatches and stops dequeuing new work.
        events: Append-only event log in ``seq`` order. Owned by the
            :class:`RunEventBus` — :meth:`RunEventBus.append_event`
            mutates this list and wakes waiters on the condition.
        next_seq: Monotonic sequence counter. Starts at ``0`` and is
            incremented by the bus on every append. Stored on the
            state so the bus can fetch it atomically under the
            condition lock.
        history_store: Optional :class:`HistoryStore` for persisting
            the event alongside any state change. The bus does *not*
            use this — callers thread it through as a field so that
            mutating ``status`` and appending the accompanying event
            can happen under a single lock acquired by the caller
            (Property 24, terminal-event-before-status ordering).
    """

    run_id: str
    status: RunStatus = "pending"
    cancel_requested: bool = False
    events: list[RunEvent] = field(default_factory=list)
    next_seq: int = 0
    history_store: "HistoryStore | None" = None


class RunEventBus:
    """Single-Run append-only event bus with asyncio.Condition fan-out.

    One bus per Run. The bus co-owns the Run's :class:`RunState`:
    :meth:`append_event` mutates ``state.events`` and ``state.next_seq``
    under the condition lock, and :meth:`subscribe` reads the events
    list to produce the snapshot-then-live stream described in
    ``design.md`` §Architecture / §Event bus.

    The bus is deliberately free of any persistence coupling (no
    SQLite, no HTTP). The caller — the scheduler (Task 12.1) or the
    preflight step (Task 12.4) — is responsible for persisting the
    event via ``state.history_store.append_event(...)`` when a
    :class:`HistoryStore` is attached. Keeping persistence at the
    call-site means the ``append_event → store.write_report →
    update_run_status`` ordering required by Property 24 is controlled
    in one place.
    """

    def __init__(self, state: RunState) -> None:
        self._state = state
        # ``asyncio.Condition()`` lazily binds to the running event loop
        # on first acquire. Constructing it here (in the sync
        # ``__init__``) does not bind, so a test that constructs a
        # ``RunEventBus`` outside an event loop for later use still
        # works.
        self._cond = asyncio.Condition()
        # Optional async callback invoked after every append. Used by
        # the scheduler (Task 13.3) to mirror every appended event
        # into the :class:`HistoryStore` without coupling the bus to
        # the store. The callback may be sync or async — awaitable
        # return values are awaited by :meth:`append_event`.
        self._on_append: Any | None = None

    @property
    def state(self) -> RunState:
        """The enclosing :class:`RunState`; exposed for callers that want to
        inspect ``events`` or ``status`` without touching the private
        attribute. The returned reference is live — callers should not
        mutate ``events`` directly."""
        return self._state

    async def append_event(self, event: RunEvent) -> None:
        """Append ``event`` to the log under the condition lock and notify waiters.

        Before appending, the event is re-stamped with the next
        per-Run ``seq`` taken from :attr:`RunState.next_seq` so
        producers do not have to track the counter themselves. The
        bus then:

        1. Increments ``state.next_seq``.
        2. Appends to ``state.events``.
        3. Calls ``notify_all`` on the condition so every subscriber
           waiting on a fresh event wakes.

        The persistence contract documented on the module docstring
        holds: this method does not touch
        ``state.history_store``. Callers that need to persist the
        event should do so *before* calling :meth:`append_event` so
        the in-memory and on-disk logs advance together, or under the
        same locking discipline they establish in the scheduler.

        The event is mutated in-place: :attr:`BaseRunEvent.seq` is
        overwritten with the Bus-assigned value. Callers that need to
        keep the pre-append instance should make a copy first.
        """
        async with self._cond:
            event.seq = self._state.next_seq
            self._state.next_seq += 1
            self._state.events.append(event)
            self._cond.notify_all()
        # Callback fires *outside* the condition lock so a slow
        # persistence layer cannot stall new appends or wake-ups.
        if self._on_append is not None:
            result = self._on_append(event)
            if inspect.isawaitable(result):
                await result

    def set_on_append(self, callback: Any | None) -> None:
        """Install ``callback`` to fire for every subsequent append.

        The scheduler uses this to mirror every event into the
        :class:`HistoryStore` (Task 13.3). Passing ``None`` detaches
        an existing callback.
        """
        self._on_append = callback

    async def subscribe(self) -> AsyncIterator[RunEvent]:
        """Async iterator that yields every event in order, replay + live.

        Implements the three-step consumer algorithm from
        ``design.md`` §Event bus:

        1. **Snapshot.** Yield a copy of every event already in the
           log, in order. This guarantees Property 34's "no gaps on
           reconnect" invariant even for subscribers that arrive
           after the Run has already started.
        2. **Live tail.** Loop on ``await condition.wait()`` and
           yield newly-appended events since the last yielded index.
        3. **Stop on terminal.** Exit the loop as soon as a terminal
           event (``run-completed``/``run-aborted``/``run-failed``)
           has been delivered. Combined with Property 31 (event log
           bookends), this guarantees the iterator terminates cleanly.

        A subscriber that drops mid-stream (Req 14.7) simply stops
        iterating. Other subscribers and the producer are unaffected
        because the condition's ``notify_all`` is a one-shot wake — a
        dropped iterator is just a coroutine that never resumes.

        Yields:
            Each :class:`RunEvent` in ``seq`` order, starting from
            ``seq == 0``.
        """
        next_index = 0
        terminal_delivered = False
        while not terminal_delivered:
            # Take the lock, snapshot any pending events, and copy
            # them into a local list so we can yield *outside* the
            # lock. Holding the condition lock across ``yield``
            # would block producers and is not necessary: the
            # events list is append-only and never removes entries,
            # so the index we remembered remains valid.
            async with self._cond:
                if next_index >= len(self._state.events):
                    # Nothing new — wait for the next ``notify_all``.
                    await self._cond.wait()
                pending = list(self._state.events[next_index:])
                next_index += len(pending)

            for ev in pending:
                yield ev
                if ev.type in _TERMINAL_EVENT_TYPES:
                    terminal_delivered = True
                    # Stop yielding after the terminal event even if
                    # more events snuck in (which would be a producer
                    # bug — Property 31 forbids post-terminal events
                    # — but we still honour the iterator contract).
                    return


class ProgressCounters:
    """Mutable counters the scheduler exposes to the :class:`ProgressTicker`.

    A small, plain container rather than a dataclass because the
    scheduler updates its fields directly from multiple coroutines
    (dispatch loop, completion loop) and adding validators would just
    be noise. The three counters partition the plan into the three
    classes tracked by :class:`RunProgressEvent` (Req 14.4):

    * ``completed`` — executions that have produced a result.
    * ``in_progress`` — executions currently dispatched to Ollama.
    * ``pending`` — executions still in the dispatch queue.

    The invariant ``completed + in_progress + pending ==
    planned_executions`` is the scheduler's responsibility; the
    ticker reads the values verbatim and does not re-check.
    """

    __slots__ = ("completed", "in_progress", "pending")

    def __init__(self, completed: int = 0, in_progress: int = 0, pending: int = 0) -> None:
        self.completed = completed
        self.in_progress = in_progress
        self.pending = pending


class ProgressTicker:
    """Emit a :class:`RunProgressEvent` on the bus roughly every ``interval_s`` seconds.

    The ticker is a tiny asyncio task the scheduler starts right after
    the ``run-started`` event and stops just before the terminal event.
    It is the producer half of Property 33 (progress cadence ≤ 2s +
    ε); the consumer half is whatever is reading events off the bus.

    Cancellation discipline:

    * :meth:`stop` cancels the underlying task and awaits it so the
      caller can guarantee no further ``run-progress`` events will be
      emitted after the call returns. This is load-bearing for
      Property 31 (event log bookends): a ``run-progress`` emitted
      after the terminal event would break the "no events follow the
      terminal event" invariant.
    * Calling :meth:`stop` multiple times is a no-op after the first.
    * The ticker never emits after the state transitions out of
      ``running``. It checks ``state.status`` at the top of each loop
      iteration so the last wakeup right before a manual
      :meth:`stop` does not race a status transition.

    The tick interval is configurable (default 2.0 s) only to keep the
    unit tests in ``test_runner_run_state.py`` fast. Production callers
    should use the default.
    """

    def __init__(
        self,
        bus: RunEventBus,
        state: RunState,
        counters: ProgressCounters,
        interval_s: float = 2.0,
    ) -> None:
        self._bus = bus
        self._state = state
        self._counters = counters
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the tick loop on the current event loop.

        Must be called from inside a running event loop. Calling
        :meth:`start` twice without an intervening :meth:`stop` is a
        programming error — the second call is ignored and the
        existing task continues running.
        """
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the tick loop and wait for it to finish.

        Idempotent. After this returns the ticker is guaranteed to
        have stopped emitting events, so the scheduler can safely
        emit the terminal event next without racing a final
        ``run-progress``.
        """
        if self._stopped:
            return
        self._stopped = True
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        """The tick loop; runs until cancelled or until status leaves ``running``.

        On each wake, it first re-checks ``state.status`` — if the
        scheduler has already transitioned out of ``running`` but
        hasn't yet cancelled the ticker (possible during the narrow
        window between ``update_run_status`` and ``stop()``), the
        loop exits without emitting. This gives a second line of
        defence against emitting after the terminal event.
        """
        try:
            while True:
                await asyncio.sleep(self._interval_s)
                if self._state.status != "running":
                    return
                event = RunProgressEvent(
                    run_id=self._state.run_id,
                    # ``seq`` is overwritten by ``append_event`` with
                    # the bus-assigned value; the placeholder here
                    # satisfies Pydantic's ``ge=0`` validator.
                    seq=0,
                    ts=datetime.now(tz=timezone.utc),
                    completed=self._counters.completed,
                    in_progress=self._counters.in_progress,
                    pending=self._counters.pending,
                )
                await self._bus.append_event(event)
        except asyncio.CancelledError:
            # Normal shutdown via :meth:`stop` — swallow the
            # cancellation so :meth:`stop` can ``await`` us without
            # having to handle the exception.
            return


__all__ = [
    "ProgressCounters",
    "ProgressTicker",
    "RunEventBus",
    "RunState",
    "RunStatus",
]
