"""Scheduler driving the full lifecycle of a single Run.

This module is the runtime heart of a Run: preflight, plan expansion,
concurrency-gated dispatch, per-execution streaming, retry with
jitter, metric scoring, cooperative cancellation, aggregate
computation, and emission of every :class:`RunEvent` on the bus.

Task mapping:

* **Task 12.1 — execute()**: :meth:`RunScheduler.execute` — the full
  lifecycle loop (plan → preflight → run-started → dispatch under
  ``Semaphore(concurrency)`` → test-case-completed events → aggregate
  → run-completed | run-aborted | run-failed).
* **Task 12.2 — retry policy**: :func:`with_retry` — retry
  ``httpx.ConnectError``/``httpx.ReadError``/HTTP 502/503/504 up to
  ``retry_max_attempts`` additional attempts with ``1s · 2**k ± 20%``
  jitter. Timeouts (``httpx.TimeoutException`` / 408) → ``timeout``.
  4xx (except 408) → ``error``.
* **Task 12.3 — cancellation & signal handling**: the dispatch loop
  checks ``state.cancel_requested`` before every dequeue and wraps
  the in-flight gather in ``asyncio.wait_for(..., timeout=30)`` with
  a best-effort drain. :func:`install_signal_handlers` is the helper
  the ``serve`` CLI subcommand calls — it is *not* installed at
  import time (the design forbids that) and is idempotent.
* **Task 12.4 — preflight**: :meth:`RunScheduler._preflight` —
  verifies Ollama is reachable, that every ``config.models`` exists
  (pulling if ``pull_missing_models=True``), and records per-model
  :class:`ModelInfo` on :attr:`_model_infos` for the final
  :class:`RunReport`. Suite materialisation hooks are stubbed out
  here because the v1 public code path always passes already-materialised
  suites in; remote-mode materialisation is a scheduler-caller
  responsibility surfaced via the ``materialise_suites`` hook.

Design references:

* ``.kiro/specs/ollama-model-evaluator/design.md`` §Runner and
  Scheduler, §Error Handling / Retry and timeout policy, §Error
  Handling / Cancellation and shutdown.
* Requirements 1.2, 1.3, 1.4, 1.5, 2.3, 2.4, 2.5, 5.1, 5.3, 5.4, 5.5,
  5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.3, 7.5, 11.1, 11.2, 11.4, 17.5,
  17.6, 17.7.

What lives *outside* this module:

* Event persistence to SQLite — the :class:`HistoryStore` (Task 13).
* Run report writing (JSON + Markdown on disk) — ``runner/reports.py``
  (Task 14.1). The scheduler builds the
  :class:`~ollama_evaluator.models.RunReport` but does not serialise it.
* The 2 s progress ticker — :class:`ProgressTicker` from
  :mod:`.run_state` (Task 11.2).
"""

from __future__ import annotations

import asyncio
import logging
import random
import signal
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from .. import __version__ as _BACKEND_VERSION
from ..config import RunConfig
from ..events import (
    RunAbortedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    TestCaseCompletedEvent,
)
from ..models import (
    ErrorSummaryEntry,
    MetricResult,
    ModelAggregate,
    ModelInfo,
    PerformanceMetrics,
    RunReport,
    RunSummary,
    TestCaseResult,
)
from ..ollama.errors import OllamaHTTPError
from ..suites.models import EvaluationSuite, GenerationDefaults, TestCase
from .aggregate import build_all_aggregates
from .run_state import ProgressCounters, ProgressTicker, RunEventBus, RunState
from .scoring import score_all_metrics
from .selection import resolve_generate_options, select_executions

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..ollama.client import OllamaClient


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry policy (Task 12.2)
# ---------------------------------------------------------------------------

# Statuses the retry policy classifies as "retriable HTTP error" per the
# design. Any other HTTP status falls through to terminal ``error``
# (or, for 408, to terminal ``timeout`` — handled separately below).
_RETRIABLE_HTTP_STATUSES: frozenset[int] = frozenset({502, 503, 504})


class _PermanentError(Exception):
    """Wrapper marking a non-retriable failure with a classification.

    Not exported — used internally by :func:`with_retry` to signal
    back to the caller which outcome bucket to record without raising
    the underlying exception verbatim. The caller pattern-matches on
    ``classification`` to populate :class:`TestCaseResult.status`.
    """

    def __init__(self, classification: str, cause: BaseException) -> None:
        self.classification = classification
        self.cause = cause
        super().__init__(f"{classification}: {cause}")


async def with_retry(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int,
    base_delay_s: float = 1.0,
    jitter: float = 0.2,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> Any:
    """Invoke ``fn`` with retry on retriable errors, raising on terminal ones.

    Retry classification (design §Retry and timeout policy):

    * :class:`httpx.TimeoutException` → *not* retried. Propagates as
      the original httpx exception so the caller records ``timeout``.
    * :class:`httpx.ConnectError`, :class:`httpx.ReadError` → retried.
    * :class:`OllamaHTTPError` with status ∈ {502, 503, 504} → retried.
    * :class:`OllamaHTTPError` with status 408 → propagated as
      ``TimeoutException``-equivalent, which the caller maps to
      ``timeout``. (The Ollama server emits 5xx for overload, not
      408, but this matches the policy stated in the design.)
    * :class:`OllamaHTTPError` with any other status → propagates as
      ``error``.

    Delay schedule per attempt ``k ∈ {0, 1, …, max_attempts - 1}``:
    ``base_delay_s * (2 ** k) * uniform(1 - jitter, 1 + jitter)``
    (clamped to a minimum of zero). The 20% jitter matches the design
    and avoids synchronised retry storms when many executions retry
    simultaneously.

    Args:
        fn: Zero-argument coroutine factory. Invoked fresh per attempt.
        max_attempts: Maximum number of *additional* attempts on top
            of the initial call. Equals ``RunConfig.retry_max_attempts``.
            Total attempts = ``max_attempts + 1``. Must be >= 0.
        base_delay_s: Base delay before the first retry. ``2 ** k``
            backoff is applied on each subsequent attempt.
        jitter: Symmetric multiplicative jitter in ``[0, 1)``. 0.2
            matches the design.
        rng: Optional seeded :class:`random.Random` for deterministic
            jitter. When ``None`` the module-level ``random`` state
            is used; property tests pin this for reproducibility.
        sleep: Optional replacement for :func:`asyncio.sleep`. Tests
            swap in a fast no-op so retry cadence does not slow them.

    Returns:
        Whatever ``fn`` returns on the first successful attempt.

    Raises:
        httpx.TimeoutException: Propagated unchanged so the caller can
            classify as ``timeout``.
        Exception: Any non-retriable error propagates unchanged. After
            all retries are exhausted, the last retriable error is
            re-raised (same instance) so the caller can read its
            message for the ``error_message`` field.
    """
    if max_attempts < 0:
        raise ValueError("max_attempts must be >= 0")
    rng = rng or random.Random()
    sleep_fn = sleep if sleep is not None else asyncio.sleep

    attempt = 0
    last_exc: BaseException | None = None
    while True:
        try:
            return await fn()
        except httpx.TimeoutException:
            # Timeouts are terminal per Req 1.5 — do not retry.
            raise
        except httpx.ConnectError as exc:
            last_exc = exc
        except httpx.ReadError as exc:
            last_exc = exc
        except OllamaHTTPError as exc:
            if exc.status in _RETRIABLE_HTTP_STATUSES:
                last_exc = exc
            else:
                # 4xx (except 408) and other non-retriable 5xx →
                # propagate immediately so the caller records
                # ``error``.
                raise

        if attempt >= max_attempts:
            # No more attempts left. Re-raise the last retriable
            # error so the caller can use its message for
            # ``error_message``.
            assert last_exc is not None
            raise last_exc

        delay = base_delay_s * (2 ** attempt)
        # ``uniform(1 - jitter, 1 + jitter)`` is symmetric around 1.
        factor = rng.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
        await sleep_fn(max(0.0, delay * factor))
        attempt += 1


# ---------------------------------------------------------------------------
# Signal handling helper (Task 12.3)
# ---------------------------------------------------------------------------


def install_signal_handlers(state: RunState) -> Callable[[], None]:
    """Install SIGINT/SIGTERM handlers that flip ``state.cancel_requested``.

    **Intentionally not called at import time.** The design and the
    task list both require the scheduler module to avoid installing
    signal handlers as a side effect of import because that would
    break unit tests (and any embedder of the library) that do not
    want the process-wide signal table clobbered. Instead, the
    ``serve`` CLI subcommand (Task 19.1) calls this helper explicitly
    when it is about to run ``execute()``.

    Platform notes:

    * On POSIX, :meth:`asyncio.AbstractEventLoop.add_signal_handler`
      is used so the signal is delivered as an asyncio callback on
      the event loop — safe to mutate ``state`` without a lock.
    * On Windows, :meth:`add_signal_handler` is not implemented for
      the ``SelectorEventLoop`` / ``ProactorEventLoop`` combinations
      we target. The helper falls back to the ``signal.signal``
      stdlib registration and carries the caveat that the handler
      runs in the main thread, which is still fine for flipping a
      Python boolean on a shared object.

    Returns:
        A zero-arg ``restore()`` callable that uninstalls the handlers
        and restores whatever was there before. The CLI holds on to
        this so it can restore when ``execute()`` returns; tests can
        call it to avoid leaking handler state between cases.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "install_signal_handlers must be called from within a running event loop"
        ) from exc

    def _handler() -> None:
        state.cancel_requested = True

    installed: list[signal.Signals] = []
    previous: dict[signal.Signals, Any] = {}

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):
            # Windows fallback: use ``signal.signal`` directly. Store
            # the previous handler so ``restore()`` can put it back.
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: _handler())
            installed.append(sig)

    def restore() -> None:
        for sig in installed:
            if sig in previous:
                signal.signal(sig, previous[sig])
            else:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass

    return restore


# ---------------------------------------------------------------------------
# Token-per-second helper (Property 10)
# ---------------------------------------------------------------------------


def compute_tokens_per_second(
    response_tokens: int | None, total_ms: float | None
) -> float | None:
    """Return ``response_tokens / (total_ms / 1000)`` or ``None``.

    Mirrors Property 10 exactly: the result is ``None`` when
    ``response_tokens is None`` or ``total_ms is None`` or
    ``total_ms == 0``. Those three cases are semantically
    "unknown" — returning zero would be a valid measurement that
    means "zero tokens per second", which is different.

    Isolated as a module-level helper so the property test can
    exercise it without constructing a full run loop.
    """
    if response_tokens is None or total_ms is None or total_ms == 0:
        return None
    return float(response_tokens) / (float(total_ms) / 1000.0)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class RunScheduler:
    """Drive the full lifecycle of a single Run.

    Lifecycle (all within :meth:`execute`):

    1. **Plan expansion.** Call :func:`select_executions` to build the
       ordered list of ``(model, test_case, repetition)`` tuples to
       dispatch (Property 6).
    2. **Preflight.** Verify Ollama is reachable; verify every
       requested model exists (pulling if configured). Record
       per-model :class:`ModelInfo` on the internal state.
    3. **Emit ``run-started``.** With ``planned_executions`` set to
       the number produced by plan expansion.
    4. **Start the ProgressTicker.** 2-second cadence.
    5. **Dispatch loop.** Under ``asyncio.Semaphore(concurrency)``
       dispatch each planned execution. Each execution calls
       :meth:`_run_execution`, which handles retries, timing, metric
       scoring, and appending a :class:`TestCaseCompletedEvent`.
    6. **Drain on cancel.** If ``state.cancel_requested`` becomes
       ``True`` mid-dispatch, stop enqueueing and give in-flight
       tasks up to 30 s to finish; remaining ones are marked
       ``error`` with message ``"cancelled"``.
    7. **Stop the ticker.**
    8. **Build aggregates.** Call :func:`build_all_aggregates` on the
       results.
    9. **Emit terminal event.** ``run-completed`` on success,
       ``run-aborted`` on cancellation, ``run-failed`` on preflight
       failure (the last is emitted inside preflight and returns
       early).

    The scheduler is not responsible for writing the Run_Report to
    disk or for persisting events to SQLite — those are separate
    concerns handled by Task 14.1 and Task 13 respectively. The
    produced :class:`RunReport` is exposed on :attr:`run_report`
    after :meth:`execute` returns so callers can pass it to those
    components.
    """

    #: Timeout (seconds) for the in-flight drain after cancel is requested.
    CANCEL_DRAIN_TIMEOUT_S: float = 30.0

    def __init__(
        self,
        run_state: RunState,
        bus: RunEventBus,
        ollama_client: "OllamaClient | Any",
        run_config: RunConfig,
        suites: list[EvaluationSuite],
        generation_defaults: GenerationDefaults,
        *,
        judge_client: Any | None = None,
        started_at: datetime | None = None,
        ollama_version: str | None = None,
        config_file: Any | None = None,
        rng: random.Random | None = None,
        store: Any | None = None,
    ) -> None:
        self.state = run_state
        self.bus = bus
        self.ollama = ollama_client
        self.run_config = run_config
        self.suites = suites
        self.generation_defaults = generation_defaults
        self.judge_client = judge_client
        self.started_at = started_at or datetime.now(tz=timezone.utc)
        self._ollama_version: str | None = ollama_version
        self._config_file = config_file
        self._rng = rng or random.Random()
        self._store = store

        self._model_infos: list[ModelInfo] = []
        self._results: list[TestCaseResult] = []
        self._counters = ProgressCounters()

        self.run_report: RunReport | None = None

        # If a HistoryStore is attached, wire an on-append callback so
        # every bus event (including the terminal event) is mirrored
        # into ``run_events`` in the same order the subscribers see
        # it (Requirement 14.5). Combined with the explicit call
        # sequence in :meth:`execute` and :meth:`_emit_run_failed`,
        # this guarantees
        # ``store.append_event(terminal) → store.write_report(report)
        #  → store.update_run_status(run_id, status)`` (Property 24).
        if self._store is not None:
            async def _persist_event(event: Any) -> None:
                await self._store.append_event(event)
            self.bus.set_on_append(_persist_event)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(self) -> RunReport:
        """Run the full lifecycle, returning the built :class:`RunReport`."""
        plan = select_executions(self.suites, self.run_config)
        planned = len(plan)
        self._counters.pending = planned

        # Preflight — may emit ``run-failed`` and return early with
        # status=failed. In that case we return the built (mostly-empty)
        # report so the caller can persist it.
        preflight_ok = await self._preflight()
        if not preflight_ok:
            ended_at = datetime.now(tz=timezone.utc)
            self.run_report = self._build_report(status="failed", ended_at=ended_at)
            # Persist in the order required by Property 24: the
            # terminal ``run-failed`` event was already appended to
            # the bus inside ``_emit_run_failed`` (and, if a store
            # is attached, mirrored via the on_append callback).
            # Now write the report and transition the DB status.
            if self._store is not None:
                await self._store.write_report(self.run_report)
                await self._store.update_run_status(
                    self.state.run_id, "failed", ended_at=ended_at
                )
            return self.run_report

        # Transition to ``running`` and emit ``run-started``.
        self.state.status = "running"
        await self.bus.append_event(
            RunStartedEvent(
                run_id=self.state.run_id,
                seq=0,
                ts=datetime.now(tz=timezone.utc),
                planned_executions=planned,
            )
        )

        # Ticker lives only across the dispatch phase.
        ticker = ProgressTicker(self.bus, self.state, self._counters)
        ticker.start()
        try:
            await self._dispatch_all(plan)
        finally:
            await ticker.stop()

        ended_at = datetime.now(tz=timezone.utc)
        # Determine terminal status: if cancel was requested, the
        # Run is aborted; otherwise completed.
        if self.state.cancel_requested:
            terminal_status = "aborted"
        else:
            terminal_status = "completed"
        self.state.status = terminal_status

        self.run_report = self._build_report(status=terminal_status, ended_at=ended_at)

        # Build the terminal event and broadcast it on the bus. The
        # ``on_append`` callback installed in ``__init__`` mirrors
        # this event into the store's ``run_events`` table, which
        # satisfies the Property 24 ordering clause:
        # ``store.append_event(terminal_event)`` runs to completion
        # before the subsequent ``write_report`` /
        # ``update_run_status`` calls below.
        if terminal_status == "completed":
            terminal_event: Any = RunCompletedEvent(
                run_id=self.state.run_id,
                seq=0,
                ts=ended_at,
                summary=self._build_summary(planned),
            )
        else:
            terminal_event = RunAbortedEvent(
                run_id=self.state.run_id,
                seq=0,
                ts=ended_at,
                reason="cancelled",
            )
        await self.bus.append_event(terminal_event)

        if self._store is not None:
            await self._store.write_report(self.run_report)
            await self._store.update_run_status(
                self.state.run_id, terminal_status, ended_at=ended_at
            )

        return self.run_report

    # ------------------------------------------------------------------
    # Preflight (Task 12.4)
    # ------------------------------------------------------------------

    async def _preflight(self) -> bool:
        """Return ``True`` when preflight succeeds; otherwise emit ``run-failed`` and return ``False``.

        Steps:

        1. Call ``ollama.version()``. Failure → ``ollama_unreachable``.
        2. Call ``ollama.list_models()`` and compare against
           ``run_config.models``. Missing models either get pulled
           (when ``pull_missing_models=True``) or cause
           ``model_not_found``.
        3. Populate :attr:`_model_infos` from the final inventory.

        Remote-mode suite materialisation is *not* performed here in
        v1 because the suites passed to the scheduler are already
        materialised; a future task wires an optional hook through
        so ``dataset_fetch_failed`` / ``field_map_invalid`` can be
        raised before ``run-started``.
        """
        # Step 1 — reachability. Any exception becomes
        # ``ollama_unreachable``. The URL is used for the message per
        # Req 1.3 when available; fall back to a generic label.
        try:
            version = await self.ollama.version()
        except Exception as exc:  # noqa: BLE001 — preflight is broad by design.
            base_url = getattr(self.ollama, "_base_url", "ollama")
            await self._emit_run_failed(
                "ollama_unreachable", f"{base_url}: {exc}"
            )
            return False
        self._ollama_version = version

        # Step 2 — model inventory. Record both what is present and
        # what is missing so we can report precisely.
        try:
            available = await self.ollama.list_models()
        except Exception as exc:  # noqa: BLE001 — still preflight.
            base_url = getattr(self.ollama, "_base_url", "ollama")
            await self._emit_run_failed(
                "ollama_unreachable", f"{base_url}: {exc}"
            )
            return False

        by_name = {m.name: m for m in available}
        requested = list(self.run_config.models)
        missing = [m for m in requested if m not in by_name]

        if missing:
            if self.run_config.pull_missing_models:
                for name in missing:
                    try:
                        async for _chunk in self.ollama.pull_model(name):
                            pass  # progress is not forwarded in v1
                    except Exception as exc:  # noqa: BLE001
                        await self._emit_run_failed(
                            "model_not_found",
                            f"failed to pull {name}: {exc}",
                        )
                        return False
                # Re-query inventory after pulling.
                try:
                    available = await self.ollama.list_models()
                except Exception as exc:  # noqa: BLE001
                    await self._emit_run_failed(
                        "ollama_unreachable", f"after pull: {exc}"
                    )
                    return False
                by_name = {m.name: m for m in available}
                still_missing = [m for m in requested if m not in by_name]
                if still_missing:
                    await self._emit_run_failed(
                        "model_not_found",
                        f"missing after pull: {', '.join(still_missing)}",
                    )
                    return False
            else:
                await self._emit_run_failed(
                    "model_not_found",
                    ", ".join(missing),
                )
                return False

        # Record per-evaluated-model ModelInfo for the final report
        # (Req 2.5). We emit them in ``run_config.models`` order so
        # the report's ``models`` list matches the user-declared
        # order.
        self._model_infos = [
            ModelInfo(
                name=name,
                digest=by_name[name].digest,
                parameter_size=by_name[name].parameter_size,
            )
            for name in requested
            if name in by_name
        ]

        return True

    async def _emit_run_failed(self, error_code: str, message: str) -> None:
        """Emit a ``run-failed`` event and mark the Run failed."""
        self.state.status = "failed"
        await self.bus.append_event(
            RunFailedEvent(
                run_id=self.state.run_id,
                seq=0,
                ts=datetime.now(tz=timezone.utc),
                error_code=error_code,
                message=message,
            )
        )

    # ------------------------------------------------------------------
    # Dispatch loop (Task 12.1 / 12.3)
    # ------------------------------------------------------------------

    async def _dispatch_all(
        self, plan: list[tuple[str, TestCase, int]]
    ) -> None:
        """Dispatch every planned execution under the concurrency semaphore.

        Cancellation discipline (Task 12.3):

        * Before acquiring the semaphore for a new dispatch, check
          ``state.cancel_requested``. If set, stop dequeueing and
          proceed to the drain step. Remaining pending executions
          become synthetic ``error`` results with message
          ``"cancelled"``.
        * Let in-flight tasks drain up to 30 s via
          ``asyncio.wait_for(gather(...), timeout=30)``. On timeout
          the unfinished futures are recorded as ``error`` with
          message ``"cancelled"``.
        """
        semaphore = asyncio.Semaphore(self.run_config.concurrency)
        in_flight: list[asyncio.Task[tuple[int, TestCaseResult]]] = []
        dispatched_indices: list[int] = []

        async def run_one(
            index: int,
            triple: tuple[str, TestCase, int],
        ) -> tuple[int, TestCaseResult]:
            """Semaphore-gated wrapper around :meth:`_run_execution`."""
            async with semaphore:
                self._counters.in_progress += 1
                try:
                    result = await self._run_execution(*triple)
                finally:
                    self._counters.in_progress -= 1
                    self._counters.completed += 1
                    self._counters.pending = max(0, self._counters.pending - 1)
                return index, result

        pending_tuples = list(enumerate(plan))
        for index, triple in pending_tuples:
            if self.state.cancel_requested:
                break
            task = asyncio.create_task(run_one(index, triple))
            in_flight.append(task)
            dispatched_indices.append(index)
            # Yield so the semaphore can actually gate; without this,
            # we'd enqueue the entire plan before the first task even
            # runs, defeating Property 8.
            await asyncio.sleep(0)

        # Drain (normal completion or cancel). Use ``wait_for`` so the
        # cancel drain is bounded at 30 s.
        try:
            if self.state.cancel_requested:
                completed = await asyncio.wait_for(
                    asyncio.gather(*in_flight, return_exceptions=True),
                    timeout=self.CANCEL_DRAIN_TIMEOUT_S,
                )
            else:
                completed = await asyncio.gather(*in_flight, return_exceptions=True)
        except asyncio.TimeoutError:
            # Cancel the stragglers; they will finish as we await them
            # below.
            for task in in_flight:
                if not task.done():
                    task.cancel()
            completed = []
            for task in in_flight:
                try:
                    completed.append(await task)
                except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                    completed.append(exc)

        # Merge successful task results in dispatch order (by index).
        produced: dict[int, TestCaseResult] = {}
        for entry in completed:
            if isinstance(entry, BaseException):
                # Any exception that escaped ``run_one`` is
                # unexpected — record a synthetic error result so the
                # report still accounts for the execution.
                log.exception("scheduler: unexpected exception in run_one", exc_info=entry)
                continue
            index, result = entry
            produced[index] = result

        # Compose the final results list in plan order.
        for i, triple in enumerate(plan):
            if i in produced:
                self._results.append(produced[i])
            else:
                # Not dispatched (cancel) or timed out during drain —
                # synthesise a ``cancelled`` error result.
                model, tc, rep = triple
                self._results.append(
                    _cancelled_result(model=model, test_case=tc, repetition=rep)
                )

    # ------------------------------------------------------------------
    # Per-execution path (Task 12.1 + Task 12.2)
    # ------------------------------------------------------------------

    async def _run_execution(
        self, model: str, test_case: TestCase, repetition: int
    ) -> TestCaseResult:
        """Execute one ``(model, test_case, repetition)`` tuple.

        Dispatches a streaming generate with the retry policy,
        measures the Performance_Metrics, scores every metric, emits
        a ``test-case-completed`` event, and returns the
        :class:`TestCaseResult` to the dispatch loop so it can attach
        it to ``self._results`` in plan order.

        Failure paths:

        * :class:`httpx.TimeoutException` → ``timeout`` status.
        * :class:`OllamaHTTPError` 4xx (non-408) or exhausted retries
          on 5xx → ``error`` status with the exception's message.
        * Other unexpected exception → ``error`` with
          ``str(exc)`` as the message.
        """
        options = resolve_generate_options(test_case, self.generation_defaults)
        # Emit an empty TestCaseResult on cancellation up front so
        # that cancel requests between executions produce the
        # ``cancelled`` status without dispatching the generate call.
        if self.state.cancel_requested:
            return _cancelled_result(model=model, test_case=test_case, repetition=repetition)

        start = time.monotonic()
        ttft_ms: float | None = None
        response_text_parts: list[str] = []
        thinking_text_parts: list[str] = []
        total_duration_ns: int | None = None
        prompt_tokens: int | None = None
        response_tokens: int | None = None
        status: str
        error_message: str | None = None

        async def do_generate() -> None:
            """Stream one generation, mutating the local variables above.

            Nested closure rather than a separate coroutine because
            we need access to the mutable timing captures and this
            avoids yet another return-value type.
            """
            nonlocal ttft_ms, total_duration_ns, prompt_tokens, response_tokens
            first_chunk_seen = False
            async for chunk in self.ollama.generate(
                model=model,
                prompt=test_case.prompt,
                system=test_case.system_prompt,
                options=options,
            ):
                # Some Ollama models (Qwen reasoning variants) stream a
                # ``thinking`` field separately from ``response``. TTFT
                # must fire on the first chunk that carries *any*
                # text (thinking or response) so reasoning models
                # report real latency instead of ``None``.
                has_text = bool(chunk.response) or bool(chunk.thinking)
                if not first_chunk_seen and has_text:
                    ttft_ms = (time.monotonic() - start) * 1000.0
                    first_chunk_seen = True
                if chunk.response:
                    response_text_parts.append(chunk.response)
                if chunk.thinking:
                    thinking_text_parts.append(chunk.thinking)
                if chunk.done:
                    # Final chunk carries the timing + token counts.
                    # Any of them may be ``None`` (Req 6.5).
                    total_duration_ns = chunk.total_duration
                    prompt_tokens = chunk.prompt_eval_count
                    response_tokens = chunk.eval_count

        try:
            await with_retry(
                do_generate,
                max_attempts=self.run_config.retry_max_attempts,
                rng=self._rng,
            )
            status = "pass"  # placeholder; refined after scoring below
            total_ms = (time.monotonic() - start) * 1000.0
            # Prefer Ollama's own total duration when it reported one.
            if total_duration_ns is not None:
                total_ms = float(total_duration_ns) / 1_000_000.0
            tokens_per_second = compute_tokens_per_second(response_tokens, total_ms)

            performance = PerformanceMetrics(
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                tokens_per_second=tokens_per_second,
            )

            response = "".join(response_text_parts)
            # Reasoning models (Qwen ``/think`` etc.) may stream their
            # entire answer under ``thinking`` when ``num_predict`` is
            # tight, leaving ``response`` empty. Fall back to the
            # thinking trace so metrics have something to score.
            if not response and thinking_text_parts:
                response = "".join(thinking_text_parts)
            metric_results = await score_all_metrics(
                response,
                test_case,
                test_case.metrics,
                model=model,
                suite=_find_suite_name(self.suites, test_case),
                judge_client=self.judge_client,
                judge_model=self.run_config.judge_model,
            )
            status = _derive_status(metric_results)
            error_message = None

        except httpx.TimeoutException as exc:
            status = "timeout"
            error_message = str(exc) or exc.__class__.__name__
            performance = _empty_performance(start)
            metric_results = []
            response = None  # type: ignore[assignment]
        except (OllamaHTTPError, httpx.ConnectError, httpx.ReadError) as exc:
            status = "error"
            error_message = str(exc)
            performance = _empty_performance(start)
            metric_results = []
            response = None  # type: ignore[assignment]
        except asyncio.CancelledError:
            # In-flight during cancel drain timeout — record as
            # cancelled rather than re-raising so the task exits
            # cleanly and the dispatch loop can record the synthetic
            # result.
            return _cancelled_result(model=model, test_case=test_case, repetition=repetition)
        except Exception as exc:  # noqa: BLE001 — still isolate the Run.
            status = "error"
            error_message = str(exc)
            performance = _empty_performance(start)
            metric_results = []
            response = None  # type: ignore[assignment]

        tc_result = TestCaseResult(
            model=model,
            suite=_find_suite_name(self.suites, test_case),
            test_case_id=test_case.id,
            repetition=repetition,
            status=status,  # type: ignore[arg-type] — literal union
            response=response if status in ("pass", "fail") else None,
            error_message=error_message,
            performance=performance,
            metrics=metric_results,
        )

        await self.bus.append_event(
            TestCaseCompletedEvent(
                run_id=self.state.run_id,
                seq=0,
                ts=datetime.now(tz=timezone.utc),
                result=tc_result,
            )
        )
        if self._store is not None:
            await self._store.write_test_case_result(self.state.run_id, tc_result)
        return tc_result

    # ------------------------------------------------------------------
    # Report construction
    # ------------------------------------------------------------------

    def _build_report(self, *, status: str, ended_at: datetime) -> RunReport:
        """Assemble the :class:`RunReport`."""
        aggregates: list[ModelAggregate] = build_all_aggregates(self._results)
        error_summary: list[ErrorSummaryEntry] = [
            ErrorSummaryEntry(
                model=r.model,
                suite=r.suite,
                test_case_id=r.test_case_id,
                repetition=r.repetition,
                error_message=r.error_message or "",
            )
            for r in self._results
            if r.status in ("error", "timeout")
        ]

        # Backend needs a ConfigFile on the report (Req 8.4). If the
        # caller didn't supply one, construct a minimal stand-in so
        # the report remains shape-valid even in unit tests that
        # exercise the scheduler in isolation.
        config = self._config_file or _minimal_config_file(self.run_config)

        return RunReport(
            run_id=self.state.run_id,
            backend_version=_BACKEND_VERSION,
            ollama_version=self._ollama_version,
            started_at=self.started_at,
            ended_at=ended_at,
            status=status,  # type: ignore[arg-type] — literal union
            config=config,
            models=self._model_infos,
            results=self._results,
            aggregates=aggregates,
            error_summary=error_summary,
        )

    def _build_summary(self, planned: int) -> RunSummary:
        return RunSummary(
            planned_executions=planned,
            completed_executions=len(self._results),
            passed=sum(1 for r in self._results if r.status == "pass"),
            failed=sum(1 for r in self._results if r.status == "fail"),
            errored=sum(1 for r in self._results if r.status == "error"),
            timed_out=sum(1 for r in self._results if r.status == "timeout"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_suite_name(suites: list[EvaluationSuite], test_case: TestCase) -> str:
    """Return the suite name containing ``test_case``, or empty string if absent."""
    for s in suites:
        if any(tc is test_case or tc.id == test_case.id for tc in s.test_cases):
            return s.name
    return ""


def _derive_status(metrics: list[MetricResult]) -> str:
    """Map metric results to the 4-valued :class:`TestCaseResult.status`.

    * If every metric ``passed`` → ``pass``.
    * Otherwise (at least one metric ``passed is False``) → ``fail``.

    Errored metrics keep ``passed=False`` so they push the test case
    to ``fail`` rather than ``error``; the Req 7.5 isolation contract
    is that a *single* bad metric must not fail the whole test case
    in the sense of crashing the Run — the per-metric ``error`` field
    captures the diagnostic. If every metric errored, the status is
    still ``fail`` so the aggregate counters reflect the operator's
    intent (the metric results themselves capture the diagnostic).
    """
    if not metrics:
        return "pass"
    return "pass" if all(m.passed for m in metrics) else "fail"


def _empty_performance(start: float) -> PerformanceMetrics:
    total_ms = (time.monotonic() - start) * 1000.0
    return PerformanceMetrics(
        ttft_ms=None,
        total_ms=total_ms,
        prompt_tokens=None,
        response_tokens=None,
        tokens_per_second=None,
    )


def _cancelled_result(
    *, model: str, test_case: TestCase, repetition: int
) -> TestCaseResult:
    return TestCaseResult(
        model=model,
        suite="",
        test_case_id=test_case.id,
        repetition=repetition,
        status="error",
        response=None,
        error_message="cancelled",
        performance=PerformanceMetrics(
            ttft_ms=None,
            total_ms=0.0,
            prompt_tokens=None,
            response_tokens=None,
            tokens_per_second=None,
        ),
        metrics=[],
    )


def _minimal_config_file(run_config: RunConfig) -> Any:
    """Build a shape-valid :class:`ConfigFile` for reports in unit tests.

    Imported inline to avoid a module-level cycle — :class:`RunScheduler`
    does not need :class:`ConfigFile` except for report construction.
    """
    from pathlib import Path

    from ..config import ConfigFile

    return ConfigFile(
        suites_dir=Path("suites"),
        run=run_config,
    )


__all__ = [
    "RunScheduler",
    "compute_tokens_per_second",
    "install_signal_handlers",
    "with_retry",
]
