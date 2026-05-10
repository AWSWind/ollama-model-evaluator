"""Single-run scheduler worker — the back half of ``POST /api/runs``.

The :class:`RunSupervisor` is the glue between the REST handler (which
accepts a :class:`RunConfig` over HTTP) and the :class:`RunScheduler`
(which drives the full Run lifecycle). It owns:

* A tiny in-process queue of pending Runs.
* Exactly one ``running`` slot (Requirement 13.3 — v1 caps concurrent
  Runs at one so Performance_Metrics stay deterministic).
* A map from ``run_id`` to the live :class:`RunState` for the
  duration of a Run, so :meth:`cancel` can flip
  ``cancel_requested`` and the WebSocket endpoint (Task 18.1) can
  look up the live bus for replay.

The supervisor intentionally does *not* own the :class:`HistoryStore`
connection; it holds a reference and calls its async methods. The
handler-time split is:

* ``submit`` runs on the FastAPI worker and **must** persist the Run
  as ``pending`` before returning so the REST response carries a
  real, queryable ``run_id`` (Property 25).
* The worker task, started at :meth:`start` time, dequeues pending
  Runs and invokes :meth:`RunScheduler.execute`. The scheduler itself
  handles the ``pending → running`` transition via the event bus.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ConfigFile, RunConfig
from ..ollama.client import OllamaClient
from ..runner.run_state import RunEventBus, RunState
from ..runner.scheduler import RunScheduler
from ..suites.loader import discover_suites
from ..suites.models import EvaluationSuite, GenerationDefaults

log = logging.getLogger(__name__)


@dataclass
class _PendingRun:
    """Queue item: the minimum info the worker needs to start a Run."""

    run_id: str
    config_file: ConfigFile
    state: RunState
    bus: RunEventBus
    # Pre-merged generation defaults inherited from the first suite
    # the scheduler will execute against. Lazily computed in the
    # worker so ``submit`` does not pay the I/O cost.
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class RunSupervisor:
    """Single-slot queue + worker around :class:`RunScheduler`.

    Lifecycle:

    * :meth:`start` schedules the worker task on the current event
      loop. Call once per FastAPI app startup.
    * :meth:`stop` cancels the worker task and waits for it to exit,
      draining any Run that is currently in-flight by flipping its
      ``cancel_requested`` flag first.
    * :meth:`submit` persists a Run as ``pending`` via the store and
      enqueues it. Returns the minted ``run_id`` so the handler can
      build the 201 response.
    * :meth:`cancel` flips ``cancel_requested`` on the matching
      :class:`RunState` (if any). Returns ``True`` when the run was
      found live, else ``False``.
    * :meth:`get_state` returns the live :class:`RunState` for
      ``run_id`` when available, so the WebSocket endpoint can
      subscribe to its bus and emit a replay.

    Single-slot semantics: the worker processes one Run to completion
    before starting the next. This matches Requirement 13.3 and keeps
    Ollama_Server memory pressure predictable.

    Args:
        store: :class:`HistoryStore` used to mint run ids, persist
            events, and record terminal status transitions.
        suites_dir: Directory the scheduler's preflight uses to
            discover Evaluation_Suites referenced by ``RunConfig.suites``.
        output_dir: Directory where ``runs/<run_id>/report.json`` is
            written. Matches ``ConfigFile.output_dir`` semantics.
        ollama_client_factory: Callable that returns a fresh
            :class:`OllamaClient` given a base URL and timeout. Tests
            inject a factory that returns a :class:`FakeOllamaClient`.
        default_ollama_base_url: ``ConfigFile.ollama_base_url`` to use
            when the submitted :class:`RunConfig` does not override it.
    """

    def __init__(
        self,
        store: Any,
        *,
        suites_dir: Path,
        output_dir: Path,
        ollama_client_factory: Callable[[str, float], Any] | None = None,
        default_ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self._store = store
        self._suites_dir = Path(suites_dir)
        self._output_dir = Path(output_dir)
        self._default_base_url = default_ollama_base_url
        self._client_factory = ollama_client_factory or _default_client_factory

        # Pending queue and live-run registry.
        self._queue: asyncio.Queue[_PendingRun] = asyncio.Queue()
        self._live_states: dict[str, RunState] = {}
        self._live_buses: dict[str, RunEventBus] = {}

        self._worker_task: asyncio.Task[None] | None = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Schedule the worker task if not already running."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Cancel the worker task and any in-flight Run.

        Idempotent. Flips ``cancel_requested`` on every known live
        :class:`RunState` so the scheduler's cancel drain (Task 12.3)
        kicks in, then cancels the worker task and awaits its exit.
        """
        if self._stopped:
            return
        self._stopped = True
        for state in self._live_states.values():
            state.cancel_requested = True
        task = self._worker_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Submission / control
    # ------------------------------------------------------------------

    async def submit(self, run_config: RunConfig) -> str:
        """Persist a new ``pending`` Run and enqueue it for execution.

        Wraps ``run_config`` in a minimal :class:`ConfigFile` using
        the supervisor's stored ``suites_dir``, ``output_dir``, and
        default Ollama base URL. The resulting :class:`ConfigFile` is
        what gets embedded in the eventual :class:`RunReport`
        (Requirement 8.4), so callers that need a richer config
        should add it to :class:`RunConfig` itself.

        Returns:
            The minted ``run_id`` from :meth:`HistoryStore.create_run`.
            Guaranteed to be unique across sequential calls
            (Property 25).
        """
        config_file = ConfigFile(
            ollama_base_url=self._default_base_url,
            suites_dir=self._suites_dir,
            output_dir=self._output_dir,
            run=run_config,
        )
        run_id = await self._store.create_run(config_file)
        state = RunState(run_id=run_id, status="pending")
        bus = RunEventBus(state)

        self._live_states[run_id] = state
        self._live_buses[run_id] = bus

        pending = _PendingRun(
            run_id=run_id,
            config_file=config_file,
            state=state,
            bus=bus,
        )
        await self._queue.put(pending)
        return run_id

    def cancel(self, run_id: str) -> bool:
        """Flip ``cancel_requested`` on the matching :class:`RunState`.

        Returns ``True`` when a live state was found (so the caller
        can return 200), else ``False`` (caller returns 404 /
        ``run_not_found``). Note the flag flip is effective even for
        Runs still in the pending queue — the scheduler checks
        ``cancel_requested`` on its very first action and transitions
        the Run to ``aborted`` without dispatching any generate call.
        """
        state = self._live_states.get(run_id)
        if state is None:
            return False
        state.cancel_requested = True
        return True

    def get_state(self, run_id: str) -> RunState | None:
        """Return the live :class:`RunState` for ``run_id``, or ``None``."""
        return self._live_states.get(run_id)

    def get_bus(self, run_id: str) -> RunEventBus | None:
        """Return the live :class:`RunEventBus` for ``run_id``, or ``None``.

        The WebSocket endpoint (Task 18.1) uses this to subscribe to a
        Run that is currently executing. Callers that arrive after a
        Run has finished should fall back to replaying persisted
        events from the store.
        """
        return self._live_buses.get(run_id)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Main loop: dequeue → execute → cleanup."""
        while True:
            try:
                pending = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._execute_one(pending)
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 — never let the worker die.
                log.exception(
                    "run supervisor: unexpected error in run %s", pending.run_id
                )
            finally:
                # Keep the live-state entries around briefly so any
                # in-flight WebSocket replay that arrives right after
                # the terminal event can still find the bus; callers
                # that care about memory can subclass and clear them
                # on a timer. For the v1 single-slot supervisor the
                # next ``submit`` starts a fresh state anyway.
                self._live_states.pop(pending.run_id, None)
                self._live_buses.pop(pending.run_id, None)

    async def _execute_one(self, pending: _PendingRun) -> None:
        """Run a single queued :class:`_PendingRun` to completion."""
        config_file = pending.config_file
        run_config = config_file.run

        # Load suites from disk and filter by run_config.suites.
        try:
            all_suites = discover_suites(self._suites_dir)
        except Exception:  # noqa: BLE001 — preflight will surface via event.
            log.exception("run supervisor: failed to discover suites")
            all_suites = []

        by_name = {s.name: s for s in all_suites}
        selected_suites: list[EvaluationSuite] = [
            by_name[n] for n in run_config.suites if n in by_name
        ]

        generation_defaults = (
            selected_suites[0].defaults if selected_suites else GenerationDefaults()
        )

        client = self._client_factory(
            config_file.ollama_base_url, run_config.ollama_timeout_s
        )
        try:
            scheduler = RunScheduler(
                run_state=pending.state,
                bus=pending.bus,
                ollama_client=client,
                run_config=run_config,
                suites=selected_suites,
                generation_defaults=generation_defaults,
                config_file=config_file,
                store=self._store,
            )
            await scheduler.execute()
        finally:
            if hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001 - best-effort cleanup.
                    log.exception("run supervisor: error closing ollama client")


def _default_client_factory(base_url: str, timeout_s: float) -> OllamaClient:
    """Factory used when the caller does not inject a test double."""
    return OllamaClient(base_url=base_url, timeout_s=timeout_s)


__all__ = [
    "RunSupervisor",
]
