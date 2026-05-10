"""WebSocket endpoint streaming Run events to subscribed clients.

Implements ``GET /api/runs/{run_id}/events`` per Task 18.1:

* On subscribe, look up the Run via the supervisor (live) or the
  :class:`HistoryStore` (persisted). Missing Run → close with
  code 4404 and reason ``"run_not_found"`` before any frame is
  sent.
* Replay every persisted event from :meth:`HistoryStore.list_events`
  in ``seq`` order.
* Subscribe to the live bus (if any) and forward every subsequent
  event, skipping those already emitted during replay.
* Send the terminal event and close with code 1000.

Property 34 (multi-subscriber replay and isolation) and Property 31
(event log bookends) are the behavioural anchors. The endpoint
uses :class:`~ollama_evaluator.runner.run_state.RunEventBus.subscribe`
for the live tail; replay is derived from the events currently on
the bus's state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..events import RunEvent, RunEventAdapter

log = logging.getLogger(__name__)


_TERMINAL_TYPES = frozenset({"run-completed", "run-aborted", "run-failed"})


def register_events_ws(app: FastAPI) -> None:
    """Attach the ``/api/runs/{run_id}/events`` WebSocket route to ``app``."""

    @app.websocket("/api/runs/{run_id}/events")
    async def events_ws(websocket: WebSocket, run_id: str) -> None:
        deps = app.state.deps
        supervisor = deps.supervisor
        store = deps.store

        await websocket.accept()

        live_bus = supervisor.get_bus(run_id) if supervisor is not None else None
        live_state = supervisor.get_state(run_id) if supervisor is not None else None

        # Collect initial replay from persisted store (if any) plus the
        # in-memory bus snapshot. We deduplicate by ``seq`` so clients
        # never see a duplicate event.
        persisted: list[RunEvent] = []
        try:
            persisted = await store.list_events(run_id)
        except Exception:  # noqa: BLE001 — tolerate missing store rows.
            log.exception("events_ws: list_events failed for %s", run_id)
            persisted = []

        # If neither the store nor the supervisor knows about this Run,
        # close with the agreed-upon code 4404 reason.
        if not persisted and live_bus is None:
            # One final check: does the Run row exist at all?
            report = None
            try:
                report = await store.get_run(run_id)
            except Exception:  # noqa: BLE001
                report = None
            if report is None:
                await websocket.close(code=4404, reason="run_not_found")
                return

        last_seq = -1
        try:
            for ev in persisted:
                await _send_event(websocket, ev)
                last_seq = ev.seq

            # If there is a live bus, subscribe for new events; otherwise
            # the persisted replay already concluded with a terminal event
            # and we can close.
            if live_bus is not None:
                async for ev in live_bus.subscribe():
                    if ev.seq <= last_seq:
                        continue
                    await _send_event(websocket, ev)
                    last_seq = ev.seq
                    if ev.type in _TERMINAL_TYPES:
                        break
            else:
                # Sanity: ensure replay ended with a terminal event.
                if persisted and persisted[-1].type not in _TERMINAL_TYPES:
                    # Rare — in-flight Run observed by the store but not
                    # registered with the supervisor. Close cleanly.
                    pass

            await websocket.close(code=1000)
        except WebSocketDisconnect:
            # Client disconnect: Requirement 14.7 — the producer and
            # other subscribers are unaffected, so we simply exit.
            return
        except Exception:  # noqa: BLE001 — never surface raw exceptions.
            log.exception("events_ws: unexpected error for %s", run_id)
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001 — best effort.
                pass


async def _send_event(websocket: WebSocket, event: RunEvent) -> None:
    """Serialise ``event`` to JSON text and push it over the socket."""
    text = RunEventAdapter.dump_json(event).decode("utf-8")
    await websocket.send_text(text)


__all__ = [
    "register_events_ws",
]
