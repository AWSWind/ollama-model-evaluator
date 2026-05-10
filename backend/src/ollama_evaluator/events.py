"""Pydantic models for the Run_Event stream.

A Run emits a discrete sequence of :class:`BaseRunEvent` values while it
is running; these events are persisted in the ``run_events`` SQLite table
(Requirement 14.5) and fan out to WebSocket subscribers of
``GET /api/runs/{run_id}/events`` (Requirement 14.1). This module
defines the closed set of event variants the Backend emits and the
discriminated union used to validate/serialise them uniformly.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Data Models / Run_Event. Tag values and field layouts here match the
design document exactly so the WebSocket wire format (text frames with a
JSON object per event, §API / WebSocket event stream) is driven by a
single Pydantic schema.

Key requirements wired through this module:

* 14.2 — ``run-started`` event carries the upfront execution plan.
* 14.3 — ``test-case-completed`` event carries the full
  :class:`~ollama_evaluator.models.TestCaseResult` for the execution.
* 14.4 — ``run-progress`` event cadence (≤ 2s) is enforced by the
  producer; the event itself carries ``completed``/``in_progress``/
  ``pending`` counters exposed in the UI progress panel.
* 14.5 — Exactly one terminal event per Run: ``run-completed``,
  ``run-aborted``, or ``run-failed``.

All models set ``ConfigDict(extra="forbid")`` so that mis-spelt producer
code fails fast rather than silently dropping fields on the wire. The
discriminated union below lets consumers validate any incoming event
dictionary against the closed tag set without writing dispatcher code,
and :data:`RunEventAdapter` wraps the union in a ``TypeAdapter`` so
``validate_python``/``validate_json``/``dump_json`` work uniformly on
any variant.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .models import RunSummary, TestCaseResult


class BaseRunEvent(BaseModel):
    """Fields common to every Run_Event variant.

    ``seq`` is a monotonically increasing sequence number assigned by the
    producer (the per-Run :class:`~ollama_evaluator.runner.run_state.RunEventBus`
    in later tasks). It starts at ``0`` for the ``run-started`` event of
    each Run and increases by 1 for every subsequent event on that Run.
    Consumers use ``seq`` to detect gaps on reconnect (Property 34,
    Requirement 14.6) and to order replay against newly-appended events.

    ``ts`` is the UTC wall-clock timestamp at which the producer appended
    the event. It is not used for ordering — that is ``seq``'s job — but
    it is persisted so the UI can render absolute times in the run-detail
    view (Requirement 16.3).

    ``run_id`` is a foreign key to the enclosing Run. Non-empty strings
    are enforced here so a malformed producer cannot smuggle a blank
    ``run_id`` into the event log where it would alias every other Run.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        ...,
        description=(
            "Identifier of the Run this event belongs to. Must be a "
            "non-empty string; matches "
            ":attr:`~ollama_evaluator.models.RunReport.run_id`."
        ),
    )
    seq: int = Field(
        ...,
        ge=0,
        description=(
            "Monotonically increasing per-Run sequence number assigned "
            "by the producer. ``0`` is the first event (``run-started``). "
            "Must be >= 0 (Requirement 14.6, Property 34)."
        ),
    )
    ts: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the producer appended this event. "
            "Used for display, not for ordering (see ``seq``)."
        ),
    )

    @field_validator("run_id")
    @classmethod
    def _run_id_non_empty(cls, value: str) -> str:
        """Reject blank ``run_id`` values; they collide with every Run."""
        if not value or not value.strip():
            raise ValueError("BaseRunEvent.run_id must be a non-empty string")
        return value


class RunStartedEvent(BaseRunEvent):
    """First event of every Run (Requirement 14.2).

    Emitted once the Run transitions from ``pending`` to ``running``,
    after preflight has succeeded (Ollama reachable, requested models
    present, remote-mode suites materialised). Carries the upfront
    execution plan so subscribers can render a progress bar immediately
    rather than waiting for the first ``run-progress`` tick.
    """

    type: Literal["run-started"] = "run-started"
    planned_executions: int = Field(
        ...,
        ge=0,
        description=(
            "Total number of executions planned for this Run, equal to "
            "``|models| * |test cases| * repetitions`` after tag and "
            "name filtering. Must be >= 0."
        ),
    )


class RunProgressEvent(BaseRunEvent):
    """Periodic progress tick emitted at most every 2 seconds (Requirement 14.4).

    The producer emits one of these roughly every 2 seconds while the
    Run is ``running`` (Property 33, cadence ≤ 2s + ε). The counters
    partition the plan into three disjoint classes:

    * ``completed`` — executions that have produced a
      :class:`~ollama_evaluator.models.TestCaseResult`, regardless of
      ``status``.
    * ``in_progress`` — executions currently dispatched to the
      Ollama_Server (at most ``concurrency``).
    * ``pending`` — executions still in the dispatch queue.

    Together they sum to ``planned_executions`` from the ``run-started``
    event; the invariant is enforced by the scheduler but not here, so
    this event can be constructed from partial state in tests.
    """

    type: Literal["run-progress"] = "run-progress"
    completed: int = Field(
        ...,
        ge=0,
        description="Executions that produced a ``TestCaseResult``.",
    )
    in_progress: int = Field(
        ...,
        ge=0,
        description="Executions currently dispatched to the Ollama_Server.",
    )
    pending: int = Field(
        ...,
        ge=0,
        description="Executions still in the dispatch queue.",
    )


class TestCaseCompletedEvent(BaseRunEvent):
    """One per executed ``(model, test_case_id, repetition)`` tuple (Requirement 14.3).

    Carries the full :class:`~ollama_evaluator.models.TestCaseResult` so
    the UI can render the execution immediately on arrival without a
    follow-up fetch (Property 32). Emitted exactly once per planned
    execution that actually ran; executions skipped by cancellation are
    not reported here (they appear only in the aborted Run_Report).
    """

    __test__ = False  # pytest: do not treat this class as a test container.

    type: Literal["test-case-completed"] = "test-case-completed"
    result: TestCaseResult = Field(
        ...,
        description=(
            "Full per-execution result, including ``status``, "
            "``response``, ``performance``, and every configured "
            "``MetricResult``."
        ),
    )


class RunCompletedEvent(BaseRunEvent):
    """Terminal event for a Run that finished every planned execution.

    Emitted exactly once per Run when the scheduler observes that every
    planned execution has produced a :class:`TestCaseResult`
    (Requirement 14.5, Property 31). The :class:`RunSummary` mirrors the
    counters exposed in the UI progress panel (Requirement 15.5); the
    full Run_Report is fetched separately from the REST API.
    """

    type: Literal["run-completed"] = "run-completed"
    summary: RunSummary = Field(
        ...,
        description=(
            "Compact pass/fail/error/timeout counters for the Run. "
            "Matches :class:`~ollama_evaluator.models.RunSummary`."
        ),
    )


class RunAbortedEvent(BaseRunEvent):
    """Terminal event for a Run cancelled mid-flight (Requirement 11.4).

    Emitted exactly once when the Run stops because of a
    ``POST /api/runs/{id}/cancel`` request or a SIGINT/SIGTERM signal.
    ``reason`` is a short, human-readable explanation suitable for the
    UI terminal-error banner (Requirement 16.6); it is free-form because
    the underlying cause can be either a user action ("cancelled") or a
    signal name ("SIGTERM").
    """

    type: Literal["run-aborted"] = "run-aborted"
    reason: str = Field(
        ...,
        description=(
            "Short, human-readable cause of the abort (e.g. "
            "``'cancelled'``, ``'SIGTERM'``). Rendered in the UI "
            "terminal-error banner."
        ),
    )


class RunFailedEvent(BaseRunEvent):
    """Terminal event for a Run that failed before or during execution.

    Emitted exactly once when the Run cannot proceed. Common causes
    (see ``design.md`` §Error-taxonomy): ``ollama_unreachable`` (Req
    1.3), ``model_not_found`` (Req 2.3), ``dataset_fetch_failed``
    (Req 17.6), ``field_map_invalid`` (Req 17.7). ``error_code`` is one
    of those tokens; ``message`` carries the specific cause for the UI
    terminal-error banner (Requirement 16.6).
    """

    type: Literal["run-failed"] = "run-failed"
    error_code: str = Field(
        ...,
        description=(
            "Stable error tag from ``design.md`` §Error-taxonomy, e.g. "
            "``'ollama_unreachable'`` or ``'dataset_fetch_failed'``."
        ),
    )
    message: str = Field(
        ...,
        description=(
            "Human-readable failure message rendered in the UI "
            "terminal-error banner."
        ),
    )


# Discriminated union: Pydantic picks the concrete subclass at validation
# time based on the ``type`` field. Using ``Annotated[Union[...], Field(
# discriminator="type")]`` (rather than a bare ``Union``) lets Pydantic
# short-circuit validation: it reads ``type``, dispatches to the matching
# model, and reports a single "no variant matched" error instead of one
# per variant on unknown tags.
RunEvent = Annotated[
    Union[
        RunStartedEvent,
        RunProgressEvent,
        TestCaseCompletedEvent,
        RunCompletedEvent,
        RunAbortedEvent,
        RunFailedEvent,
    ],
    Field(discriminator="type"),
]

# ``TypeAdapter`` wraps the union so callers can ``validate_python``,
# ``validate_json``, and ``dump_json`` any event variant through a
# single entry point. This is what the History_Store (Task 13) and the
# WebSocket endpoint (Task 18) use to serialise/deserialise events
# without special-casing each variant.
RunEventAdapter: TypeAdapter[RunEvent] = TypeAdapter(RunEvent)


__all__ = [
    "BaseRunEvent",
    "RunAbortedEvent",
    "RunCompletedEvent",
    "RunEvent",
    "RunEventAdapter",
    "RunFailedEvent",
    "RunProgressEvent",
    "RunStartedEvent",
    "TestCaseCompletedEvent",
]
