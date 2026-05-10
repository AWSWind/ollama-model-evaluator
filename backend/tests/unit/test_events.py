"""Unit tests for the Run_Event discriminated union (Task 2.4).

These tests verify the invariants from
``.kiro/specs/ollama-model-evaluator/design.md`` §Data Models /
Run_Event and its companion Requirements 14.2–14.5:

* Every concrete event variant can be constructed from a plain ``dict``
  via :data:`RunEventAdapter` and the ``type`` tag selects the right
  subclass.
* Every event round-trips through JSON without loss
  (``validate_json(dump_json(ev)) == ev``) so the WebSocket wire format
  and the SQLite ``run_events`` persistence both share a single schema.
* An unknown ``type`` tag produces a ``ValidationError`` rather than a
  silent pass: the discriminated union is closed.
* Every model rejects unknown fields (``extra="forbid"``) so typos in
  producer code fail fast.
* ``seq`` is bounded below by ``0`` and ``run_id`` rejects blank strings
  — both are load-bearing for the event log ordering (Requirement 14.6)
  and for the foreign-key relationship to
  :class:`~ollama_evaluator.models.RunReport.run_id`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ollama_evaluator.config import ConfigFile, RunConfig  # noqa: F401 - used in helpers
from ollama_evaluator.events import (
    BaseRunEvent,
    RunAbortedEvent,
    RunCompletedEvent,
    RunEventAdapter,
    RunFailedEvent,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# A fixed timestamp keeps ``seq``-based ordering and JSON output stable
# across runs so tests compare by value rather than by "close enough".
_TS = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _base_fields(**overrides: Any) -> dict[str, Any]:
    """Return the shared ``BaseRunEvent`` fields with optional overrides."""
    defaults: dict[str, Any] = {"run_id": "run-abc", "seq": 0, "ts": _TS}
    defaults.update(overrides)
    return defaults


def _test_case_result() -> TestCaseResult:
    """Build a minimal, fully valid :class:`TestCaseResult` for nesting."""
    return TestCaseResult(
        model="llama3:8b",
        suite="reasoning-basics",
        test_case_id="case-1",
        repetition=1,
        status="pass",
        response="42",
        error_message=None,
        performance=PerformanceMetrics(total_ms=500.0),
        metrics=[MetricResult(name="exact-match", score=1.0, passed=True)],
    )


def _run_summary() -> RunSummary:
    return RunSummary(
        planned_executions=4,
        completed_executions=4,
        passed=3,
        failed=1,
        errored=0,
        timed_out=0,
    )


# ---------------------------------------------------------------------------
# BaseRunEvent invariants — surfaced through every concrete variant.
# ---------------------------------------------------------------------------


class TestBaseRunEventInvariants:
    """Constraints from :class:`BaseRunEvent` apply to every subclass."""

    def test_seq_must_be_non_negative(self) -> None:
        """Requirement 14.6: producer-assigned ``seq`` is >= 0."""
        with pytest.raises(ValidationError) as excinfo:
            RunStartedEvent(
                run_id="run-abc",
                seq=-1,
                ts=_TS,
                planned_executions=0,
            )
        assert "seq" in str(excinfo.value)

    def test_seq_zero_is_accepted(self) -> None:
        """``seq=0`` is the ``run-started`` slot and must be valid."""
        event = RunStartedEvent(
            run_id="run-abc",
            seq=0,
            ts=_TS,
            planned_executions=0,
        )
        assert event.seq == 0

    @pytest.mark.parametrize("bad_run_id", ["", "   ", "\t\n"])
    def test_run_id_must_be_non_empty(self, bad_run_id: str) -> None:
        """Blank ``run_id`` is rejected so every event keys a real Run."""
        with pytest.raises(ValidationError) as excinfo:
            RunStartedEvent(
                run_id=bad_run_id,
                seq=0,
                ts=_TS,
                planned_executions=0,
            )
        assert "run_id" in str(excinfo.value)

    def test_base_event_is_abstract_enough_to_forbid_extras(self) -> None:
        """``BaseRunEvent`` itself forbids unknown fields."""
        with pytest.raises(ValidationError) as excinfo:
            BaseRunEvent.model_validate(
                {"run_id": "run-abc", "seq": 0, "ts": _TS, "surprise": 1}
            )
        assert "surprise" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Concrete variant construction + tag defaults.
# ---------------------------------------------------------------------------


class TestConcreteEventVariants:
    """Each variant has the correct literal ``type`` default and extras=forbid."""

    def test_run_started_defaults_tag(self) -> None:
        event = RunStartedEvent(**_base_fields(), planned_executions=12)
        assert event.type == "run-started"
        assert event.planned_executions == 12

    def test_run_started_rejects_negative_planned(self) -> None:
        with pytest.raises(ValidationError):
            RunStartedEvent(**_base_fields(), planned_executions=-1)

    def test_run_progress_defaults_tag(self) -> None:
        event = RunProgressEvent(
            **_base_fields(seq=3),
            completed=2,
            in_progress=1,
            pending=7,
        )
        assert event.type == "run-progress"
        assert (event.completed, event.in_progress, event.pending) == (2, 1, 7)

    @pytest.mark.parametrize("field", ["completed", "in_progress", "pending"])
    def test_run_progress_counters_non_negative(self, field: str) -> None:
        payload: dict[str, Any] = {
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            field: -1,
        }
        with pytest.raises(ValidationError) as excinfo:
            RunProgressEvent(**_base_fields(), **payload)
        assert field in str(excinfo.value)

    def test_test_case_completed_carries_full_result(self) -> None:
        """Requirement 14.3: the event includes the full TestCaseResult."""
        result = _test_case_result()
        event = TestCaseCompletedEvent(
            **_base_fields(seq=5), result=result
        )
        assert event.type == "test-case-completed"
        assert event.result == result

    def test_run_completed_carries_summary(self) -> None:
        summary = _run_summary()
        event = RunCompletedEvent(**_base_fields(seq=99), summary=summary)
        assert event.type == "run-completed"
        assert event.summary == summary

    def test_run_aborted_carries_reason(self) -> None:
        event = RunAbortedEvent(**_base_fields(seq=99), reason="cancelled")
        assert event.type == "run-aborted"
        assert event.reason == "cancelled"

    def test_run_failed_carries_error_code_and_message(self) -> None:
        event = RunFailedEvent(
            **_base_fields(seq=99),
            error_code="ollama_unreachable",
            message="could not connect to http://localhost:11434",
        )
        assert event.type == "run-failed"
        assert event.error_code == "ollama_unreachable"
        assert "localhost" in event.message


# ---------------------------------------------------------------------------
# extra="forbid" on every concrete event variant.
# ---------------------------------------------------------------------------


class TestExtraFieldsForbidden:
    """``ConfigDict(extra='forbid')`` is load-bearing across the union."""

    def test_run_started_forbids_extras(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunStartedEvent.model_validate(
                {
                    **_base_fields(),
                    "type": "run-started",
                    "planned_executions": 1,
                    "surprise": 1,
                }
            )
        assert "surprise" in str(excinfo.value)

    def test_run_progress_forbids_extras(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunProgressEvent.model_validate(
                {
                    **_base_fields(),
                    "type": "run-progress",
                    "completed": 0,
                    "in_progress": 0,
                    "pending": 0,
                    "typo": "boom",
                }
            )
        assert "typo" in str(excinfo.value)

    def test_test_case_completed_forbids_extras(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TestCaseCompletedEvent.model_validate(
                {
                    **_base_fields(),
                    "type": "test-case-completed",
                    "result": _test_case_result().model_dump(mode="python"),
                    "extra": 1,
                }
            )
        assert "extra" in str(excinfo.value)

    def test_run_completed_forbids_extras(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunCompletedEvent.model_validate(
                {
                    **_base_fields(),
                    "type": "run-completed",
                    "summary": _run_summary().model_dump(mode="python"),
                    "extra": 1,
                }
            )
        assert "extra" in str(excinfo.value)

    def test_run_aborted_forbids_extras(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunAbortedEvent.model_validate(
                {
                    **_base_fields(),
                    "type": "run-aborted",
                    "reason": "cancelled",
                    "extra": 1,
                }
            )
        assert "extra" in str(excinfo.value)

    def test_run_failed_forbids_extras(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunFailedEvent.model_validate(
                {
                    **_base_fields(),
                    "type": "run-failed",
                    "error_code": "ollama_unreachable",
                    "message": "nope",
                    "extra": 1,
                }
            )
        assert "extra" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Discriminated-union dispatch via RunEventAdapter.validate_python.
# ---------------------------------------------------------------------------


class TestRunEventAdapterDispatch:
    """``RunEventAdapter.validate_python`` picks the variant from ``type``."""

    def test_run_started_variant(self) -> None:
        payload: dict[str, Any] = {
            **_base_fields(),
            "type": "run-started",
            "planned_executions": 3,
        }
        event = RunEventAdapter.validate_python(payload)
        assert isinstance(event, RunStartedEvent)
        assert event.planned_executions == 3

    def test_run_progress_variant(self) -> None:
        payload: dict[str, Any] = {
            **_base_fields(seq=1),
            "type": "run-progress",
            "completed": 1,
            "in_progress": 2,
            "pending": 3,
        }
        event = RunEventAdapter.validate_python(payload)
        assert isinstance(event, RunProgressEvent)

    def test_test_case_completed_variant(self) -> None:
        payload: dict[str, Any] = {
            **_base_fields(seq=2),
            "type": "test-case-completed",
            "result": _test_case_result().model_dump(mode="python"),
        }
        event = RunEventAdapter.validate_python(payload)
        assert isinstance(event, TestCaseCompletedEvent)
        assert event.result.test_case_id == "case-1"

    def test_run_completed_variant(self) -> None:
        payload: dict[str, Any] = {
            **_base_fields(seq=9),
            "type": "run-completed",
            "summary": _run_summary().model_dump(mode="python"),
        }
        event = RunEventAdapter.validate_python(payload)
        assert isinstance(event, RunCompletedEvent)

    def test_run_aborted_variant(self) -> None:
        payload: dict[str, Any] = {
            **_base_fields(seq=9),
            "type": "run-aborted",
            "reason": "SIGTERM",
        }
        event = RunEventAdapter.validate_python(payload)
        assert isinstance(event, RunAbortedEvent)

    def test_run_failed_variant(self) -> None:
        payload: dict[str, Any] = {
            **_base_fields(seq=9),
            "type": "run-failed",
            "error_code": "model_not_found",
            "message": "missing: llama3:8b",
        }
        event = RunEventAdapter.validate_python(payload)
        assert isinstance(event, RunFailedEvent)

    def test_unknown_type_tag_is_rejected(self) -> None:
        """The union is closed; unknown tags are errors, not fallbacks."""
        with pytest.raises(ValidationError) as excinfo:
            RunEventAdapter.validate_python(
                {
                    **_base_fields(),
                    "type": "run-resurrected",
                    "planned_executions": 0,
                }
            )
        # Pydantic's discriminator error references the offending field.
        assert "type" in str(excinfo.value)

    def test_missing_type_tag_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunEventAdapter.validate_python(
                {**_base_fields(), "planned_executions": 0}
            )
        assert "type" in str(excinfo.value)


# ---------------------------------------------------------------------------
# JSON round-trip across every variant.
# ---------------------------------------------------------------------------


def _each_variant() -> list[tuple[str, Any]]:
    """Return ``(label, event)`` pairs covering every concrete variant."""
    return [
        (
            "run-started",
            RunStartedEvent(**_base_fields(seq=0), planned_executions=4),
        ),
        (
            "run-progress",
            RunProgressEvent(
                **_base_fields(seq=1),
                completed=1,
                in_progress=2,
                pending=3,
            ),
        ),
        (
            "test-case-completed",
            TestCaseCompletedEvent(
                **_base_fields(seq=2), result=_test_case_result()
            ),
        ),
        (
            "run-completed",
            RunCompletedEvent(**_base_fields(seq=99), summary=_run_summary()),
        ),
        (
            "run-aborted",
            RunAbortedEvent(**_base_fields(seq=99), reason="cancelled"),
        ),
        (
            "run-failed",
            RunFailedEvent(
                **_base_fields(seq=99),
                error_code="dataset_fetch_failed",
                message="network error",
            ),
        ),
    ]


class TestRunEventJsonRoundTrip:
    """``validate_json(dump_json(ev)) == ev`` for every variant."""

    @pytest.mark.parametrize(
        "label,event",
        _each_variant(),
        ids=[label for label, _ in _each_variant()],
    )
    def test_round_trip_preserves_equality(
        self, label: str, event: Any
    ) -> None:
        encoded = RunEventAdapter.dump_json(event)
        rebuilt = RunEventAdapter.validate_json(encoded)
        assert rebuilt == event
        # And the variant class is preserved through the tag.
        assert type(rebuilt) is type(event)

    def test_dump_json_is_valid_utf8_bytes(self) -> None:
        """``TypeAdapter.dump_json`` returns ``bytes`` suitable for storage."""
        event = RunStartedEvent(**_base_fields(), planned_executions=0)
        encoded = RunEventAdapter.dump_json(event)
        assert isinstance(encoded, (bytes, bytearray))
        # Round-trip through ``json.loads`` to confirm the bytes are UTF-8
        # JSON (we do not care about the exact field order).
        import json

        payload = json.loads(encoded)
        assert payload["type"] == "run-started"
        assert payload["run_id"] == "run-abc"
