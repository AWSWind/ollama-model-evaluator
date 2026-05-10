"""Unit tests for the Run_Report Pydantic models (Task 2.3).

These tests verify the invariants specified in
``.kiro/specs/ollama-model-evaluator/design.md`` §Data Models for the
Run_Report and its child models:

* Every model accepts a valid minimal instance with the fields from
  design.md populated.
* Every model rejects unknown fields (``extra="forbid"``) so that hand
  edited reports surface typos immediately — load-bearing for the
  Property 18 round-trip invariant.
* ``TestCaseResult.status`` is a four-valued literal; only the four
  listed strings are accepted.
* ``MetricResult.error`` defaults to ``None`` and can be populated when
  a metric raises during scoring (Requirement 7.5).
* ``RunReport`` rejects ``ended_at < started_at`` (Property 16).
* ``repetition`` is 1-indexed on both ``TestCaseResult`` and
  ``ErrorSummaryEntry``; zero and negative values are rejected.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ollama_evaluator.config import ConfigFile, RunConfig
from ollama_evaluator.models import (
    ErrorSummaryEntry,
    MetricAggregate,
    MetricResult,
    ModelAggregate,
    ModelInfo,
    PerformanceMetrics,
    RunReport,
    RunSummary,
    TestCaseResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _performance(**overrides: Any) -> PerformanceMetrics:
    defaults: dict[str, Any] = {
        "ttft_ms": 100.0,
        "total_ms": 500.0,
        "prompt_tokens": 10,
        "response_tokens": 20,
        "tokens_per_second": 40.0,
    }
    defaults.update(overrides)
    return PerformanceMetrics(**defaults)


def _metric_result(**overrides: Any) -> MetricResult:
    defaults: dict[str, Any] = {
        "name": "exact-match",
        "score": 1.0,
        "passed": True,
    }
    defaults.update(overrides)
    return MetricResult(**defaults)


def _test_case_result(**overrides: Any) -> TestCaseResult:
    defaults: dict[str, Any] = {
        "model": "llama3:8b",
        "suite": "reasoning-basics",
        "test_case_id": "case-1",
        "repetition": 1,
        "status": "pass",
        "response": "42",
        "error_message": None,
        "performance": _performance(),
        "metrics": [_metric_result()],
    }
    defaults.update(overrides)
    return TestCaseResult(**defaults)


def _metric_aggregate(**overrides: Any) -> MetricAggregate:
    defaults: dict[str, Any] = {
        "metric": "exact-match",
        "mean": 1.0,
        "stddev": 0.0,
        "count": 1,
    }
    defaults.update(overrides)
    return MetricAggregate(**defaults)


def _model_aggregate(**overrides: Any) -> ModelAggregate:
    defaults: dict[str, Any] = {
        "model": "llama3:8b",
        "passed": 1,
        "failed": 0,
        "errored": 0,
        "timed_out": 0,
        "mean_ttft_ms": 100.0,
        "mean_total_ms": 500.0,
        "mean_tokens_per_second": 40.0,
        "metric_aggregates": {"exact-match": _metric_aggregate()},
    }
    defaults.update(overrides)
    return ModelAggregate(**defaults)


def _run_config() -> RunConfig:
    return RunConfig(models=["llama3:8b"], suites=["reasoning-basics"])


def _config_file() -> ConfigFile:
    return ConfigFile(suites_dir=Path("./suites"), run=_run_config())


def _run_report(**overrides: Any) -> RunReport:
    started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    defaults: dict[str, Any] = {
        "run_id": "run-abc",
        "backend_version": "0.1.0",
        "ollama_version": "0.1.32",
        "started_at": started,
        "ended_at": started + timedelta(seconds=30),
        "status": "completed",
        "config": _config_file(),
        "models": [ModelInfo(name="llama3:8b")],
        "results": [_test_case_result()],
        "aggregates": [_model_aggregate()],
        "error_summary": [],
    }
    defaults.update(overrides)
    return RunReport(**defaults)


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_minimal_valid_instance(self) -> None:
        perf = PerformanceMetrics(total_ms=500.0)
        assert perf.total_ms == 500.0
        assert perf.ttft_ms is None
        assert perf.prompt_tokens is None
        assert perf.response_tokens is None
        assert perf.tokens_per_second is None

    def test_populated_instance(self) -> None:
        perf = _performance()
        assert perf.ttft_ms == 100.0
        assert perf.prompt_tokens == 10
        assert perf.response_tokens == 20
        assert perf.tokens_per_second == 40.0

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            PerformanceMetrics.model_validate(
                {"total_ms": 500.0, "mystery_field": 1.0}
            )
        assert "mystery_field" in str(excinfo.value)

    def test_total_ms_required(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            PerformanceMetrics.model_validate({})
        assert "total_ms" in str(excinfo.value)


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_minimal_valid_instance(self) -> None:
        result = MetricResult(name="exact-match", score=0.0, passed=False)
        assert result.name == "exact-match"
        assert result.score == 0.0
        assert result.passed is False
        # Requirement 7.5: default shape for a successful scoring call.
        assert result.threshold is None
        assert result.details == {}
        assert result.error is None

    def test_error_can_be_populated(self) -> None:
        """Requirement 7.5: ``error`` is populated when a metric raises."""
        result = MetricResult(
            name="regex-match",
            score=0.0,
            passed=False,
            error="regex compile failed: bad pattern",
        )
        assert result.error == "regex compile failed: bad pattern"
        assert result.passed is False

    def test_details_can_carry_arbitrary_structured_data(self) -> None:
        result = MetricResult(
            name="response-capture",
            score=0.0,
            passed=True,
            details={"response": "hello", "length": 5},
        )
        assert result.details == {"response": "hello", "length": 5}

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricResult.model_validate(
                {
                    "name": "exact-match",
                    "score": 1.0,
                    "passed": True,
                    "typo_field": "boom",
                }
            )
        assert "typo_field" in str(excinfo.value)


# ---------------------------------------------------------------------------
# TestCaseResult
# ---------------------------------------------------------------------------


class TestTestCaseResultModel:
    def test_minimal_valid_instance(self) -> None:
        result = _test_case_result()
        assert result.model == "llama3:8b"
        assert result.status == "pass"
        assert result.repetition == 1
        assert len(result.metrics) == 1

    @pytest.mark.parametrize("status", ["pass", "fail", "error", "timeout"])
    def test_status_accepts_each_literal_value(self, status: str) -> None:
        result = _test_case_result(status=status)
        assert result.status == status

    @pytest.mark.parametrize(
        "bad_status",
        ["passed", "FAIL", "", "success", "ok", "errored"],
    )
    def test_status_rejects_other_values(self, bad_status: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _test_case_result(status=bad_status)
        assert "status" in str(excinfo.value)

    @pytest.mark.parametrize("bad_rep", [0, -1, -10])
    def test_repetition_must_be_positive(self, bad_rep: int) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _test_case_result(repetition=bad_rep)
        assert "repetition" in str(excinfo.value)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TestCaseResult.model_validate(
                {
                    "model": "llama3:8b",
                    "suite": "reasoning-basics",
                    "test_case_id": "case-1",
                    "repetition": 1,
                    "status": "pass",
                    "response": "42",
                    "error_message": None,
                    "performance": {"total_ms": 500.0},
                    "metrics": [],
                    "surprise": "nope",
                }
            )
        assert "surprise" in str(excinfo.value)

    def test_empty_metrics_list_allowed_on_result(self) -> None:
        """A ``TestCaseResult`` may have zero metrics on ``error``/``timeout``."""
        result = _test_case_result(
            status="error",
            response=None,
            error_message="boom",
            metrics=[],
        )
        assert result.metrics == []


# ---------------------------------------------------------------------------
# MetricAggregate
# ---------------------------------------------------------------------------


class TestMetricAggregate:
    def test_minimal_valid_instance(self) -> None:
        agg = _metric_aggregate()
        assert agg.metric == "exact-match"
        assert agg.count == 1

    def test_stddev_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _metric_aggregate(stddev=-0.1)
        assert "stddev" in str(excinfo.value)

    def test_count_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _metric_aggregate(count=-1)
        assert "count" in str(excinfo.value)

    def test_count_zero_allowed(self) -> None:
        """``count == 0`` is a legal empty aggregate."""
        agg = _metric_aggregate(count=0, mean=0.0, stddev=0.0)
        assert agg.count == 0

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricAggregate.model_validate(
                {
                    "metric": "exact-match",
                    "mean": 0.5,
                    "stddev": 0.1,
                    "count": 2,
                    "mystery": 1,
                }
            )
        assert "mystery" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ModelAggregate
# ---------------------------------------------------------------------------


class TestModelAggregate:
    def test_minimal_valid_instance(self) -> None:
        agg = _model_aggregate()
        assert agg.model == "llama3:8b"
        assert agg.passed == 1
        assert agg.mean_total_ms == 500.0
        assert "exact-match" in agg.metric_aggregates

    def test_nullable_performance_means(self) -> None:
        """Req 6.5: mean performance fields may be ``None``."""
        agg = _model_aggregate(
            mean_ttft_ms=None,
            mean_tokens_per_second=None,
        )
        assert agg.mean_ttft_ms is None
        assert agg.mean_tokens_per_second is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ModelAggregate.model_validate(
                {
                    "model": "m",
                    "passed": 0,
                    "failed": 0,
                    "errored": 0,
                    "timed_out": 0,
                    "mean_ttft_ms": None,
                    "mean_total_ms": 0.0,
                    "mean_tokens_per_second": None,
                    "metric_aggregates": {},
                    "typo": 1,
                }
            )
        assert "typo" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_minimal_valid_instance(self) -> None:
        info = ModelInfo(name="llama3:8b")
        assert info.name == "llama3:8b"
        assert info.digest is None
        assert info.parameter_size is None

    def test_populated_instance(self) -> None:
        info = ModelInfo(
            name="llama3:8b",
            digest="sha256:abc123",
            parameter_size="8B",
        )
        assert info.digest == "sha256:abc123"
        assert info.parameter_size == "8B"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ModelInfo.model_validate({"name": "llama3:8b", "mystery": 1})
        assert "mystery" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ErrorSummaryEntry
# ---------------------------------------------------------------------------


class TestErrorSummaryEntry:
    def test_minimal_valid_instance(self) -> None:
        entry = ErrorSummaryEntry(
            model="llama3:8b",
            suite="reasoning-basics",
            test_case_id="case-1",
            repetition=1,
            error_message="boom",
        )
        assert entry.error_message == "boom"

    @pytest.mark.parametrize("bad_rep", [0, -1, -5])
    def test_repetition_must_be_positive(self, bad_rep: int) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ErrorSummaryEntry(
                model="m",
                suite="s",
                test_case_id="c1",
                repetition=bad_rep,
                error_message="boom",
            )
        assert "repetition" in str(excinfo.value)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ErrorSummaryEntry.model_validate(
                {
                    "model": "m",
                    "suite": "s",
                    "test_case_id": "c",
                    "repetition": 1,
                    "error_message": "boom",
                    "typo": 1,
                }
            )
        assert "typo" in str(excinfo.value)


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_minimal_valid_instance(self) -> None:
        summary = RunSummary(
            planned_executions=4,
            completed_executions=4,
            passed=3,
            failed=1,
            errored=0,
            timed_out=0,
        )
        assert summary.passed == 3

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunSummary.model_validate(
                {
                    "planned_executions": 1,
                    "completed_executions": 1,
                    "passed": 1,
                    "failed": 0,
                    "errored": 0,
                    "timed_out": 0,
                    "extra": 1,
                }
            )
        assert "extra" in str(excinfo.value)


# ---------------------------------------------------------------------------
# RunReport
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_minimal_valid_instance(self) -> None:
        report = _run_report()
        assert report.run_id == "run-abc"
        assert report.status == "completed"
        assert report.ended_at is not None
        assert report.ended_at >= report.started_at
        assert len(report.results) == 1
        assert len(report.aggregates) == 1

    def test_run_id_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_report(run_id="")
        assert "run_id" in str(excinfo.value)

    def test_run_id_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _run_report(run_id="   ")

    def test_ended_at_may_be_none(self) -> None:
        """A still-running Run has no ``ended_at``."""
        report = _run_report(status="running", ended_at=None)
        assert report.ended_at is None

    def test_ended_at_before_started_at_rejected(self) -> None:
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        earlier = started - timedelta(seconds=5)
        with pytest.raises(ValidationError) as excinfo:
            _run_report(started_at=started, ended_at=earlier)
        message = str(excinfo.value)
        assert "ended_at" in message

    def test_ended_at_equal_to_started_at_accepted(self) -> None:
        """Equal timestamps are allowed — a zero-length Run is still valid."""
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        report = _run_report(started_at=started, ended_at=started)
        assert report.started_at == report.ended_at

    @pytest.mark.parametrize(
        "status",
        ["pending", "running", "completed", "aborted", "failed"],
    )
    def test_status_accepts_each_literal_value(self, status: str) -> None:
        # ``ended_at`` is permitted to be None for in-flight states.
        report = _run_report(status=status, ended_at=None)
        assert report.status == status

    @pytest.mark.parametrize("bad_status", ["done", "ok", "", "ABORTED"])
    def test_status_rejects_other_values(self, bad_status: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_report(status=bad_status)
        assert "status" in str(excinfo.value)

    def test_extra_fields_forbidden(self) -> None:
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValidationError) as excinfo:
            RunReport.model_validate(
                {
                    "run_id": "r",
                    "backend_version": "0.1.0",
                    "ollama_version": None,
                    "started_at": started,
                    "ended_at": None,
                    "status": "pending",
                    "config": _config_file().model_dump(mode="json"),
                    "models": [],
                    "results": [],
                    "aggregates": [],
                    "error_summary": [],
                    "typo_field": 1,
                }
            )
        assert "typo_field" in str(excinfo.value)

    def test_round_trip_via_json(self) -> None:
        """Smoke check for Property 18; full property test is Task 2.5."""
        report = _run_report()
        rebuilt = RunReport.model_validate_json(report.model_dump_json())
        assert rebuilt == report
