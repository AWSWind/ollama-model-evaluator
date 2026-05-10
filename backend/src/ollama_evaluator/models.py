"""Pydantic data models for Run_Reports and their aggregate statistics.

This module defines the models that describe the *output* of a Run: the
per-execution results, the per-metric/per-model aggregates, the run-level
summary, and the top-level :class:`RunReport` that is serialised to JSON
on disk and over the REST API.

These models complement the user-facing *input* models defined in
:mod:`ollama_evaluator.suites.models` (Evaluation_Suite) and
:mod:`ollama_evaluator.config` (Config_File / RunConfig): ``RunReport``
embeds a :class:`~ollama_evaluator.config.ConfigFile` verbatim so that a
completed Run is reproducible from its report alone (Requirement 8.4).

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Data Models / Run_Report. Key requirements driving the shapes and
validators in this module:

* 2.5, 8.4 — ``RunReport`` carries per-model ``ModelInfo`` (tag, digest,
  parameter size) and the full submitted ``ConfigFile``.
* 6.1–6.5 — ``PerformanceMetrics`` captures TTFT, total latency, prompt
  and response token counts (nullable when the Ollama_Server omits
  them), and computed tokens-per-second.
* 7.3–7.5 — ``MetricResult`` carries a numeric ``score``, a ``passed``
  classification, an optional ``threshold``, free-form metric-specific
  ``details``, and an optional ``error`` populated when the metric
  implementation raises while scoring.
* 7.6 — ``ModelAggregate`` carries per-metric ``MetricAggregate`` with
  mean and population standard deviation across repetitions.
* 8.2 — ``RunReport.aggregates`` carries one ``ModelAggregate`` per
  evaluated model.
* 11.3 — ``RunReport.error_summary`` carries one entry per result with
  ``status in {error, timeout}``.

All models forbid unknown fields (``ConfigDict(extra="forbid")``) so that
typos in hand-edited reports are surfaced immediately. Keeping
``extra="forbid"`` on the output side is load-bearing for Property 18
(``RunReport`` round-trip): after ``model_dump_json`` / ``model_validate_json``
the two instances must compare equal, which only holds reliably if the
schema is closed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import ConfigFile


class PerformanceMetrics(BaseModel):
    """Per-execution timing and token-count measurements.

    ``ttft_ms`` and the token-count fields are nullable because the
    Ollama_Server does not always populate them (Requirement 6.5). A
    missing value must not be coerced to ``0`` — ``0`` is a valid
    measurement that means "zero tokens" or "no delay" and would be
    indistinguishable from "unknown" if we silently defaulted to it.

    ``tokens_per_second`` is derived by the runner from ``response_tokens``
    and ``total_ms`` (Requirement 6.4, Property 10). It is stored on the
    model rather than recomputed on demand so that reports remain stable
    even if the derivation rule changes in a future version.
    """

    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = Field(
        default=None,
        description=(
            "Wall-clock time from request dispatch to the first streamed "
            "chunk, in milliseconds (Requirement 6.1). ``None`` when the "
            "server omitted the timing (Requirement 6.5)."
        ),
    )
    total_ms: float = Field(
        ...,
        description=(
            "Total response time from request dispatch to the final "
            "chunk, in milliseconds (Requirement 6.2)."
        ),
    )
    prompt_tokens: int | None = Field(
        default=None,
        description=(
            "Prompt token count reported by the Ollama_Server "
            "(Requirement 6.3). ``None`` when the server omitted it."
        ),
    )
    response_tokens: int | None = Field(
        default=None,
        description=(
            "Response token count reported by the Ollama_Server "
            "(Requirement 6.3). ``None`` when the server omitted it."
        ),
    )
    tokens_per_second: float | None = Field(
        default=None,
        description=(
            "Derived throughput: ``response_tokens / (total_ms / 1000)`` "
            "(Requirement 6.4). ``None`` when ``response_tokens`` is "
            "``None`` or ``total_ms`` is 0."
        ),
    )


class MetricResult(BaseModel):
    """Score produced by a single metric for a single Test_Case execution.

    A ``MetricResult`` is always carried on a :class:`TestCaseResult`,
    even when the metric itself raises during scoring (Requirement 7.5).
    In the error path the runner sets ``error`` to the raised message and
    ``passed`` to ``False``; ``score`` is still required and the runner
    records ``0.0`` by convention so the aggregation code can operate on
    a uniform numeric field.

    ``details`` is a free-form mapping reserved for metric-specific
    context (for example, the captured response for the
    ``response-capture`` metric, the matched regex group for
    ``regex-match``, or the JSON-Schema validation errors for
    ``json-schema-valid``). It intentionally has no internal schema here.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Identifier of the metric that produced this result.",
    )
    score: float = Field(
        ...,
        description="Numeric score in the metric-defined range (Requirement 7.3).",
    )
    passed: bool = Field(
        ...,
        description="Pass/fail classification against the metric's threshold (Requirement 7.4).",
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "Threshold used by the metric to classify ``score`` as "
            "passed. ``None`` for metrics that do not parametrise a "
            "threshold (e.g. boolean metrics)."
        ),
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metric-specific structured context. Free-form; validated "
            "by the metric implementation, not here."
        ),
    )
    error: str | None = Field(
        default=None,
        description=(
            "Populated when the metric raised during scoring "
            "(Requirement 7.5). ``passed`` is ``False`` in that case and "
            "the Run continues with the remaining metrics."
        ),
    )


class TestCaseResult(BaseModel):
    """Outcome of a single ``(model, test_case, repetition)`` execution.

    ``status`` is a four-value literal rather than a boolean because a
    Run distinguishes four outcome classes in both the report and the
    event stream (Property 9):

    * ``pass`` — every configured metric classified the response as passing.
    * ``fail`` — at least one metric classified the response as failing.
    * ``error`` — the Ollama_Server call itself failed after all retries,
      or a metric raised in a way that the runner treats as a hard error.
    * ``timeout`` — the Ollama_Server call exceeded ``ollama_timeout_s``.

    ``repetition`` is 1-indexed (the first repetition is ``1``, not ``0``)
    so that the value can be displayed to users without re-indexing and
    so that ``repetition <= RunConfig.repetitions`` is the natural bound.
    """

    __test__ = False  # pytest: do not treat this class as a test container.

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        description="Ollama model tag this execution ran against.",
    )
    suite: str = Field(
        ...,
        description="Name of the Evaluation_Suite the test case belongs to.",
    )
    test_case_id: str = Field(
        ...,
        description="``TestCase.id`` within the enclosing suite.",
    )
    repetition: int = Field(
        ...,
        description=(
            "1-indexed repetition number for this ``(model, test_case)`` "
            "pair. Must be >= 1."
        ),
    )
    status: Literal["pass", "fail", "error", "timeout"] = Field(
        ...,
        description="Four-valued outcome (see class docstring).",
    )
    response: str | None = Field(
        default=None,
        description=(
            "Raw model response text, when available. ``None`` for "
            "``error``/``timeout`` outcomes that never produced output."
        ),
    )
    error_message: str | None = Field(
        default=None,
        description=(
            "Human-readable error message for ``error``/``timeout`` "
            "outcomes. ``None`` for ``pass``/``fail``."
        ),
    )
    performance: PerformanceMetrics = Field(
        ...,
        description="Timing and token-count measurements for this execution.",
    )
    metrics: list[MetricResult] = Field(
        ...,
        description=(
            "One ``MetricResult`` per configured metric on the test "
            "case, in the order declared on the ``TestCase``."
        ),
    )

    @field_validator("repetition")
    @classmethod
    def _repetition_positive(cls, value: int) -> int:
        """Enforce 1-indexed repetitions; ``0`` is not a valid slot."""
        if value < 1:
            raise ValueError(
                f"TestCaseResult.repetition must be >= 1 (got {value})"
            )
        return value


class MetricAggregate(BaseModel):
    """Mean and population stddev for a single ``(model, metric)`` pair.

    Keyed by metric name inside :class:`ModelAggregate.metric_aggregates`.
    The aggregation itself is computed by :mod:`runner.aggregate` using
    :func:`statistics.fmean` and :func:`statistics.pstdev` (Property 15).

    ``stddev`` is the *population* standard deviation, not the sample
    variant. For a single-repetition Run (``count == 1``) this is the
    well-defined value ``0``; sample stddev would be undefined. Sticking
    to pstdev everywhere means the aggregator never has to special-case
    ``count == 1``.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(
        ...,
        description="Metric name this aggregate corresponds to.",
    )
    mean: float = Field(
        ...,
        description="Arithmetic mean of the per-repetition scores.",
    )
    stddev: float = Field(
        ...,
        description="Population standard deviation of the per-repetition scores. Must be >= 0.",
    )
    count: int = Field(
        ...,
        description="Number of per-repetition samples that fed this aggregate. Must be >= 0.",
    )

    @field_validator("stddev")
    @classmethod
    def _stddev_non_negative(cls, value: float) -> float:
        """A standard deviation is by definition non-negative."""
        if value < 0:
            raise ValueError(
                f"MetricAggregate.stddev must be >= 0 (got {value})"
            )
        return value

    @field_validator("count")
    @classmethod
    def _count_non_negative(cls, value: int) -> int:
        """Sample counts cannot be negative; ``0`` is allowed for metrics with no data."""
        if value < 0:
            raise ValueError(
                f"MetricAggregate.count must be >= 0 (got {value})"
            )
        return value


class ModelAggregate(BaseModel):
    """Aggregate statistics for a single evaluated model.

    One ``ModelAggregate`` is produced per entry in
    ``RunConfig.models``. Counters (``passed``/``failed``/``errored``/
    ``timed_out``) sum the per-execution ``TestCaseResult.status`` values
    filtered to this model. Performance means average across every
    ``(test_case, repetition)`` for this model; ``mean_ttft_ms`` and
    ``mean_tokens_per_second`` are nullable because the underlying
    per-execution fields are nullable (Requirement 6.5).

    ``metric_aggregates`` is keyed by metric name so the UI and
    comparison report can look up a metric without scanning a list
    (Property 19).
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        description="Ollama model tag these aggregates describe.",
    )
    passed: int = Field(
        ...,
        description="Number of executions with ``status == 'pass'``.",
    )
    failed: int = Field(
        ...,
        description="Number of executions with ``status == 'fail'``.",
    )
    errored: int = Field(
        ...,
        description="Number of executions with ``status == 'error'``.",
    )
    timed_out: int = Field(
        ...,
        description="Number of executions with ``status == 'timeout'``.",
    )
    mean_ttft_ms: float | None = Field(
        ...,
        description=(
            "Mean time-to-first-token in milliseconds, averaged across "
            "executions with a non-``None`` ``ttft_ms``. ``None`` when "
            "no execution reported TTFT."
        ),
    )
    mean_total_ms: float = Field(
        ...,
        description=(
            "Mean total response time in milliseconds, averaged across "
            "every execution for this model."
        ),
    )
    mean_tokens_per_second: float | None = Field(
        ...,
        description=(
            "Mean tokens-per-second, averaged across executions with a "
            "non-``None`` ``tokens_per_second``. ``None`` when no "
            "execution produced the derived value."
        ),
    )
    metric_aggregates: dict[str, MetricAggregate] = Field(
        ...,
        description="Per-metric mean/stddev aggregates keyed by metric name.",
    )


class ModelInfo(BaseModel):
    """Identifying metadata for an evaluated Ollama model.

    Populated from the Ollama_Server's ``/api/tags`` response at Run
    start (Requirement 2.5). ``digest`` and ``parameter_size`` are
    nullable because some Ollama_Server versions omit them and because
    a Run submitted against an inventory that is still pulling a model
    may run before the digest is observable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description='Ollama model tag, e.g. ``"llama3:8b"``.',
    )
    digest: str | None = Field(
        default=None,
        description="Content-addressed digest reported by the Ollama_Server.",
    )
    parameter_size: str | None = Field(
        default=None,
        description='Human-readable parameter count (e.g. ``"8B"``).',
    )


class ErrorSummaryEntry(BaseModel):
    """One entry in :class:`RunReport.error_summary` (Requirement 11.3).

    Emitted once per ``TestCaseResult`` whose ``status`` is ``error`` or
    ``timeout``. The entry repeats the identifying tuple from the
    underlying result so the summary section can be rendered on its own
    without cross-referencing ``RunReport.results`` (Property 22).
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        description="Ollama model tag the failed execution ran against.",
    )
    suite: str = Field(
        ...,
        description="Evaluation_Suite name the failed test case belongs to.",
    )
    test_case_id: str = Field(
        ...,
        description="``TestCase.id`` of the failed execution.",
    )
    repetition: int = Field(
        ...,
        description="1-indexed repetition number of the failed execution.",
    )
    error_message: str = Field(
        ...,
        description="Human-readable error message from the underlying result.",
    )

    @field_validator("repetition")
    @classmethod
    def _repetition_positive(cls, value: int) -> int:
        """Match ``TestCaseResult.repetition``: must be >= 1."""
        if value < 1:
            raise ValueError(
                f"ErrorSummaryEntry.repetition must be >= 1 (got {value})"
            )
        return value


class RunSummary(BaseModel):
    """Compact counters attached to the terminal ``run-completed`` event.

    Mirrors the counters exposed in the UI progress panel (Requirement
    15.5) and in the CLI run summary line. ``planned_executions`` is the
    upfront plan produced by the scheduler (``|models| * |test cases| *
    repetitions``); ``completed_executions`` is the number actually
    dispatched and stored. The two match when the Run completes
    normally; ``completed_executions < planned_executions`` is possible
    when the Run is aborted mid-flight (Requirement 11.4).
    """

    model_config = ConfigDict(extra="forbid")

    planned_executions: int = Field(
        ...,
        description="Total executions planned at Run start.",
    )
    completed_executions: int = Field(
        ...,
        description="Number of executions that produced a ``TestCaseResult``.",
    )
    passed: int = Field(
        ...,
        description="Number of results with ``status == 'pass'``.",
    )
    failed: int = Field(
        ...,
        description="Number of results with ``status == 'fail'``.",
    )
    errored: int = Field(
        ...,
        description="Number of results with ``status == 'error'``.",
    )
    timed_out: int = Field(
        ...,
        description="Number of results with ``status == 'timeout'``.",
    )


class RunReport(BaseModel):
    """Top-level Run_Report persisted to disk and returned by the REST API.

    A ``RunReport`` is the single source of truth for a completed Run
    (Requirement 8.4): it embeds the full submitted :class:`ConfigFile`,
    the identifying metadata for every evaluated model, every
    per-execution result, the per-model aggregates, and an error
    summary. Property 16 (Run_Report completeness) asserts this
    invariant; Property 18 (round-trip) asserts that
    ``RunReport.model_validate_json(r.model_dump_json()) == r`` for
    every valid ``r``.

    ``status`` is a five-value literal covering the full state machine:

    * ``pending`` — the Run has been created but dispatch has not started.
    * ``running`` — at least one execution is in flight or queued.
    * ``completed`` — every planned execution has produced a result.
    * ``aborted`` — the Run was stopped by a cancel request or signal
      before all planned executions completed (Requirement 11.4).
    * ``failed`` — the Run could not start (e.g. missing models in
      preflight, Property 5) and no per-execution results exist.

    The ``ended_at >= started_at`` invariant is enforced in a model
    validator rather than field-by-field because it is a cross-field
    constraint (Property 16).
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        ...,
        description="Unique identifier for this Run (Property 25). Must be non-empty.",
    )
    backend_version: str = Field(
        ...,
        description="Version string of the Backend that produced this report.",
    )
    ollama_version: str | None = Field(
        ...,
        description=(
            "Ollama_Server version string at Run start. ``None`` when "
            "the Backend could not reach the server before failing."
        ),
    )
    started_at: datetime = Field(
        ...,
        description="UTC timestamp when the Run started.",
    )
    ended_at: datetime | None = Field(
        ...,
        description=(
            "UTC timestamp when the Run reached a terminal state. "
            "``None`` while the Run is still ``pending`` or ``running``."
        ),
    )
    status: Literal["pending", "running", "completed", "aborted", "failed"] = Field(
        ...,
        description="Terminal or in-flight state of the Run (see class docstring).",
    )
    config: ConfigFile = Field(
        ...,
        description="Full submitted ``ConfigFile`` (Requirement 8.4).",
    )
    models: list[ModelInfo] = Field(
        ...,
        description="One ``ModelInfo`` per evaluated model (Requirement 2.5).",
    )
    results: list[TestCaseResult] = Field(
        ...,
        description="One ``TestCaseResult`` per ``(model, test_case, repetition)``.",
    )
    aggregates: list[ModelAggregate] = Field(
        ...,
        description="One ``ModelAggregate`` per evaluated model (Requirement 8.2).",
    )
    error_summary: list[ErrorSummaryEntry] = Field(
        ...,
        description=(
            "One entry per result with ``status`` in ``{'error', "
            "'timeout'}`` (Requirement 11.3)."
        ),
    )

    @field_validator("run_id")
    @classmethod
    def _run_id_non_empty(cls, value: str) -> str:
        """Reject blank ``run_id`` values; they are used as primary keys."""
        if not value or not value.strip():
            raise ValueError("RunReport.run_id must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _ended_at_after_started_at(self) -> RunReport:
        """Enforce ``ended_at is None or ended_at >= started_at`` (Property 16)."""
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError(
                "RunReport.ended_at must be >= started_at "
                f"(got started_at={self.started_at!r}, ended_at={self.ended_at!r})"
            )
        return self


__all__ = [
    "ErrorSummaryEntry",
    "MetricAggregate",
    "MetricResult",
    "ModelAggregate",
    "ModelInfo",
    "PerformanceMetrics",
    "RunReport",
    "RunSummary",
    "TestCaseResult",
]
