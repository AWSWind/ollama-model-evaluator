"""Comparison_Report builder for two Run_Reports.

Given two :class:`~ollama_evaluator.models.RunReport` instances
``a`` and ``b``, :func:`compare` produces a
:class:`ComparisonReport` that lists, per ``(model, metric)`` pair
present in both reports, the two mean scores and their signed
difference (``diff = mean_b - mean_a``), and, per ``model``
present in both reports, the difference in mean tokens-per-second
and mean total response time. Requirement 9 captures this
behaviour; Property 19 pins the keyset and arithmetic rules.

The design's "comparison over intersection" text is load-bearing:

* ``metric_diffs`` is keyed *exactly* by the intersection of
  ``{(model, metric)}`` across both reports. Any model/metric
  present only in ``a`` or only in ``b`` is dropped.
* ``performance_diffs`` is keyed *exactly* by the intersection of
  ``{model}`` across both reports.
* :class:`NoCommonDimensionsError` is raised **iff** both
  intersections are empty. One intersection being empty is fine;
  only the simultaneous empty case is an error.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Data Models / Comparison_Report.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import RunReport


class NoCommonDimensionsError(Exception):
    """Raised when both ``metric_diffs`` and ``performance_diffs`` are empty.

    The Comparison_Report has nothing to say when the two Runs
    share no ``(model, metric)`` pair and no model at all. Callers
    — notably the REST handler for ``/api/compare`` (Task 17.2) —
    translate this into a ``400 no_common_dimensions`` envelope
    (Requirement 9.4).
    """


class MetricDiff(BaseModel):
    """One row of :class:`ComparisonReport.metric_diffs`.

    ``diff`` is ``mean_b - mean_a``. Positive means Run B scored
    higher than Run A on the given metric; negative means the
    opposite. The field is a ``float`` (not ``float | None``)
    because the enclosing keyset is the *intersection* — a
    ``MetricDiff`` is only emitted when both reports have a value.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model tag the metric was scored for.")
    metric: str = Field(..., description="Metric name.")
    mean_a: float = Field(..., description="Mean score in Run A.")
    mean_b: float = Field(..., description="Mean score in Run B.")
    diff: float = Field(..., description="``mean_b - mean_a``.")


class PerformanceDiff(BaseModel):
    """One row of :class:`ComparisonReport.performance_diffs`.

    Populated per model present in both reports. ``*_tps_*`` fields
    are nullable because the underlying per-model mean is nullable
    (Requirement 6.5). ``tps_diff`` is ``None`` when either side is
    ``None``; ``total_ms_diff`` is always a concrete ``float``
    because ``mean_total_ms`` is non-nullable on
    :class:`ModelAggregate`.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model tag.")
    mean_tokens_per_second_a: float | None = Field(
        ..., description="Run A's mean tokens-per-second for this model."
    )
    mean_tokens_per_second_b: float | None = Field(
        ..., description="Run B's mean tokens-per-second for this model."
    )
    mean_total_ms_a: float = Field(
        ..., description="Run A's mean total response time for this model (ms)."
    )
    mean_total_ms_b: float = Field(
        ..., description="Run B's mean total response time for this model (ms)."
    )
    tps_diff: float | None = Field(
        ...,
        description=(
            "``mean_tokens_per_second_b - mean_tokens_per_second_a``; "
            "``None`` when either side is ``None``."
        ),
    )
    total_ms_diff: float = Field(
        ..., description="``mean_total_ms_b - mean_total_ms_a``."
    )


class ComparisonReport(BaseModel):
    """Top-level Comparison_Report returned by :func:`compare`.

    ``metric_diffs`` and ``performance_diffs`` are lists (not dicts)
    to match the design's §Data Models / Comparison_Report table and
    because list ordering is stable across re-renderings in the UI
    (Property 41). Ordering is deterministic: the lists are sorted
    lexicographically by key tuple so two independent invocations on
    equal inputs produce equal reports.
    """

    model_config = ConfigDict(extra="forbid")

    run_a: str = Field(..., description="``RunReport.run_id`` of the base Run.")
    run_b: str = Field(..., description="``RunReport.run_id`` of the comparison Run.")
    metric_diffs: list[MetricDiff] = Field(
        ..., description="Per-``(model, metric)`` diffs over the intersection."
    )
    performance_diffs: list[PerformanceDiff] = Field(
        ..., description="Per-model performance diffs over the intersection."
    )


def compare(a: RunReport, b: RunReport) -> ComparisonReport:
    """Build a :class:`ComparisonReport` of the two Run_Reports.

    Raises:
        NoCommonDimensionsError: when ``a`` and ``b`` share no
            ``(model, metric)`` pair and no model at all.
    """
    # Build lookup dicts keyed by model name; ``ModelAggregate``
    # exposes ``metric_aggregates`` keyed by metric name already.
    a_models = {agg.model: agg for agg in a.aggregates}
    b_models = {agg.model: agg for agg in b.aggregates}

    # Metric intersection: for each shared model, take the
    # intersection of metric names present in both sides.
    metric_diffs: list[MetricDiff] = []
    for model in sorted(a_models.keys() & b_models.keys()):
        a_metrics = a_models[model].metric_aggregates
        b_metrics = b_models[model].metric_aggregates
        for metric in sorted(a_metrics.keys() & b_metrics.keys()):
            mean_a = a_metrics[metric].mean
            mean_b = b_metrics[metric].mean
            metric_diffs.append(
                MetricDiff(
                    model=model,
                    metric=metric,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    diff=mean_b - mean_a,
                )
            )

    # Performance intersection: one row per shared model.
    performance_diffs: list[PerformanceDiff] = []
    for model in sorted(a_models.keys() & b_models.keys()):
        agg_a = a_models[model]
        agg_b = b_models[model]
        tps_a = agg_a.mean_tokens_per_second
        tps_b = agg_b.mean_tokens_per_second
        tps_diff = (
            (tps_b - tps_a) if (tps_a is not None and tps_b is not None) else None
        )
        performance_diffs.append(
            PerformanceDiff(
                model=model,
                mean_tokens_per_second_a=tps_a,
                mean_tokens_per_second_b=tps_b,
                mean_total_ms_a=agg_a.mean_total_ms,
                mean_total_ms_b=agg_b.mean_total_ms,
                tps_diff=tps_diff,
                total_ms_diff=agg_b.mean_total_ms - agg_a.mean_total_ms,
            )
        )

    if not metric_diffs and not performance_diffs:
        raise NoCommonDimensionsError(
            f"runs {a.run_id!r} and {b.run_id!r} share no models or metrics"
        )

    return ComparisonReport(
        run_a=a.run_id,
        run_b=b.run_id,
        metric_diffs=metric_diffs,
        performance_diffs=performance_diffs,
    )


__all__ = [
    "ComparisonReport",
    "MetricDiff",
    "NoCommonDimensionsError",
    "PerformanceDiff",
    "compare",
]
