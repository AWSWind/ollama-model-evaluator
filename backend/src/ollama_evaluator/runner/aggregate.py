"""Per-run aggregation of :class:`TestCaseResult`s into :class:`ModelAggregate`s.

This module takes the flat list of per-execution results produced by the
scheduler (one per ``(model, test_case, repetition)`` tuple) and rolls
them up into the per-model summaries persisted on :class:`RunReport`
(Requirements 7.6, 8.2). Two public entry points:

* :func:`aggregate_metric_scores` — small numeric helper returning
  ``(mean, stddev, count)`` over a list of floats. Driven by Property 15
  (repetition aggregates): ``mean == fmean([s_1 … s_R])`` and
  ``stddev == pstdev([s_1 … s_R])``. Used by :func:`build_model_aggregate`
  to compute per-``(model, metric)`` :class:`MetricAggregate`.

* :func:`build_model_aggregate` — constructs a :class:`ModelAggregate`
  for a single model given every :class:`TestCaseResult` that ran
  against it. Counts the four outcome statuses, averages the
  performance fields (skipping ``None`` ones per Requirement 6.5),
  and computes a :class:`MetricAggregate` for every metric that
  appeared on any of the model's results.

* :func:`build_all_aggregates` — top-level convenience that groups an
  unordered mixed-model result list by model, sorts model names
  lexicographically (so the ``RunReport.aggregates`` ordering is
  deterministic across runs with the same model set), and delegates
  to :func:`build_model_aggregate`.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Components / §Data Models / Run_Report and §Runner and Scheduler.

Numeric conventions
-------------------

* **Population stddev (``pstdev``), not sample stddev.** Matches
  :class:`MetricAggregate`'s field-level documentation: for a
  single-repetition Run (``count == 1``) pstdev returns ``0.0``,
  whereas stdev is undefined. Sticking to pstdev everywhere means the
  aggregator never has to special-case ``count == 1``; the stddev
  bar on a single-sample metric in the UI simply renders as ``0``.

* **Fmean, not mean.** :func:`statistics.fmean` is explicitly
  documented to be faster and always return :class:`float` for float
  inputs. Using it here matches the pattern in Property 15.

* **Empty input → ``(0.0, 0.0, 0)``.** The caller-facing contract for
  :func:`aggregate_metric_scores` is documented on the function: an
  empty list returns an all-zeros triple so the aggregator can emit a
  :class:`MetricAggregate` for metrics that errored on every execution
  (``count == 0``) without raising. :class:`MetricAggregate` validates
  ``count >= 0`` and ``stddev >= 0``, so the all-zeros triple is a
  valid input to its constructor.

Metric inclusion rule
---------------------

A metric contributes to its per-metric aggregate only when the
:class:`MetricResult` for that metric carries ``error is None`` on the
result. Results where the metric raised (``error != None``) record a
score of ``0.0`` by convention (see :class:`MetricResult` docstring in
:mod:`ollama_evaluator.models`), which is a placeholder rather than a
measured score — averaging zeros from error paths into the aggregate
would systematically bias the metric's mean downward. Property 14
(metric error isolation) is the dedicated check for this invariant;
the filter here is the implementation that makes it hold.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from ..models import MetricAggregate, ModelAggregate, TestCaseResult


def aggregate_metric_scores(scores: list[float]) -> tuple[float, float, int]:
    """Return ``(mean, stddev, count)`` for the per-repetition ``scores``.

    Empty list contract
    -------------------
    An empty ``scores`` list returns ``(0.0, 0.0, 0)`` so
    :func:`build_model_aggregate` can emit a :class:`MetricAggregate`
    for metrics that errored on every execution. :class:`MetricAggregate`
    validates ``count >= 0`` and ``stddev >= 0``, so the all-zeros
    triple is a valid input to its constructor. This also matches the
    shape that Property 15 asserts at the empty-list edge.

    Single-element contract
    -----------------------
    With one sample, :func:`statistics.pstdev` returns ``0.0``
    (population stddev of a single value is well-defined — it is the
    distance from the value to the mean, which is itself the value).
    We short-circuit through :func:`statistics.fmean` / :func:`pstdev`
    anyway so the behaviour is identical to the general case; the
    shortcut here is only for the empty list.

    Args:
        scores: Per-repetition metric scores from successful
            (non-``error``) executions.

    Returns:
        ``(mean, stddev, count)`` where ``mean`` is
        :func:`statistics.fmean`, ``stddev`` is
        :func:`statistics.pstdev` (both guaranteed finite for finite
        inputs), and ``count == len(scores)``.
    """
    count = len(scores)
    if count == 0:
        return (0.0, 0.0, 0)
    mean = statistics.fmean(scores)
    # ``pstdev`` is documented to return 0.0 for a single-element input;
    # no special-case needed here.
    stddev = statistics.pstdev(scores)
    return (mean, stddev, count)


def build_model_aggregate(
    model: str, results: list[TestCaseResult]
) -> ModelAggregate:
    """Roll ``results`` up into a :class:`ModelAggregate` for ``model``.

    The input ``results`` must all reference the same ``model`` — this
    is not re-validated here; :func:`build_all_aggregates` enforces the
    partition before calling this helper.

    Aggregation rules:

    * **Status counters.** Count ``pass``, ``fail``, ``error``, and
      ``timeout`` outcomes. Sum matches ``len(results)``.

    * **Mean performance fields.** Averaged across results with a
      non-``None`` value per Requirement 6.5. ``mean_ttft_ms`` and
      ``mean_tokens_per_second`` return ``None`` when *no* result
      reported the field. ``mean_total_ms`` averages over *every*
      result (``total_ms`` is non-nullable on :class:`PerformanceMetrics`)
      and returns ``0.0`` when ``results`` is empty.

    * **Per-metric aggregates.** For every metric name appearing in
      any ``results[i].metrics``, collect the scores from results where
      that metric had ``error is None`` and pass them through
      :func:`aggregate_metric_scores`. Metrics that appeared only in
      error paths produce a :class:`MetricAggregate` with
      ``count == 0``.
    """
    # Status counters ---------------------------------------------------
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errored = sum(1 for r in results if r.status == "error")
    timed_out = sum(1 for r in results if r.status == "timeout")

    # Performance aggregates -------------------------------------------
    ttft_samples = [r.performance.ttft_ms for r in results if r.performance.ttft_ms is not None]
    tps_samples = [
        r.performance.tokens_per_second
        for r in results
        if r.performance.tokens_per_second is not None
    ]
    mean_ttft_ms = statistics.fmean(ttft_samples) if ttft_samples else None
    mean_tokens_per_second = statistics.fmean(tps_samples) if tps_samples else None

    total_ms_samples = [r.performance.total_ms for r in results]
    mean_total_ms = statistics.fmean(total_ms_samples) if total_ms_samples else 0.0

    # Per-metric aggregates --------------------------------------------
    # Insertion-order dict (Python 3.7+) so metrics appear in the order
    # the scheduler first observed them. Since we read in the given
    # ``results`` order and iterate ``results[i].metrics`` in
    # declaration order, the final dict reflects the first-encountered
    # ordering across results — stable for a given test-case config.
    per_metric_scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for m in result.metrics:
            # Only successful scorings contribute to the aggregate;
            # error results record ``score=0.0`` by convention which is
            # not a measured value (see module docstring).
            if m.error is None:
                per_metric_scores[m.name].append(m.score)
            else:
                # Record the metric name so it still appears in the
                # aggregate dict, even if every execution errored on
                # it. The score list stays empty and
                # ``aggregate_metric_scores([])`` returns
                # ``(0.0, 0.0, 0)``.
                per_metric_scores.setdefault(m.name, [])

    metric_aggregates: dict[str, MetricAggregate] = {}
    for name, scores in per_metric_scores.items():
        mean, stddev, count = aggregate_metric_scores(scores)
        metric_aggregates[name] = MetricAggregate(
            metric=name,
            mean=mean,
            stddev=stddev,
            count=count,
        )

    return ModelAggregate(
        model=model,
        passed=passed,
        failed=failed,
        errored=errored,
        timed_out=timed_out,
        mean_ttft_ms=mean_ttft_ms,
        mean_total_ms=mean_total_ms,
        mean_tokens_per_second=mean_tokens_per_second,
        metric_aggregates=metric_aggregates,
    )


def build_all_aggregates(results: list[TestCaseResult]) -> list[ModelAggregate]:
    """Group ``results`` by model and build one :class:`ModelAggregate` each.

    Models are emitted in lexicographically sorted order so
    ``RunReport.aggregates`` is deterministic across runs with the same
    model set — this is the ordering the UI and the Markdown report
    rely on for stable comparisons (Requirement 8.3).

    An empty ``results`` list returns an empty list; there is no
    implicit "empty aggregate" placeholder because the runner knows
    which models were planned and can synthesise empty
    :class:`ModelAggregate`s itself if it needs to represent the
    "planned but nothing executed" state (e.g. during abort). That's a
    runner concern, not an aggregation concern.
    """
    grouped: dict[str, list[TestCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.model].append(result)

    return [
        build_model_aggregate(model, grouped[model]) for model in sorted(grouped)
    ]


__all__ = [
    "aggregate_metric_scores",
    "build_all_aggregates",
    "build_model_aggregate",
]
