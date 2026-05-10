"""Per-response metric scoring with error isolation (Requirement 7.5).

This module owns the single rule stated in Requirement 7.5: *a metric
that raises during scoring must not crash the Test_Case — it must
produce a :class:`MetricResult` with ``error`` populated and the other
metrics must still run*. The scheduler delegates to
:func:`score_all_metrics` once per ``(model, test_case, repetition)``
execution to score every metric configured on the :class:`TestCase`.

Contract
--------

``score_all_metrics(response, test_case, metric_configs)`` returns a
list of :class:`MetricResult` objects one-for-one with
``metric_configs``, in the same order. For each config:

* If the metric resolves via :func:`get_metric` and ``score`` returns a
  :class:`MetricResult`, that result is passed through verbatim.
* If the metric lookup or ``score`` call raises **any** exception, the
  exception is caught and converted into a :class:`MetricResult` with
  ``score=0.0``, ``passed=False``, and ``error=str(exc)``. Other
  metrics in the list are unaffected — Property 14 (metric error
  isolation) asserts this invariant.

``UnknownMetricError`` from the registry is treated like any other
metric-side raise: the result is recorded as an error with the missing
name in the message. The runner could alternately refuse to dispatch
the Run when an unknown metric is referenced, but that policy is the
suite-loader's responsibility (it rejects unknown metric names at load
time); by the time we are here, a missing metric is a runtime
surprise worth surfacing in the per-result error rather than aborting.

Rationale for wrapping every exception
---------------------------------------

We deliberately catch :class:`Exception` rather than
:class:`ValueError` only: a buggy metric implementation might raise
anything, and the runner's job is to isolate *the Run* from metric
bugs. :class:`KeyboardInterrupt` and :class:`SystemExit` inherit from
:class:`BaseException`, not :class:`Exception`, so they still
propagate — the signal-handler path in Task 12.3 can still cancel the
Run cleanly.
"""

from __future__ import annotations

from ..metrics import get_metric
from ..metrics.base import MetricContext
from ..models import MetricResult
from ..suites.models import MetricConfig, TestCase


async def score_all_metrics(
    response: str,
    test_case: TestCase,
    metric_configs: list[MetricConfig],
    *,
    model: str = "",
    suite: str = "",
    judge_client: object | None = None,
    judge_model: str | None = None,
) -> list[MetricResult]:
    """Score every metric in ``metric_configs`` against ``response``.

    Returns a list of :class:`MetricResult` the same length as
    ``metric_configs``, in the same order. Metrics that raise during
    scoring produce an error result instead of propagating; metrics
    that succeed return their result verbatim.

    Args:
        response: Raw model response text.
        test_case: The :class:`TestCase` driving the execution.
            Passed to each metric via :class:`MetricContext`.
        metric_configs: Metric configurations to score, in order. This
            is the caller's declaration order — the result list
            preserves it (Property 14).
        model: Ollama model tag for :class:`MetricContext.model`. Kept
            as a keyword with an empty-string default so unit tests
            and property tests can call this helper without needing
            the full Run context.
        suite: Suite name for :class:`MetricContext.suite`.
        judge_client: Judge client for metrics that need one
            (``llm-as-judge``). ``None`` when no judge metric is
            configured; per-metric validation raises if a metric
            demands one and gets ``None``.
        judge_model: Judge model tag; see :class:`MetricContext`.

    Returns:
        A list of :class:`MetricResult` in input order with
        ``len(result) == len(metric_configs)``.
    """
    results: list[MetricResult] = []
    for metric_config in metric_configs:
        ctx = MetricContext(
            model=model,
            suite=suite,
            test_case=test_case,
            metric_config=metric_config,
            judge_client=judge_client,
            judge_model=judge_model,
        )
        try:
            metric = get_metric(metric_config.name)
            result = await metric.score(response, ctx)
        except Exception as exc:  # noqa: BLE001 — error isolation is the whole point.
            # Preserve the metric name in the result even on error so
            # the runner can correlate this entry with the config that
            # produced it. ``score=0.0`` matches the convention
            # documented on :class:`MetricResult`: error results
            # record zero so the aggregation code can treat
            # ``score`` as a uniform numeric field without a None
            # branch.
            results.append(
                MetricResult(
                    name=metric_config.name,
                    score=0.0,
                    passed=False,
                    error=str(exc),
                )
            )
        else:
            results.append(result)

    return results


__all__ = ["score_all_metrics"]
