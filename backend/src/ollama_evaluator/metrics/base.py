"""Metric protocol and shared context model for the metric framework.

This module defines two small contracts that every scoring metric — built-in
or user-extended — must satisfy (Requirements 7.3, 7.4):

* :class:`MetricContext` — a Pydantic carrier bundling everything a metric
  implementation needs beyond the raw response string. It lets metric
  implementations stay side-effect free (no global lookups, no module-level
  state) while still having access to the enclosing Run's identifying
  metadata (``model``, ``suite``), the full :class:`TestCase` (for
  ``expected_output``, ``reference_data``, etc.), the metric's own
  :class:`MetricConfig` (for per-metric ``params``), and — for metrics
  such as ``llm-as-judge`` — a judge client and model tag.

* :class:`Metric` — a :class:`typing.Protocol` that every concrete metric
  satisfies structurally. It exposes a ``name`` class attribute used as
  the registry key (see :mod:`ollama_evaluator.metrics`) and an
  asynchronous :meth:`Metric.score` method that returns a
  :class:`MetricResult` from :mod:`ollama_evaluator.models`.

Design note: the design document's §Metric framework shows a synchronous
``score(response, test_case, ctx)`` signature. We follow the more
prescriptive signature from Task 5.1 here: ``score`` is ``async`` (so
LLM-as-judge metrics can ``await`` the judge client without bolting on an
adapter), and ``test_case`` is carried inside :class:`MetricContext`
rather than passed as a separate argument. This keeps every metric's
``score`` callsite uniform: ``await metric.score(response, ctx)``.

``MetricContext.judge_client`` is typed :class:`typing.Any` rather than
:class:`ollama_evaluator.ollama.client.OllamaClient` to avoid a cyclic
import (metrics are imported from the runner, which in turn imports the
Ollama client which depends on nothing under ``metrics/``). Runtime
guarantees about the object's shape are the responsibility of the metric
implementation that consumes it. ``arbitrary_types_allowed=True`` lets
Pydantic keep the non-BaseModel client instance as-is; ``extra="forbid"``
still rejects unknown fields so a typo in ``MetricContext(...)``
construction surfaces immediately.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..models import MetricResult
from ..suites.models import MetricConfig, TestCase


class MetricContext(BaseModel):
    """Per-execution context handed to every :meth:`Metric.score` call.

    A fresh ``MetricContext`` is constructed by the scheduler for each
    ``(model, test_case, repetition, metric)`` invocation. It bundles the
    enclosing Run's identifying metadata together with the specific
    ``TestCase`` and ``MetricConfig`` that drove this particular metric
    call, so the metric implementation has everything it needs in a
    single, immutable argument.

    ``judge_client`` / ``judge_model`` are populated only for metrics
    that need to call out to an LLM (the ``llm-as-judge`` metric, for
    example). They are optional so that purely local metrics
    (``exact-match``, ``regex-match``, ``json-schema-valid``, …) can
    accept the same context shape without carrying a judge client they
    would never use.

    The model uses ``arbitrary_types_allowed=True`` so that a concrete
    Ollama client instance (which is not a Pydantic model) can be passed
    through ``judge_client``; we intentionally do not import the client
    class here to keep :mod:`ollama_evaluator.metrics` free of a cycle
    through :mod:`ollama_evaluator.ollama`. ``extra="forbid"`` remains in
    effect — the only acceptable extension point for metric-specific
    configuration is :attr:`MetricConfig.params`, not extra
    ``MetricContext`` fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: str
    """Ollama model tag this execution ran against (e.g. ``"llama3:8b"``)."""

    suite: str
    """Name of the enclosing :class:`~ollama_evaluator.suites.models.EvaluationSuite`."""

    test_case: TestCase
    """The :class:`TestCase` that produced ``response``. Carries ``expected_output``,
    ``reference_data``, tags, and per-case generation overrides that a metric may read."""

    metric_config: MetricConfig
    """The specific :class:`MetricConfig` entry that drove this metric invocation.
    Metric implementations read their metric-specific parameters from
    ``metric_config.params`` and validate them there."""

    judge_client: Any | None = None
    """Populated for ``llm-as-judge`` (and any other metric that needs to call the
    Ollama server). Typed ``Any`` deliberately to avoid importing the client class
    here — the importing module lives below :mod:`ollama_evaluator.metrics` and
    introducing that dependency would create a cycle."""

    judge_model: str | None = None
    """Ollama model tag the judge metric should use. ``None`` when no
    judge metric is configured for the enclosing Run."""


@runtime_checkable
class Metric(Protocol):
    """Structural contract every scoring metric satisfies.

    A metric implementation is any object exposing:

    * A ``name`` attribute — the stable, user-visible identifier that
      keys the metric registry (see :mod:`ollama_evaluator.metrics`).
      The registry rejects registrations whose ``name`` collides with
      a different object only when re-registration is not intended;
      by design, re-registering the same name replaces the prior entry
      (see the registry docstring).

    * An ``async`` :meth:`score` method that takes the raw response
      string produced by the Ollama model plus a :class:`MetricContext`
      and returns a :class:`MetricResult`.

    ``Metric`` is a :class:`typing.Protocol` so that metrics can be
    plain classes, dataclasses, or module-level singletons: the registry
    does not require inheritance. It is decorated with
    :func:`typing.runtime_checkable` so that tests (and, if needed, the
    registry) can assert protocol conformance at runtime via
    :func:`isinstance`.
    """

    name: str

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        """Score ``response`` against ``ctx.test_case`` using ``ctx.metric_config``.

        Implementations MUST NOT raise on malformed model output; they
        should instead return a :class:`MetricResult` with ``passed=False``
        and a populated ``error`` field (Requirement 7.5). Runtime
        errors that the metric does not handle will be caught at the
        runner boundary and converted into a ``MetricResult`` with
        ``error`` set.
        """
        ...


__all__ = [
    "Metric",
    "MetricContext",
    "MetricResult",
]
