"""Metric registry and public re-exports for the metric framework.

This module owns the process-global registry of :class:`Metric`
implementations and exposes the small API the rest of the Backend uses
to register, look up, and enumerate metrics by name:

* :func:`register_metric` — add (or replace) a metric implementation
  keyed by its ``name`` attribute.
* :func:`get_metric` — retrieve a previously registered metric; raises
  :class:`UnknownMetricError` (a ``KeyError`` subclass) when the name
  is not registered.
* :func:`list_metrics` — enumerate every registered metric name in
  sorted order, for deterministic CLI output and diagnostics.

It also re-exports :class:`Metric`, :class:`MetricContext`, and the
:class:`MetricResult` symbol from :mod:`ollama_evaluator.models`, so
metric implementations and callers can write a single import
(``from ollama_evaluator.metrics import Metric, MetricContext,
MetricResult``) without reaching into submodules.

Re-registration semantics
-------------------------
Calling :func:`register_metric` twice with metrics that share the same
``name`` replaces the prior entry. This is intentional:

* It lets tests install lightweight fakes for metrics without needing
  a special "unregister" hook (see ``tests/unit/test_metrics_base.py``
  for the pattern: a ``clean_registry`` fixture that snapshots and
  restores the registry between tests).
* It lets the CLI ``serve`` and library entry points re-import
  built-in metric modules without crashing when a reload happens
  (e.g. under ``uvicorn --reload``).

The registry is *not* thread-safe by design. All callers live inside a
single asyncio event loop (the FastAPI worker or the CLI ``run`` task);
concurrent registration from multiple OS threads is out of scope.
"""

from __future__ import annotations

from .base import Metric, MetricContext, MetricResult


class UnknownMetricError(KeyError):
    """Raised by :func:`get_metric` when the requested metric name is not registered.

    Subclasses :class:`KeyError` so callers that use ``dict``-style
    patterns (``try: registry[name] except KeyError``) still catch the
    error, while giving targeted callers a more descriptive class to
    match on when they want to surface a better error message to the
    user (e.g. the suite validator that reports unknown metric names on
    :class:`~ollama_evaluator.suites.models.MetricConfig`).

    ``str(UnknownMetricError("foo"))`` returns the missing name so the
    error is useful when logged directly. This matches the ``KeyError``
    convention of carrying the missing key as the single argument.
    """


_REGISTRY: dict[str, Metric] = {}
"""Process-global name -> :class:`Metric` mapping.

Private by convention. Callers should go through :func:`register_metric`,
:func:`get_metric`, and :func:`list_metrics` rather than mutating this
dict directly, because future versions may add validation (for example,
rejecting metrics whose ``name`` attribute does not match the
registration key) that the public API can enforce in one place.
"""


def register_metric(metric: Metric) -> None:
    """Register ``metric`` under its ``metric.name`` in the process registry.

    Re-registration with the same ``name`` replaces the prior
    implementation (see the module docstring for rationale). This is
    load-bearing for tests and for reload-based development flows.

    The function accepts a :class:`Metric`-shaped object; there is no
    class inheritance requirement. At the moment no structural
    validation is performed beyond reading ``metric.name``; future
    versions may assert ``callable(metric.score)`` and a non-empty
    ``name`` here.
    """
    _REGISTRY[metric.name] = metric


def get_metric(name: str) -> Metric:
    """Return the metric registered under ``name``.

    Raises :class:`UnknownMetricError` (a :class:`KeyError` subclass)
    with ``name`` as the sole argument when no such metric is
    registered. The error's ``args[0]`` is always the missing name, so
    callers that surface validation errors to the user (the suite
    loader, the API error envelope) can render it directly.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - trivial re-raise
        raise UnknownMetricError(name) from exc


def list_metrics() -> list[str]:
    """Return every registered metric name in lexicographically sorted order.

    Returning a sorted list (rather than a ``dict_keys`` view) makes
    CLI output, log lines, and test assertions deterministic regardless
    of registration order. The list is a fresh copy, so callers may
    mutate it freely.
    """
    return sorted(_REGISTRY)


__all__ = [
    "Metric",
    "MetricContext",
    "MetricResult",
    "UnknownMetricError",
    "get_metric",
    "list_metrics",
    "register_builtin_metrics",
    "register_judge_metric",
    "register_metric",
]


# ---------------------------------------------------------------------------
# Auto-registration of built-in metrics
# ---------------------------------------------------------------------------
#
# Importing the package (``import ollama_evaluator.metrics``) auto-registers
# the five built-in string/structural metrics (``exact-match``, ``regex-match``,
# ``contains``, ``json-schema-valid``, ``length-range``) plus the
# ``llm-as-judge`` metric from :mod:`ollama_evaluator.metrics.judge` so that
# the rest of the Backend can resolve them via :func:`get_metric` without
# needing to call the ``register_*`` helpers explicitly.
#
# The imports are placed here, at the *bottom* of the module, so that
# :func:`register_metric` is already defined by the time ``builtin.py`` and
# ``judge.py`` are executed. Those modules themselves do a late import of
# ``register_metric`` from this package to break what would otherwise be a
# circular import at load time.
#
# Both registration helpers are idempotent (the registry has replacement
# semantics — see the module docstring), so the extra ``_BUILTINS_REGISTERED``
# guard below is a micro-optimisation rather than a correctness requirement.
# It keeps a second import (for example after ``importlib.reload``) cheap and
# avoids constructing six singleton objects just to throw them away.
from .builtin import register_builtin_metrics  # noqa: E402
from .judge import register_judge_metric  # noqa: E402

_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    """Register built-in and judge metrics exactly once per interpreter."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    register_builtin_metrics()
    register_judge_metric()
    _BUILTINS_REGISTERED = True


_ensure_builtins_registered()
