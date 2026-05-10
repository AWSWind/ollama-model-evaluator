"""Unit tests for the Metric registry and base protocol (Task 5.1).

These tests cover the small public surface exposed by
:mod:`ollama_evaluator.metrics`:

* :func:`register_metric` stores a metric keyed by its ``name`` attribute.
* :func:`get_metric` returns the previously registered object.
* Re-registering the same name **replaces** the prior implementation.
  This is intentional (see the module docstring on
  :mod:`ollama_evaluator.metrics`) so tests can install fakes and the
  CLI can tolerate module reloads without crashing.
* :func:`get_metric` raises :class:`UnknownMetricError` with the
  missing name as its argument when the metric is not registered.
* :func:`list_metrics` returns registered names in lexicographic order
  regardless of registration order, so CLI output and log lines stay
  deterministic.

A ``clean_registry`` fixture snapshots and restores the registry
around every test. The registry is module-global state, so isolation is
required to keep tests independent — otherwise a registration leaked
from one test could satisfy another test's missing-metric assertion.

The tests also sanity-check that :class:`MetricContext` (a Pydantic
model) accepts the expected fields and forbids extras, since the
registry is useless without a well-defined context type to hand to
``Metric.score``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from ollama_evaluator.metrics import (
    Metric,
    MetricContext,
    MetricResult,
    UnknownMetricError,
    _REGISTRY,
    get_metric,
    list_metrics,
    register_metric,
)
from ollama_evaluator.suites.models import MetricConfig, TestCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Snapshot the registry before each test and restore it afterwards.

    The Metric registry lives at module scope on
    :mod:`ollama_evaluator.metrics` and is intentionally mutable
    (re-registration replaces the prior entry). Tests in this module
    add and replace entries; without restoration those changes would
    leak into unrelated tests and flake ``get_metric("missing")``
    assertions.

    We snapshot by copying the underlying dict, mutate through the
    public API, and restore the snapshot in a ``finally`` so that a
    failing test still leaves the registry in its pre-test state.
    """
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyMetric:
    """Minimal :class:`Metric`-shaped object used throughout these tests.

    Carries an identifying ``tag`` so that re-registration tests can
    tell the "old" and "new" instances apart.
    """

    def __init__(self, name: str, tag: str = "v1") -> None:
        self.name = name
        self.tag = tag

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        # The registry tests never actually call ``score``; the method is
        # present only to satisfy the ``Metric`` protocol when a test
        # opts into ``isinstance(m, Metric)`` via the runtime-checkable
        # decorator.
        return MetricResult(
            name=self.name,
            score=0.0,
            passed=True,
            details={"tag": self.tag, "response": response, "suite": ctx.suite},
        )


def _make_context(**overrides: Any) -> MetricContext:
    """Build a valid :class:`MetricContext` for exercising the protocol.

    Every required field is populated with a sensible default so tests
    only need to specify the field(s) they care about.
    """
    defaults: dict[str, Any] = {
        "model": "llama3:8b",
        "suite": "reasoning-basics",
        "test_case": TestCase(
            id="c1",
            prompt="What is 2 + 2?",
            metrics=[MetricConfig(name="exact-match")],
        ),
        "metric_config": MetricConfig(name="exact-match"),
    }
    defaults.update(overrides)
    return MetricContext(**defaults)


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_and_get(self) -> None:
        """A registered metric is returned verbatim by ``get_metric``."""
        dummy = _DummyMetric("dummy")
        register_metric(dummy)

        assert get_metric("dummy") is dummy

    def test_reregistration_replaces_prior_implementation(self) -> None:
        """Re-registering the same name replaces the prior object (documented behaviour).

        This is load-bearing: tests install fakes this way and the CLI
        tolerates module reloads without crashing. See the registry
        module docstring for the full rationale.
        """
        original = _DummyMetric("dummy", tag="v1")
        replacement = _DummyMetric("dummy", tag="v2")

        register_metric(original)
        register_metric(replacement)

        resolved = get_metric("dummy")
        assert resolved is replacement
        assert resolved.tag == "v2"
        # Only one entry exists under that name — replacement, not a duplicate.
        assert list_metrics() == ["dummy"]

    def test_register_distinct_names_coexist(self) -> None:
        """Different ``name`` values register independently."""
        a = _DummyMetric("alpha")
        b = _DummyMetric("beta")

        register_metric(a)
        register_metric(b)

        assert get_metric("alpha") is a
        assert get_metric("beta") is b


class TestGetMetric:
    def test_missing_metric_raises_unknown_metric_error(self) -> None:
        """``get_metric`` raises ``UnknownMetricError`` with the missing name."""
        with pytest.raises(UnknownMetricError) as excinfo:
            get_metric("nonexistent")

        # The missing name is carried as the single argument so callers
        # can surface it in user-facing messages without string-parsing
        # the exception.
        assert excinfo.value.args == ("nonexistent",)

    def test_unknown_metric_error_is_a_key_error(self) -> None:
        """``UnknownMetricError`` subclasses ``KeyError`` for dict-style patterns."""
        with pytest.raises(KeyError):
            get_metric("also-missing")

    def test_missing_metric_error_name_survives_after_other_registrations(self) -> None:
        """Registering other metrics does not make a missing name resolvable."""
        register_metric(_DummyMetric("present"))

        with pytest.raises(UnknownMetricError) as excinfo:
            get_metric("absent")

        assert excinfo.value.args == ("absent",)


class TestListMetrics:
    def test_empty_registry_returns_empty_list(self) -> None:
        assert list_metrics() == []

    def test_returns_names_in_sorted_order(self) -> None:
        """Order of registration must not affect the output order.

        Registering ``gamma`` then ``alpha`` then ``beta`` should still
        produce ``["alpha", "beta", "gamma"]``.
        """
        register_metric(_DummyMetric("gamma"))
        register_metric(_DummyMetric("alpha"))
        register_metric(_DummyMetric("beta"))

        assert list_metrics() == ["alpha", "beta", "gamma"]

    def test_returns_fresh_list_each_call(self) -> None:
        """Mutating the returned list must not affect subsequent calls."""
        register_metric(_DummyMetric("only"))

        first = list_metrics()
        first.append("sneaky")

        assert list_metrics() == ["only"]


# ---------------------------------------------------------------------------
# MetricContext shape
# ---------------------------------------------------------------------------


class TestMetricContext:
    def test_accepts_expected_fields(self) -> None:
        ctx = _make_context(judge_client="not-a-real-client", judge_model="llama3:8b")

        assert ctx.model == "llama3:8b"
        assert ctx.suite == "reasoning-basics"
        assert ctx.test_case.id == "c1"
        assert ctx.metric_config.name == "exact-match"
        assert ctx.judge_client == "not-a-real-client"
        assert ctx.judge_model == "llama3:8b"

    def test_optional_judge_fields_default_to_none(self) -> None:
        ctx = _make_context()

        assert ctx.judge_client is None
        assert ctx.judge_model is None

    def test_extra_fields_forbidden(self) -> None:
        """``extra="forbid"`` catches typos in context construction."""
        with pytest.raises(ValidationError) as excinfo:
            MetricContext(
                model="llama3:8b",
                suite="s",
                test_case=TestCase(
                    id="c1",
                    prompt="Hi",
                    metrics=[MetricConfig(name="exact-match")],
                ),
                metric_config=MetricConfig(name="exact-match"),
                mystery_field=42,  # type: ignore[call-arg]
            )
        assert "mystery_field" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Metric protocol runtime check
# ---------------------------------------------------------------------------


class TestMetricProtocol:
    def test_dummy_metric_satisfies_protocol(self) -> None:
        """A duck-typed metric satisfies the ``Metric`` protocol at runtime."""
        dummy = _DummyMetric("dummy")

        assert isinstance(dummy, Metric)

    def test_non_metric_object_does_not_satisfy_protocol(self) -> None:
        """A plain object without ``name`` / ``score`` does not satisfy ``Metric``."""

        class NotAMetric:
            pass

        assert not isinstance(NotAMetric(), Metric)
