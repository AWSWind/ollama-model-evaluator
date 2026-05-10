"""Property 14: Metric error isolation.

For every Test_Case with ``k`` configured metrics where a subset
``E ⊆ {0, ..., k-1}`` of them raise during scoring,
:func:`score_all_metrics` returns a list of ``k`` :class:`MetricResult`
objects such that:

1. **Order preserved.** ``result[i].name == metric_configs[i].name``
   for every ``i in 0..k-1``.
2. **Length preserved.** ``len(result) == k``.
3. **Errored metrics carry an error.** For every ``i in E``,
   ``result[i].error is not None``, ``result[i].passed is False``,
   and ``result[i].score == 0.0`` (the documented convention for
   error results).
4. **Non-errored metrics scored normally.** For every ``i not in E``,
   ``result[i].error is None`` and the result matches what the
   corresponding metric would have returned on its own.

The property is stated in ``.kiro/specs/ollama-model-evaluator/design.md``
§Correctness Properties as Property 14 and validates Requirement 7.5
(metric errors do not crash the Test_Case).

Approach
--------
The registry is snapshotted before each Hypothesis example and
restored afterwards. Each example draws a list of ``k`` metric names
(guaranteed unique so the registry state is well-defined) plus a
subset ``E`` of indices that should raise. For each index we install a
fake metric in the registry — either a ``_PassingFake`` that returns
a deterministic :class:`MetricResult` tagged with the index, or a
``_RaisingFake`` that raises with a message tagged with the index.
Running :func:`score_all_metrics` then lets us assert the four
invariants above by comparing the returned list against the drawn
``E`` set index-by-index.

Using real registrations (rather than monkey-patching
:func:`get_metric`) exercises the full code path including the
:func:`get_metric` lookup, so a regression where the scheduler
bypasses the registry would also falsify the property.

``max_examples=20`` and ``deadline=None`` match the testing-strategy
floor set in ``design.md``.
"""

from __future__ import annotations

import asyncio
from string import ascii_lowercase, digits

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.metrics import _REGISTRY, register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult
from ollama_evaluator.runner.scoring import score_all_metrics
from ollama_evaluator.suites.models import MetricConfig, TestCase

# Use names that could never collide with a real built-in metric so
# replacing them in the registry (and forgetting to restore in a rare
# failure path) cannot hide a bug in real metrics.
_FAKE_METRIC_NAME_ALPHABET = ascii_lowercase + digits + "-"


class _PassingFake:
    """Metric stand-in that returns a deterministic :class:`MetricResult`.

    Tagged with ``self.name`` so the property can recover which
    metric produced the result and assert order-preservation.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        del response
        return MetricResult(
            name=ctx.metric_config.name,
            score=1.0,
            passed=True,
            details={"tag": self.name},
        )


class _RaisingFake:
    """Metric stand-in that raises :class:`RuntimeError` when scored.

    The error message embeds ``self.name`` so the assertion can verify
    the error text carries the offending metric identifier — useful
    because a real metric error message usually names the metric in
    some way and the runner forwards it verbatim.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        del response, ctx
        raise RuntimeError(f"boom from {self.name}")


def _run_scoring(
    metric_configs: list[MetricConfig], test_case: TestCase
) -> list[MetricResult]:
    """Invoke :func:`score_all_metrics` synchronously via :func:`asyncio.run`."""
    return asyncio.run(
        score_all_metrics(
            "response-under-test",
            test_case,
            metric_configs,
        )
    )


@st.composite
def _metric_plan(
    draw: st.DrawFn,
) -> tuple[list[str], set[int]]:
    """Draw ``(metric_names, error_indices)`` for a Test_Case scoring plan.

    ``metric_names`` has between 1 and 5 unique entries (property-14
    scales well beyond that but larger ``k`` only adds duplicate
    coverage at O(k) cost per example). ``error_indices`` is drawn
    independently, so empty / full / proper-subset cases all appear
    across 100 examples.

    Name uniqueness is required because the registry is keyed by name;
    duplicates would collapse two configs onto the same registered
    metric and blur the "subset ``E``" semantics.
    """
    k = draw(st.integers(min_value=1, max_value=5))
    names: list[str] = draw(
        st.lists(
            st.text(
                alphabet=_FAKE_METRIC_NAME_ALPHABET, min_size=4, max_size=10
            ).map(lambda s: "pbt-" + s),
            min_size=k,
            max_size=k,
            unique=True,
        )
    )
    error_indices: set[int] = set(
        draw(st.lists(st.integers(min_value=0, max_value=k - 1), unique=True))
    )
    return names, error_indices


@given(plan=_metric_plan())
@settings(max_examples=20, deadline=None)
def test_metric_error_isolation(plan: tuple[list[str], set[int]]) -> None:
    """**Validates: Requirement 7.5**

    Metrics in the error subset ``E`` produce :class:`MetricResult`
    with ``error != None`` and ``passed=False``; metrics outside ``E``
    produce their normal result. The returned list preserves the order
    and length of ``metric_configs``.
    """
    names, error_indices = plan

    # Snapshot the registry so this example cannot leak registrations
    # into other tests. ``_REGISTRY.clear()`` is intentionally not used
    # here — we keep the real metrics in place and only add / replace
    # the fakes named ``"pbt-..."`` which are guaranteed not to collide.
    snapshot = dict(_REGISTRY)
    try:
        # Install fakes for every drawn name. Registry has replacement
        # semantics so a previously-registered fake with the same name
        # would simply be overwritten here; uniqueness of ``names``
        # guarantees this is a no-op.
        for idx, name in enumerate(names):
            if idx in error_indices:
                register_metric(_RaisingFake(name))
            else:
                register_metric(_PassingFake(name))

        metric_configs = [MetricConfig(name=name) for name in names]
        test_case = TestCase(
            id="c1",
            prompt="prompt",
            metrics=metric_configs,
        )

        results = _run_scoring(metric_configs, test_case)
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)

    # 1. Length preserved.
    assert len(results) == len(names)

    for idx, (name, result) in enumerate(zip(names, results, strict=True)):
        # 2. Order preserved — result at position ``idx`` is for the
        #    metric at position ``idx``.
        assert result.name == name

        if idx in error_indices:
            # 3. Errored metrics carry an error and the documented
            #    failure shape (``score=0.0``, ``passed=False``).
            assert result.error is not None
            assert result.passed is False
            assert result.score == 0.0
            # The metric name shows up in the error message because
            # our fake embeds it; that's how a real metric typically
            # surfaces its identity too.
            assert name in result.error
        else:
            # 4. Non-errored metrics are scored normally.
            assert result.error is None
            assert result.passed is True
            assert result.score == 1.0
            # The fake tags the result with its own name via ``details.tag``.
            assert result.details.get("tag") == name


def test_empty_metric_list_returns_empty_list() -> None:
    """Edge case: zero metrics means zero results — no errors raised."""
    test_case = TestCase(
        id="c1",
        prompt="prompt",
        # ``TestCase.metrics`` requires a non-empty list, so we build
        # one but pass an empty list to the scorer instead.
        metrics=[MetricConfig(name="exact-match")],
    )

    results: list[MetricResult] = asyncio.run(
        score_all_metrics("response", test_case, [])
    )
    assert results == []
