"""Property 15: Repetition aggregates.

For every list of per-repetition scores ``[s_1, ..., s_R]``,
:func:`aggregate_metric_scores` returns ``(mean, stddev, count)`` with:

* ``mean == statistics.fmean([s_1, ..., s_R])`` (when ``R >= 1``)
* ``stddev == statistics.pstdev([s_1, ..., s_R])`` (when ``R >= 1``)
* ``count == R``

Edge cases documented on the function:

* Empty list → ``(0.0, 0.0, 0)``.
* Single element → ``stddev == 0.0`` (pstdev of a singleton is well-defined).

The property is stated in ``.kiro/specs/ollama-model-evaluator/design.md``
§Correctness Properties as Property 15 and validates Requirement 7.6
(per-repetition mean/stddev aggregation).

Approach
--------
Hypothesis draws lists of floats of varying length (including zero and
one) and compares the aggregator's output against the
:mod:`statistics` module directly. Using the standard library as the
oracle is deliberate: ``fmean`` and ``pstdev`` are the exact functions
the aggregator delegates to, so an independent re-implementation
wouldn't add signal. What we *are* asserting is that the aggregator
wires the pieces together correctly and does not drop / reorder
samples.

``max_examples=20`` and ``deadline=None`` match the testing-strategy
floor set in ``design.md``.
"""

from __future__ import annotations

import math
import statistics

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.runner.aggregate import aggregate_metric_scores

# Bounded finite floats keep the property focused on the aggregator's
# plumbing. NaN would falsify ``==`` on both sides of every assertion
# spuriously (NaN != NaN); infinities would push the sums out of IEEE-754
# double precision. Neither is a realistic metric score.
_score = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


# ---------------------------------------------------------------------------
# Edge cases (documented on the function)
# ---------------------------------------------------------------------------


def test_empty_list_returns_zero_triple() -> None:
    """**Validates: Requirement 7.6**

    Empty input returns the documented ``(0.0, 0.0, 0)`` triple so
    the aggregator can emit a :class:`MetricAggregate` for metrics
    that errored on every execution without raising.
    """
    assert aggregate_metric_scores([]) == (0.0, 0.0, 0)


@given(score=_score)
@settings(max_examples=20, deadline=None)
def test_single_element_stddev_is_zero(score: float) -> None:
    """**Validates: Requirement 7.6**

    With one sample, ``pstdev`` returns ``0.0`` — the aggregator
    delegates verbatim, so this is a single-sample invariant.
    """
    mean, stddev, count = aggregate_metric_scores([score])

    assert count == 1
    assert math.isclose(mean, score)
    assert stddev == 0.0


# ---------------------------------------------------------------------------
# General case — agrees with :mod:`statistics`
# ---------------------------------------------------------------------------


@given(scores=st.lists(_score, min_size=1, max_size=20))
@settings(max_examples=20, deadline=None)
def test_matches_statistics_module(scores: list[float]) -> None:
    """**Validates: Requirement 7.6**

    For every non-empty input, the aggregator's ``mean``/``stddev``
    equal :func:`statistics.fmean` and :func:`statistics.pstdev`
    respectively, and ``count == len(scores)``.

    ``math.isclose`` is used rather than strict ``==`` because both
    sides of the comparison go through float arithmetic and the
    aggregator's implementation could rearrange additions in a future
    refactor without changing the mathematical result. The default
    :math:`rel\\_tol=1e-9` is tighter than any realistic user-facing
    difference and still tolerates tiny reassociation-driven epsilons.
    """
    mean, stddev, count = aggregate_metric_scores(scores)

    assert count == len(scores)
    assert math.isclose(mean, statistics.fmean(scores))
    assert math.isclose(stddev, statistics.pstdev(scores))
