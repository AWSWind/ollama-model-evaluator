"""Feature: ollama-model-evaluator, Property 10: Tokens-per-second arithmetic.

Property 10 (from ``design.md`` §Correctness Properties):

    *For any* execution with ``response_tokens = n`` and ``total_ms =
    t > 0``, ``PerformanceMetrics.tokens_per_second == n / (t /
    1000)``; if ``response_tokens`` is ``None`` or ``t == 0``,
    ``tokens_per_second`` is ``None``.

Validates: Requirement 6.4.

Verified against the module-level helper
:func:`ollama_evaluator.runner.scheduler.compute_tokens_per_second`
so the property holds independently of the scheduler's full dispatch
loop.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ollama_evaluator.runner.scheduler import compute_tokens_per_second


@given(
    n=st.one_of(st.none(), st.integers(min_value=0, max_value=1_000_000)),
    t=st.one_of(st.just(0.0), st.floats(min_value=0.001, max_value=600_000.0)),
)
@settings(max_examples=20)
def test_tokens_per_second_matches_arithmetic_rule(n: int | None, t: float) -> None:
    """*For any* ``(response_tokens, total_ms)``, the helper returns
    ``n / (t / 1000)`` when both are populated and ``t > 0``, else ``None``."""
    result = compute_tokens_per_second(n, t)
    if n is None or t == 0.0:
        assert result is None
    else:
        expected = float(n) / (float(t) / 1000.0)
        assert result == expected


@given(
    n=st.one_of(st.none(), st.integers(min_value=0, max_value=1_000_000)),
    t=st.one_of(st.none(), st.floats(min_value=0.001, max_value=600_000.0)),
)
@settings(max_examples=20)
def test_none_total_ms_returns_none(n: int | None, t: float | None) -> None:
    """When either input is ``None`` (including ``total_ms``), the
    derived value is ``None``."""
    assume(n is None or t is None)
    assert compute_tokens_per_second(n, t) is None
