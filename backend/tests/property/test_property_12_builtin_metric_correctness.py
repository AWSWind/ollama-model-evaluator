"""Property 12: Built-in metric correctness.

For every built-in scoring metric ``m`` in the set
``{exact-match, regex-match, contains, json-schema-valid, length-range}``,
``m.score(response, ctx)`` returns a :class:`MetricResult` whose
``score`` is a :class:`float`, whose ``passed`` is a :class:`bool`, and
whose pair ``(score, passed)`` obeys the rules in the design document's
§Metric framework pass/score table:

================== ============================================= ===========================
Metric             Score rule                                    Pass condition
================== ============================================= ===========================
``exact-match``    ``1.0`` on match, ``0.0`` on mismatch         ``score == 1.0``
``regex-match``    ``1.0`` on match, ``0.0`` on mismatch         ``score == 1.0``
``contains``       fraction of matched substrings                ``score >= threshold``
``json-schema-valid`` ``1.0`` iff parse + validate succeed       ``score == 1.0``
``length-range``   ``1.0`` iff in range, else ``0.0``            ``score == 1.0``
================== ============================================= ===========================

The property is stated in ``.kiro/specs/ollama-model-evaluator/design.md``
§Correctness Properties as Property 12 and is driven by Requirements
7.1 (every metric returns a :class:`MetricResult`), 7.3 (numeric
``score``), and 7.4 (boolean ``passed`` that matches the metric's
pass rule).

Approach
--------
Each metric has its own Hypothesis test that draws varied
``(response, params)`` pairs from a metric-specific strategy, calls the
metric via :func:`asyncio.run`, and asserts the invariants above. The
strategies intentionally cover both "should pass" and "should fail"
shapes so the pass/score rules are exercised in both directions within
a single run of 100+ examples.

``asyncio.run`` is used rather than a pytest-asyncio coroutine test
because Hypothesis drives synchronous test bodies; the ``score``
methods are ``async`` so each property test wraps the call in
``asyncio.run`` once per example (see also
:mod:`tests.property.test_property_18_run_report_roundtrip` for the
same pattern applied to report round-trips).

``max_examples=20`` and ``deadline=None`` match the testing-strategy
floor set in ``design.md``.
"""

from __future__ import annotations

import asyncio
import json
from string import ascii_letters, digits
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.metrics.builtin import (
    Contains,
    ExactMatch,
    JsonSchemaValid,
    LengthRange,
    RegexMatch,
)
from ollama_evaluator.models import MetricResult
from ollama_evaluator.suites.models import MetricConfig, TestCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ASCII-only alphabet for response/pattern generation. Unicode and control
# characters would add noise (JSON escaping, unintended regex tokens) that is
# orthogonal to the invariants under test — the pass/score rules only care
# about equality, presence, and length.
_SIMPLE_ALPHABET = ascii_letters + digits + " -_."


def _build_ctx(metric_name: str, params: dict[str, Any], expected: str | None) -> MetricContext:
    """Build a :class:`MetricContext` for invoking a built-in metric.

    The :class:`TestCase` attached to the context carries ``expected``
    (for ``exact-match``) and the driving :class:`MetricConfig` so the
    ``ctx.test_case.metrics`` invariant (non-empty list) is preserved.
    """
    metric_config = MetricConfig(name=metric_name, params=params)
    return MetricContext(
        model="m",
        suite="s",
        test_case=TestCase(
            id="c1",
            prompt="prompt",
            expected_output=expected,
            metrics=[metric_config],
        ),
        metric_config=metric_config,
    )


def _assert_is_float(value: Any) -> None:
    """Assert ``value`` is a real :class:`float`, not just numeric.

    Pydantic will coerce ``int`` → ``float`` on ``MetricResult.score``, so
    after construction every score *is* a float regardless of the metric's
    internal representation. This assertion belts-and-braces the contract.
    """
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# exact-match (Property 12 — score ∈ {0.0, 1.0}, passed == (score == 1.0))
# ---------------------------------------------------------------------------


@st.composite
def _exact_match_inputs(
    draw: st.DrawFn,
) -> tuple[str, str, dict[str, Any]]:
    """Draw ``(response, expected_output, params)`` for ``exact-match``.

    Half the time the response is deliberately equal to the expected
    output so the "pass" branch is exercised; the other half draws
    independent strings so mismatch dominates. ``trim`` and
    ``case_sensitive`` are each drawn independently to cover all four
    flag combinations.
    """
    text = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=0, max_size=20))
    match_case = draw(st.booleans())
    if match_case:
        response = text
        expected = text
    else:
        expected = text
        response = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=0, max_size=20))
    params: dict[str, Any] = {
        "case_sensitive": draw(st.booleans()),
        "trim": draw(st.booleans()),
    }
    return response, expected, params


@given(data=_exact_match_inputs())
@settings(max_examples=20, deadline=None)
def test_exact_match_correctness(data: tuple[str, str, dict[str, Any]]) -> None:
    """**Validates: Requirements 7.1, 7.3, 7.4**

    ``exact-match``: ``score ∈ {0.0, 1.0}``, ``passed == (score == 1.0)``.
    """
    response, expected, params = data
    metric = ExactMatch()
    ctx = _build_ctx("exact-match", params, expected)

    result: MetricResult = asyncio.run(metric.score(response, ctx))

    _assert_is_float(result.score)
    assert isinstance(result.passed, bool)
    assert result.score in (0.0, 1.0)
    assert result.passed is (result.score == 1.0)


# ---------------------------------------------------------------------------
# regex-match (score ∈ {0.0, 1.0}, passed == (score == 1.0))
# ---------------------------------------------------------------------------


@st.composite
def _regex_match_inputs(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    """Draw ``(response, params)`` for ``regex-match``.

    Pattern is a literal substring drawn from the response half the
    time (so the "match" branch fires), and an ASCII literal chosen
    independently the other half (so misses also appear). The pattern
    is always a literal string — special regex metacharacters are
    excluded from the alphabet so ``re.compile`` never raises. Testing
    invalid-pattern error paths is the domain of the unit tests, not
    this property.
    """
    response = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=0, max_size=30))
    if response and draw(st.booleans()):
        # Draw a non-empty substring of ``response`` so regex matches.
        start = draw(st.integers(min_value=0, max_value=len(response) - 1))
        end = draw(st.integers(min_value=start + 1, max_value=len(response)))
        pattern = response[start:end]
    else:
        pattern = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=1, max_size=10))

    flags = draw(st.sampled_from(["", "i", "m", "s", "im", "is", "ms", "ims"]))
    return response, {"pattern": pattern, "flags": flags}


@given(data=_regex_match_inputs())
@settings(max_examples=20, deadline=None)
def test_regex_match_correctness(data: tuple[str, dict[str, Any]]) -> None:
    """**Validates: Requirements 7.1, 7.3, 7.4**

    ``regex-match``: ``score ∈ {0.0, 1.0}``, ``passed == (score == 1.0)``.
    """
    response, params = data
    metric = RegexMatch()
    ctx = _build_ctx("regex-match", params, expected=None)

    result: MetricResult = asyncio.run(metric.score(response, ctx))

    _assert_is_float(result.score)
    assert isinstance(result.passed, bool)
    assert result.score in (0.0, 1.0)
    assert result.passed is (result.score == 1.0)


# ---------------------------------------------------------------------------
# contains (score == matched/total, passed == score >= threshold)
# ---------------------------------------------------------------------------


@st.composite
def _contains_inputs(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    """Draw ``(response, params)`` for ``contains``.

    The substrings list is always non-empty (the metric rejects empty
    lists at score time). Each substring is drawn from either a
    substring of the response (so some matches occur) or an independent
    literal (so some misses occur). ``mode`` and ``threshold`` are
    drawn to exercise every pass-condition branch.
    """
    response = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=1, max_size=30))
    n = draw(st.integers(min_value=1, max_value=4))
    substrings: list[str] = []
    for _ in range(n):
        if draw(st.booleans()):
            start = draw(st.integers(min_value=0, max_value=len(response) - 1))
            end = draw(st.integers(min_value=start + 1, max_value=len(response)))
            substrings.append(response[start:end])
        else:
            substrings.append(
                draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=1, max_size=8))
            )
    mode = draw(st.sampled_from(["any", "all"]))
    # Threshold covers both halves of the score range so pass/fail both fire.
    threshold = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    params = {"substrings": substrings, "mode": mode, "threshold": threshold}
    return response, params


@given(data=_contains_inputs())
@settings(max_examples=20, deadline=None)
def test_contains_correctness(data: tuple[str, dict[str, Any]]) -> None:
    """**Validates: Requirements 7.1, 7.3, 7.4**

    ``contains``: ``score`` is the fraction of substrings matched and is
    therefore in ``[0.0, 1.0]``; ``passed == (score >= threshold)``.
    Also recomputes the expected fraction from the inputs and asserts
    the metric agrees — that guards the fraction arithmetic in addition
    to the type/pass invariants.
    """
    response, params = data
    metric = Contains()
    ctx = _build_ctx("contains", params, expected=None)

    result: MetricResult = asyncio.run(metric.score(response, ctx))

    _assert_is_float(result.score)
    assert isinstance(result.passed, bool)
    assert 0.0 <= result.score <= 1.0

    substrings: list[str] = params["substrings"]
    expected_matches = sum(1 for s in substrings if s in response)
    expected_fraction = expected_matches / len(substrings)
    assert result.score == expected_fraction
    assert result.passed is (result.score >= params["threshold"])


# ---------------------------------------------------------------------------
# json-schema-valid (score ∈ {0.0, 1.0}, passed == (score == 1.0))
# ---------------------------------------------------------------------------


# ``json`` values that the schema below will either accept or reject.
# Generating both well-formed and malformed payloads in roughly equal
# numbers keeps the "pass" and "fail" branches exercised without filtering.
_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000, max_value=1_000),
    st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.text(alphabet=_SIMPLE_ALPHABET, min_size=0, max_size=10),
)


@st.composite
def _json_schema_inputs(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    """Draw ``(response, params)`` for ``json-schema-valid``.

    Responses are one of:

    1. Valid JSON matching a permissive ``{"type": "object"}`` schema.
    2. Valid JSON that does *not* match a stricter schema.
    3. Plain non-JSON text (parse error branch).

    Each branch fires with non-trivial probability across 100 examples,
    so both ``score == 1.0`` and ``score == 0.0`` are observed.
    """
    branch = draw(st.sampled_from(["valid-object", "mismatch", "not-json"]))
    if branch == "valid-object":
        # A permissive schema + a plain dict — always validates.
        payload = draw(
            st.dictionaries(
                keys=st.text(alphabet=_SIMPLE_ALPHABET, min_size=1, max_size=5),
                values=_json_scalar,
                max_size=3,
            )
        )
        return json.dumps(payload), {"schema": {"type": "object"}}
    if branch == "mismatch":
        # Strict schema requiring ``{"age": int}``; payload omits ``age``.
        payload = draw(
            st.dictionaries(
                keys=st.text(alphabet=_SIMPLE_ALPHABET, min_size=1, max_size=5),
                values=_json_scalar,
                max_size=3,
            ).filter(lambda d: "age" not in d)
        )
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }
        return json.dumps(payload), {"schema": schema}
    # "not-json": deliberately non-JSON text.
    text = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=1, max_size=15))
    # Prefix with a letter so the response never accidentally parses as a
    # bare JSON number/string/keyword.
    text = "x" + text
    return text, {"schema": {"type": "object"}}


@given(data=_json_schema_inputs())
@settings(max_examples=20, deadline=None)
def test_json_schema_valid_correctness(data: tuple[str, dict[str, Any]]) -> None:
    """**Validates: Requirements 7.1, 7.3, 7.4**

    ``json-schema-valid``: ``score ∈ {0.0, 1.0}``, ``passed == (score == 1.0)``.
    """
    response, params = data
    metric = JsonSchemaValid()
    ctx = _build_ctx("json-schema-valid", params, expected=None)

    result: MetricResult = asyncio.run(metric.score(response, ctx))

    _assert_is_float(result.score)
    assert isinstance(result.passed, bool)
    assert result.score in (0.0, 1.0)
    assert result.passed is (result.score == 1.0)


# ---------------------------------------------------------------------------
# length-range (score ∈ {0.0, 1.0}, passed == (score == 1.0))
# ---------------------------------------------------------------------------


@st.composite
def _length_range_inputs(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    """Draw ``(response, params)`` for ``length-range``.

    Bounds are constrained to satisfy the metric's validators
    (``min >= 0``, ``max >= 0``, ``min <= max`` when both set, at least
    one bound present). The response length is drawn independently so
    both in-range and out-of-range responses show up.
    """
    response = draw(st.text(alphabet=_SIMPLE_ALPHABET, min_size=0, max_size=40))
    unit = draw(st.sampled_from(["chars", "tokens"]))
    # Pick which bound(s) are set; always include at least one.
    has_min = draw(st.booleans())
    has_max = draw(st.booleans()) or not has_min
    params: dict[str, Any] = {"unit": unit}
    min_bound = draw(st.integers(min_value=0, max_value=30)) if has_min else None
    max_bound = (
        draw(st.integers(min_value=min_bound or 0, max_value=40)) if has_max else None
    )
    if min_bound is not None:
        params["min"] = min_bound
    if max_bound is not None:
        params["max"] = max_bound
    return response, params


@given(data=_length_range_inputs())
@settings(max_examples=20, deadline=None)
def test_length_range_correctness(data: tuple[str, dict[str, Any]]) -> None:
    """**Validates: Requirements 7.1, 7.3, 7.4**

    ``length-range``: ``score ∈ {0.0, 1.0}``, ``passed == (score == 1.0)``.
    """
    response, params = data
    metric = LengthRange()
    ctx = _build_ctx("length-range", params, expected=None)

    result: MetricResult = asyncio.run(metric.score(response, ctx))

    _assert_is_float(result.score)
    assert isinstance(result.passed, bool)
    assert result.score in (0.0, 1.0)
    assert result.passed is (result.score == 1.0)
