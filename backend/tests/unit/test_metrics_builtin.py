"""Unit tests for the built-in metrics (Task 5.2).

These tests exercise every built-in metric class defined in
:mod:`ollama_evaluator.metrics.builtin` through the public
``async score(response, ctx) -> MetricResult`` protocol, and verify
:func:`register_builtin_metrics` registers all five classes
idempotently.

Scope: pass/fail correctness per the design document's §Metric framework
pass/score table, parameter-parsing edge cases (required vs optional),
and the runner-facing "missing required param raises ``ValueError``"
contract. Behavioural edge cases that are metric-family-independent
(``MetricResult`` extra-forbid, threshold rendering) are already covered
by ``test_run_report_models.py`` and are not re-tested here.

The ``clean_registry`` fixture in ``test_metrics_base.py`` snapshots and
restores the registry; we rely on the same mechanism here so that a
registration in one test does not leak into the next.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ollama_evaluator.metrics import (
    _REGISTRY,
    MetricContext,
    get_metric,
    list_metrics,
)
from ollama_evaluator.metrics.builtin import (
    Contains,
    ExactMatch,
    HumanevalExecReserved,
    JsonSchemaValid,
    LengthRange,
    RegexMatch,
    ResponseCapture,
    register_builtin_metrics,
)
from ollama_evaluator.models import MetricResult
from ollama_evaluator.suites.models import MetricConfig, TestCase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot-and-restore the metric registry around a single test.

    Used only by :class:`TestRegisterBuiltinMetrics`. Other tests do not
    need this fixture because they construct metric classes directly and
    do not touch the registry.
    """
    snapshot = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    metric_name: str,
    params: dict[str, Any] | None = None,
    expected_output: str | None = None,
) -> MetricContext:
    """Build a valid :class:`MetricContext` for a given metric name.

    The :class:`TestCase` attached to the context carries the metric
    configuration itself so the ``ctx.test_case.metrics`` invariant
    (non-empty list) holds.
    """
    metric_config = MetricConfig(name=metric_name, params=params or {})
    return MetricContext(
        model="llama3:8b",
        suite="suite-a",
        test_case=TestCase(
            id="c1",
            prompt="Prompt",
            expected_output=expected_output,
            metrics=[metric_config],
        ),
        metric_config=metric_config,
    )


# ---------------------------------------------------------------------------
# exact-match
# ---------------------------------------------------------------------------


class TestExactMatch:
    async def test_exact_match_passes_on_identical(self) -> None:
        metric = ExactMatch()
        ctx = _ctx("exact-match", expected_output="hello")

        result = await metric.score("hello", ctx)

        assert result.name == "exact-match"
        assert result.score == 1.0
        assert result.passed is True
        assert result.threshold == 1.0
        assert result.details["matched"] is True

    async def test_exact_match_fails_on_mismatch(self) -> None:
        metric = ExactMatch()
        ctx = _ctx("exact-match", expected_output="hello")

        result = await metric.score("goodbye", ctx)

        assert result.score == 0.0
        assert result.passed is False
        assert result.details["matched"] is False

    async def test_exact_match_trim_default_true(self) -> None:
        """With the default ``trim=True``, surrounding whitespace is ignored."""
        metric = ExactMatch()
        ctx = _ctx("exact-match", expected_output="hello")

        result = await metric.score("  hello\n", ctx)

        assert result.score == 1.0

    async def test_exact_match_trim_false_keeps_whitespace(self) -> None:
        metric = ExactMatch()
        ctx = _ctx(
            "exact-match",
            params={"trim": False},
            expected_output="hello",
        )

        result = await metric.score("  hello\n", ctx)

        assert result.score == 0.0

    async def test_exact_match_case_sensitive_default_true(self) -> None:
        """Default behaviour is case-sensitive."""
        metric = ExactMatch()
        ctx = _ctx("exact-match", expected_output="Hello")

        result = await metric.score("hello", ctx)

        assert result.score == 0.0

    async def test_exact_match_case_insensitive(self) -> None:
        metric = ExactMatch()
        ctx = _ctx(
            "exact-match",
            params={"case_sensitive": False},
            expected_output="HELLO",
        )

        result = await metric.score("hello", ctx)

        assert result.score == 1.0

    async def test_exact_match_missing_expected_output_raises(self) -> None:
        """Missing ``expected_output`` is a Test_Case authoring error (raises)."""
        metric = ExactMatch()
        ctx = _ctx("exact-match", expected_output=None)

        with pytest.raises(ValueError, match="expected_output"):
            await metric.score("hello", ctx)


# ---------------------------------------------------------------------------
# regex-match
# ---------------------------------------------------------------------------


class TestRegexMatch:
    async def test_regex_match_passes_on_search_hit(self) -> None:
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": r"answer:\s*(\d+)"})

        result = await metric.score("The answer: 42 is correct.", ctx)

        assert result.score == 1.0
        assert result.passed is True
        assert result.threshold == 1.0
        assert result.details["match"] == "answer: 42"
        assert result.details["groups"] == ["42"]

    async def test_regex_match_fails_on_no_match(self) -> None:
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": r"\d{4}"})

        result = await metric.score("no digits here", ctx)

        assert result.score == 0.0
        assert result.passed is False
        assert result.details["matched"] is False

    async def test_regex_match_flag_i_case_insensitive(self) -> None:
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": r"HELLO", "flags": "i"})

        result = await metric.score("hello world", ctx)

        assert result.score == 1.0

    async def test_regex_match_flag_m_multiline(self) -> None:
        """``^`` matches the start of each line only with the ``m`` flag."""
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": r"^bar", "flags": "m"})

        result = await metric.score("foo\nbar\nbaz", ctx)

        assert result.score == 1.0

    async def test_regex_match_flag_s_dotall(self) -> None:
        """``.`` matches newline characters only with the ``s`` flag."""
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": r"a.b", "flags": "s"})

        result = await metric.score("a\nb", ctx)

        assert result.score == 1.0

    async def test_regex_match_combined_flags(self) -> None:
        metric = RegexMatch()
        ctx = _ctx(
            "regex-match",
            params={"pattern": r"^HI.THERE", "flags": "ims"},
        )

        result = await metric.score("foo\nhi\nthere", ctx)

        assert result.score == 1.0

    async def test_regex_match_missing_pattern_raises(self) -> None:
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={})

        with pytest.raises(ValueError, match="pattern"):
            await metric.score("anything", ctx)

    async def test_regex_match_invalid_pattern_raises(self) -> None:
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": "[unclosed"})

        with pytest.raises(ValueError, match="invalid"):
            await metric.score("anything", ctx)

    async def test_regex_match_unknown_flag_raises(self) -> None:
        metric = RegexMatch()
        ctx = _ctx("regex-match", params={"pattern": "x", "flags": "z"})

        with pytest.raises(ValueError, match="not supported"):
            await metric.score("x", ctx)


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


class TestContains:
    async def test_contains_all_mode_passes_when_every_substring_present(
        self,
    ) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={"substrings": ["foo", "bar"], "mode": "all"},
        )

        result = await metric.score("foo and bar", ctx)

        assert result.score == 1.0
        assert result.passed is True
        assert result.details["matched_count"] == 2
        assert result.details["mode"] == "all"

    async def test_contains_all_mode_fails_when_any_missing(self) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={"substrings": ["foo", "bar"], "mode": "all"},
        )

        result = await metric.score("foo only", ctx)

        # Fraction semantics: 1 of 2 substrings matched.
        assert result.score == 0.5
        assert result.passed is False
        assert result.details["matched"] == ["foo"]

    async def test_contains_any_mode_passes_on_single_match(self) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={"substrings": ["foo", "bar"], "mode": "any"},
        )

        result = await metric.score("foo only", ctx)

        # One of two matched -> score 0.5; default threshold in any-mode
        # is ~epsilon, so a single match passes.
        assert result.score == 0.5
        assert result.passed is True

    async def test_contains_any_mode_fails_when_none_match(self) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={"substrings": ["foo", "bar"], "mode": "any"},
        )

        result = await metric.score("nothing", ctx)

        assert result.score == 0.0
        assert result.passed is False

    async def test_contains_explicit_threshold(self) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={
                "substrings": ["a", "b", "c", "d"],
                "mode": "all",
                "threshold": 0.5,
            },
        )

        result = await metric.score("a b only", ctx)

        # 2/4 matched = 0.5, threshold is 0.5 -> passes.
        assert result.score == 0.5
        assert result.passed is True

    async def test_contains_empty_list_raises(self) -> None:
        metric = Contains()
        ctx = _ctx("contains", params={"substrings": []})

        with pytest.raises(ValueError, match="non-empty"):
            await metric.score("anything", ctx)

    async def test_contains_missing_substrings_raises(self) -> None:
        metric = Contains()
        ctx = _ctx("contains", params={})

        with pytest.raises(ValueError, match="substrings"):
            await metric.score("anything", ctx)

    async def test_contains_invalid_mode_raises(self) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={"substrings": ["x"], "mode": "either"},
        )

        with pytest.raises(ValueError, match="'any' or 'all'"):
            await metric.score("x", ctx)

    async def test_contains_non_string_element_raises(self) -> None:
        metric = Contains()
        ctx = _ctx(
            "contains",
            params={"substrings": ["a", 42]},
        )

        with pytest.raises(ValueError, match="substrings"):
            await metric.score("a", ctx)


# ---------------------------------------------------------------------------
# json-schema-valid
# ---------------------------------------------------------------------------


class TestJsonSchemaValid:
    async def test_valid_json_and_schema_passes(self) -> None:
        metric = JsonSchemaValid()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        ctx = _ctx("json-schema-valid", params={"schema": schema})

        result = await metric.score('{"name": "Ada", "age": 36}', ctx)

        assert result.score == 1.0
        assert result.passed is True
        assert result.details["parsed"] == {"name": "Ada", "age": 36}

    async def test_invalid_json_fails_with_parse_error(self) -> None:
        metric = JsonSchemaValid()
        ctx = _ctx(
            "json-schema-valid",
            params={"schema": {"type": "object"}},
        )

        result = await metric.score("not json at all", ctx)

        assert result.score == 0.0
        assert result.passed is False
        assert "parse_error" in result.details

    async def test_valid_json_but_schema_mismatch_fails(self) -> None:
        metric = JsonSchemaValid()
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }
        ctx = _ctx("json-schema-valid", params={"schema": schema})

        result = await metric.score('{"name": "Ada"}', ctx)

        assert result.score == 0.0
        assert result.passed is False
        assert "validation_error" in result.details
        assert result.details["parsed"] == {"name": "Ada"}

    async def test_response_with_leading_whitespace_still_parses(self) -> None:
        """``json.loads(response.strip())`` handles leading/trailing whitespace."""
        metric = JsonSchemaValid()
        ctx = _ctx(
            "json-schema-valid",
            params={"schema": {"type": "object"}},
        )

        result = await metric.score("\n  {}\n  ", ctx)

        assert result.score == 1.0

    async def test_missing_schema_raises(self) -> None:
        metric = JsonSchemaValid()
        ctx = _ctx("json-schema-valid", params={})

        with pytest.raises(ValueError, match="schema"):
            await metric.score("{}", ctx)

    async def test_invalid_schema_raises(self) -> None:
        """An invalid schema is a metric-config bug, not a response failure."""
        metric = JsonSchemaValid()
        # ``type`` must be a string or list of strings; ``42`` is neither.
        ctx = _ctx(
            "json-schema-valid",
            params={"schema": {"type": 42}},
        )

        with pytest.raises(ValueError, match="invalid schema"):
            await metric.score("{}", ctx)


# ---------------------------------------------------------------------------
# length-range
# ---------------------------------------------------------------------------


class TestLengthRange:
    async def test_chars_within_range_passes(self) -> None:
        metric = LengthRange()
        ctx = _ctx("length-range", params={"min": 3, "max": 10})

        result = await metric.score("hello", ctx)

        assert result.score == 1.0
        assert result.passed is True
        assert result.details["length"] == 5
        assert result.details["unit"] == "chars"

    async def test_chars_below_min_fails(self) -> None:
        metric = LengthRange()
        ctx = _ctx("length-range", params={"min": 10})

        result = await metric.score("short", ctx)

        assert result.score == 0.0
        assert result.passed is False
        assert result.details["in_range"] is False

    async def test_chars_above_max_fails(self) -> None:
        metric = LengthRange()
        ctx = _ctx("length-range", params={"max": 3})

        result = await metric.score("too long", ctx)

        assert result.score == 0.0
        assert result.passed is False

    async def test_min_only_open_upper_bound(self) -> None:
        """``max`` absent means no upper bound."""
        metric = LengthRange()
        ctx = _ctx("length-range", params={"min": 2})

        result = await metric.score("x" * 10_000, ctx)

        assert result.score == 1.0

    async def test_max_only_open_lower_bound(self) -> None:
        """``min`` absent means no lower bound."""
        metric = LengthRange()
        ctx = _ctx("length-range", params={"max": 5})

        result = await metric.score("", ctx)

        assert result.score == 1.0
        assert result.details["length"] == 0

    async def test_tokens_unit_uses_whitespace_split(self) -> None:
        metric = LengthRange()
        ctx = _ctx(
            "length-range",
            params={"min": 3, "max": 3, "unit": "tokens"},
        )

        result = await metric.score("one two three", ctx)

        assert result.score == 1.0
        assert result.details["unit"] == "tokens"
        assert result.details["length"] == 3

    async def test_tokens_unit_fails_when_count_off(self) -> None:
        metric = LengthRange()
        ctx = _ctx(
            "length-range",
            params={"min": 5, "unit": "tokens"},
        )

        result = await metric.score("one two three", ctx)

        assert result.score == 0.0

    async def test_both_bounds_none_raises(self) -> None:
        metric = LengthRange()
        ctx = _ctx("length-range", params={})

        with pytest.raises(ValueError, match="min.*max"):
            await metric.score("x", ctx)

    async def test_negative_min_raises(self) -> None:
        metric = LengthRange()
        ctx = _ctx("length-range", params={"min": -1})

        with pytest.raises(ValueError, match=">= 0"):
            await metric.score("x", ctx)

    async def test_min_greater_than_max_raises(self) -> None:
        metric = LengthRange()
        ctx = _ctx("length-range", params={"min": 10, "max": 5})

        with pytest.raises(ValueError, match="<="):
            await metric.score("x", ctx)

    async def test_invalid_unit_raises(self) -> None:
        metric = LengthRange()
        ctx = _ctx(
            "length-range",
            params={"min": 1, "unit": "words"},
        )

        with pytest.raises(ValueError, match="unit"):
            await metric.score("x", ctx)


# ---------------------------------------------------------------------------
# response-capture (Task 5.4, Requirement 17.9)
# ---------------------------------------------------------------------------


class TestResponseCapture:
    async def test_always_passes_with_zero_score(self) -> None:
        """``response-capture`` always returns ``score=0.0`` and ``passed=True``.

        The HumanEval v1 adapter uses this metric to defer scoring to an
        external grader (Requirement 17.9); the fixed ``score=0.0``
        signals "no score reported" and ``passed=True`` keeps the Run
        from classifying the Test_Case as a failure.
        """
        metric = ResponseCapture()
        ctx = _ctx("response-capture")

        result = await metric.score("any response at all", ctx)

        assert result.name == "response-capture"
        assert result.score == 0.0
        assert result.passed is True

    async def test_captures_response_verbatim_in_details(self) -> None:
        """The raw response is stored verbatim in ``details.response``."""
        metric = ResponseCapture()
        ctx = _ctx("response-capture")
        raw = "def add(a, b):\n    return a + b\n"

        result = await metric.score(raw, ctx)

        assert result.details == {"response": raw}

    async def test_empty_response_also_passes(self) -> None:
        """Even an empty response passes — scoring is fully deferred."""
        metric = ResponseCapture()
        ctx = _ctx("response-capture")

        result = await metric.score("", ctx)

        assert result.passed is True
        assert result.score == 0.0
        assert result.details == {"response": ""}


# ---------------------------------------------------------------------------
# humaneval-exec (reserved name; Task 5.4, Requirement 17.9)
# ---------------------------------------------------------------------------


class TestHumanevalExecReserved:
    async def test_score_raises_not_implemented(self) -> None:
        """The reserved metric raises :class:`NotImplementedError` with a clear message.

        Registering a raising stub (rather than leaving the name
        unregistered) lets the suite loader accept Test_Cases that
        declare this metric today while the runner's per-metric
        ``try/except`` wrapper converts the raise into a
        :class:`MetricResult` with ``error`` populated (Requirement 7.5).
        """
        metric = HumanevalExecReserved()
        ctx = _ctx("humaneval-exec")

        with pytest.raises(NotImplementedError, match="reserved"):
            await metric.score("anything", ctx)

    def test_has_expected_name(self) -> None:
        """``name`` is ``humaneval-exec`` per the registry key reservation."""
        assert HumanevalExecReserved().name == "humaneval-exec"


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


class TestRegisterBuiltinMetrics:
    def test_registers_all_seven(self, clean_registry: None) -> None:
        """All seven built-in metrics resolve via :func:`get_metric` by name.

        Includes the five string/structural metrics (Task 5.2), the
        HumanEval v1 default ``response-capture``, and the reserved
        ``humaneval-exec`` (Task 5.4, Requirement 17.9).
        """
        _REGISTRY.clear()

        register_builtin_metrics()

        expected_names = {
            "exact-match",
            "regex-match",
            "contains",
            "json-schema-valid",
            "length-range",
            "response-capture",
            "humaneval-exec",
        }
        assert expected_names.issubset(set(list_metrics()))

        # Each name resolves to an instance of the expected class.
        assert isinstance(get_metric("exact-match"), ExactMatch)
        assert isinstance(get_metric("regex-match"), RegexMatch)
        assert isinstance(get_metric("contains"), Contains)
        assert isinstance(get_metric("json-schema-valid"), JsonSchemaValid)
        assert isinstance(get_metric("length-range"), LengthRange)
        assert isinstance(get_metric("response-capture"), ResponseCapture)
        assert isinstance(get_metric("humaneval-exec"), HumanevalExecReserved)

    def test_register_is_idempotent(self, clean_registry: None) -> None:
        """Calling twice replaces singletons without raising or duplicating.

        The registry has replacement semantics (documented in
        :mod:`ollama_evaluator.metrics`), so a second call simply
        installs fresh singletons. ``list_metrics()`` must still report
        exactly the seven built-in names — no duplicates.
        """
        _REGISTRY.clear()

        register_builtin_metrics()
        names_first = set(list_metrics())
        register_builtin_metrics()
        names_second = set(list_metrics())

        assert names_first == names_second
        assert len(names_second) == 7

    async def test_registered_metric_is_usable(self, clean_registry: None) -> None:
        """The object fetched from the registry can be invoked via ``score``."""
        _REGISTRY.clear()
        register_builtin_metrics()

        metric = get_metric("exact-match")
        ctx = _ctx("exact-match", expected_output="ping")

        result = await metric.score("ping", ctx)

        assert isinstance(result, MetricResult)
        assert result.score == 1.0
