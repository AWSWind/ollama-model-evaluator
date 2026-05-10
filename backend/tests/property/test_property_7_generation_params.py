"""Property 7: Generation parameter resolution.

For every :class:`TestCase` ``tc`` and :class:`GenerationDefaults`
``d``, the resolved generation parameter for each field equals
``tc.<field> if tc.<field> is not None else d.<field>``::

    temperature     := tc.temperature     if not None else d.temperature
    max_tokens      := tc.max_tokens      if not None else d.max_tokens
    stop_sequences  := tc.stop_sequences  if not None else d.stop_sequences

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness
Properties as Property 7 and validates Requirements 5.3 and 5.4.

``stop_sequences`` carries a three-way distinction on
:class:`TestCase` — ``None`` means "inherit", ``[]`` means "explicitly
clear defaults", and a non-empty list means "override". The resolver
preserves this distinction: ``[]`` on the test case must *not* fall
through to ``d.stop_sequences`` (it is an explicit override to "no
stop sequences").

Approach
--------
Hypothesis draws a :class:`TestCase` and a :class:`GenerationDefaults`
independently and checks the three field-level equalities against
the :class:`GenerateOptions` produced by
:func:`resolve_generate_options`. Because
:class:`GenerateOptions` uses Ollama-native names (``num_predict``
for ``max_tokens``, ``stop`` for ``stop_sequences``), the assertions
translate at the boundary.

``max_examples=20`` and ``deadline=None`` match the floor set in
``design.md``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.runner.selection import resolve_generate_options
from ollama_evaluator.suites.models import GenerationDefaults, MetricConfig, TestCase

_TAG_POOL = ["math", "code", "reading"]

# Stop-sequences strings are ASCII-only and short: the property is
# about None-vs-value resolution, not about string content.
_STOP_TOKEN = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6
)


@st.composite
def _test_case(draw: st.DrawFn) -> TestCase:
    """Draw a :class:`TestCase` whose generation fields cover every arm.

    Each of ``temperature`` / ``max_tokens`` / ``stop_sequences`` is
    drawn from a :func:`st.one_of` that includes the "inherit" arm
    (``None``) plus a concrete-value arm so both branches are
    covered in ~50% of draws.
    """
    stop_sequences_strategy = st.one_of(
        st.none(),
        st.just([]),
        st.lists(_STOP_TOKEN, min_size=1, max_size=3),
    )
    temperature_strategy = st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    max_tokens_strategy = st.one_of(
        st.none(), st.integers(min_value=1, max_value=10_000)
    )
    return TestCase(
        id="tc-1",
        prompt="prompt",
        tags=draw(
            st.lists(st.sampled_from(_TAG_POOL), min_size=0, max_size=2, unique=True)
        ),
        temperature=draw(temperature_strategy),
        max_tokens=draw(max_tokens_strategy),
        stop_sequences=draw(stop_sequences_strategy),
        metrics=[MetricConfig(name="exact-match")],
    )


@st.composite
def _generation_defaults(draw: st.DrawFn) -> GenerationDefaults:
    """Draw a :class:`GenerationDefaults` with varied non-default values."""
    temperature_strategy = st.floats(
        min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
    )
    max_tokens_strategy = st.one_of(
        st.none(), st.integers(min_value=1, max_value=10_000)
    )
    return GenerationDefaults(
        temperature=draw(temperature_strategy),
        max_tokens=draw(max_tokens_strategy),
        stop_sequences=draw(st.lists(_STOP_TOKEN, min_size=0, max_size=3)),
    )


@given(tc=_test_case(), defaults=_generation_defaults())
@settings(max_examples=20, deadline=None)
def test_temperature_resolution(
    tc: TestCase, defaults: GenerationDefaults
) -> None:
    """**Validates: Requirements 5.3, 5.4**

    ``opts.temperature == tc.temperature`` when the latter is not
    ``None``; otherwise it falls back to ``defaults.temperature``.
    """
    opts = resolve_generate_options(tc, defaults)
    expected = tc.temperature if tc.temperature is not None else defaults.temperature
    assert opts.temperature == expected


@given(tc=_test_case(), defaults=_generation_defaults())
@settings(max_examples=20, deadline=None)
def test_max_tokens_resolution(
    tc: TestCase, defaults: GenerationDefaults
) -> None:
    """**Validates: Requirements 5.3, 5.4**

    :class:`GenerateOptions.num_predict` mirrors ``tc.max_tokens``
    with the same None-vs-value fallback.
    """
    opts = resolve_generate_options(tc, defaults)
    expected = tc.max_tokens if tc.max_tokens is not None else defaults.max_tokens
    assert opts.num_predict == expected


@given(tc=_test_case(), defaults=_generation_defaults())
@settings(max_examples=20, deadline=None)
def test_stop_sequences_resolution(
    tc: TestCase, defaults: GenerationDefaults
) -> None:
    """**Validates: Requirements 5.3, 5.4**

    :class:`GenerateOptions.stop` reflects the three-way contract on
    ``tc.stop_sequences``:

    * ``None`` → inherit from ``defaults.stop_sequences``.
    * ``[]`` (explicit) → stays ``[]``; the defaults are *not*
      consulted.
    * Non-empty list → passed through verbatim.
    """
    opts = resolve_generate_options(tc, defaults)
    expected_list = (
        tc.stop_sequences if tc.stop_sequences is not None else defaults.stop_sequences
    )
    # Compare as lists to preserve order sensitivity — Ollama treats
    # stop sequences as ordered (in practice, any-of) but we do not
    # want to hide an accidental reordering regression.
    assert list(opts.stop or []) == list(expected_list)
