"""Property 42: Adapter output is a valid Evaluation_Suite.

For every public-benchmark adapter ``a ∈ {mmlu, hellaswag,
truthfulqa, gsm8k, humaneval}`` and every well-formed sequence of
source rows ``R`` for that adapter,
``a.rows_to_suite(R, o)`` returns an :class:`EvaluationSuite` that

1. **Validates structurally** (Property 2): non-empty ``name``,
   unique ``test_cases[i].id``, non-empty ``prompt`` on every test
   case, non-empty ``metrics`` on every test case.
2. **Round-trips through dump/load** (Property 1) in both ``"yaml"``
   and ``"json"`` formats.

The property is stated in ``.kiro/specs/ollama-model-evaluator/design.md``
§Correctness Properties as Property 42 and validates Requirements
3.3, 4.1, 4.3, 17.1, 17.2, 17.8.

Approach
--------
For each adapter we write a Hypothesis strategy that produces
well-formed row dicts matching the adapter's input contract (the
columns documented in the Dataset sources table of the design). Each
test composes that strategy with a strategy for :class:`AdapterOptions`
and asserts the two invariants above. Source rows are ASCII-only and
bounded in size so the tests focus on adapter logic rather than on
YAML/JSON string-escaping concerns (which are already covered by the
upstream libraries that Property 1 exercises via the
:mod:`tests.property.generators` module).
"""

from __future__ import annotations

import string
from collections.abc import Callable
from typing import Any, Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.suites import dump_suite, load_suite_from_string
from ollama_evaluator.suites.adapter_base import AdapterOptions
from ollama_evaluator.suites.gsm8k import Gsm8kAdapter
from ollama_evaluator.suites.hellaswag import HellaSwagAdapter
from ollama_evaluator.suites.humaneval import HumanEvalAdapter
from ollama_evaluator.suites.mmlu import MMluAdapter
from ollama_evaluator.suites.models import EvaluationSuite
from ollama_evaluator.suites.truthfulqa import TruthfulQaAdapter

# ASCII letters + digits + space/dash/underscore keep generated content
# safe to round-trip through JSON and YAML without surfacing escaping
# bugs unrelated to the property under test. This matches the alphabet
# used by ``tests.property.generators.simple_strings``.
_TEXT_ALPHABET = string.ascii_letters + string.digits + " -_"

# A short, bounded ASCII string used for every free-form text field
# generated below. Size bounds are chosen to keep Hypothesis fast
# (each draw builds a full ``EvaluationSuite``).
_short_text = st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=20)

# MMLU subjects can be any identifier-shaped string. We constrain to a
# small pool so the MMLU "one suite per subject" path is exercised on
# every draw — if subjects were fully random, most draws would produce
# single-row suites per subject.
_mmlu_subjects = st.sampled_from(
    ["abstract_algebra", "high_school_biology", "philosophy", "logic"]
)

# HellaSwag activity labels; free-form strings but a small sampled pool
# keeps the tag-shape realistic.
_hellaswag_activities = st.sampled_from(
    ["Baby", "Home-and-Garden", "Sports", "Food"]
)

# TruthfulQA categories from the published dataset.
_truthfulqa_categories = st.sampled_from(
    ["Misconceptions", "Health", "Law", "Science"]
)


# ---------------------------------------------------------------------------
# Row strategies — one per adapter
# ---------------------------------------------------------------------------


@st.composite
def _mmlu_row(draw: st.DrawFn) -> dict[str, Any]:
    """Draw one MMLU row matching ``design.md`` §Dataset sources."""
    return {
        "question": draw(_short_text),
        "choices": [
            draw(_short_text),
            draw(_short_text),
            draw(_short_text),
            draw(_short_text),
        ],
        "answer": draw(st.integers(min_value=0, max_value=3)),
        "subject": draw(_mmlu_subjects),
    }


@st.composite
def _hellaswag_row(draw: st.DrawFn) -> dict[str, Any]:
    """Draw one HellaSwag row."""
    return {
        "ctx": draw(_short_text),
        "endings": [
            draw(_short_text),
            draw(_short_text),
            draw(_short_text),
            draw(_short_text),
        ],
        "label": draw(st.sampled_from(["0", "1", "2", "3"])),
        "ind": draw(st.integers(min_value=0, max_value=1_000_000)),
        "activity_label": draw(_hellaswag_activities),
    }


@st.composite
def _truthfulqa_row(draw: st.DrawFn) -> dict[str, Any]:
    """Draw one TruthfulQA MC1 row.

    Generates between 2 and 5 choices so multi-option MC1 behaviour is
    covered (real TruthfulQA has 4–13 options). Exactly one label is
    ``1``; the others are ``0``.
    """
    num_choices = draw(st.integers(min_value=2, max_value=5))
    choices = [draw(_short_text) for _ in range(num_choices)]
    correct_idx = draw(st.integers(min_value=0, max_value=num_choices - 1))
    labels = [1 if i == correct_idx else 0 for i in range(num_choices)]
    return {
        "question": draw(_short_text),
        "mc1_targets": {"choices": choices, "labels": labels},
        "category": draw(_truthfulqa_categories),
    }


@st.composite
def _gsm8k_row(draw: st.DrawFn) -> dict[str, Any]:
    """Draw one GSM8K row; the ``answer`` always ends with ``#### N``."""
    # Keep the number a pure integer to match the real dataset. The
    # regex accepts decimals and thousands separators, but the gold
    # field never uses them in v1.0.
    gold = draw(st.integers(min_value=-10_000, max_value=10_000))
    solution_text = draw(_short_text)
    return {
        "question": draw(_short_text),
        "answer": f"{solution_text}\n#### {gold}",
    }


@st.composite
def _humaneval_row(draw: st.DrawFn) -> dict[str, Any]:
    """Draw one HumanEval row."""
    return {
        "prompt": draw(_short_text),
        "canonical_solution": draw(_short_text),
        "test": draw(_short_text),
        "entry_point": draw(_short_text),
    }


@st.composite
def _adapter_options(draw: st.DrawFn) -> AdapterOptions:
    """Draw a valid :class:`AdapterOptions` covering both ``None`` arms.

    ``limit`` is drawn in ``1..50`` to match the 1–10 row counts used
    by the row strategies below. ``seed`` exercises both the
    "inherit source order" and "shuffle deterministically" branches.
    """
    return AdapterOptions(
        limit=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=50))),
        seed=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**30))),
        # ``subject`` is MMLU-specific; other adapters ignore it.
        subject=draw(st.one_of(st.none(), _mmlu_subjects)),
        form="mc1",
    )


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _assert_structurally_valid(suite: EvaluationSuite) -> None:
    """Assert Property 2's structural invariants on ``suite``."""
    # A suite with no test cases is permitted by the model validators
    # only when the input ``rows`` list was empty. Adapters should
    # short-circuit on empty input into a valid empty suite *in
    # principle*, but :class:`EvaluationSuite` rejects empty
    # ``test_cases``, so we assert at least one here to surface any
    # regression that returns an invalid empty suite.
    assert len(suite.test_cases) > 0, "adapter produced a suite with no test cases"
    assert suite.name.strip() != ""
    seen_ids: set[str] = set()
    for tc in suite.test_cases:
        assert tc.prompt != "", f"empty prompt on {tc.id!r}"
        assert len(tc.metrics) > 0, f"no metrics on {tc.id!r}"
        assert tc.id not in seen_ids, f"duplicate id {tc.id!r}"
        seen_ids.add(tc.id)


def _assert_round_trips(
    suite: EvaluationSuite, fmt: Literal["yaml", "json"]
) -> None:
    """Assert the Property-1 round-trip for ``suite`` in ``fmt``."""
    rebuilt = load_suite_from_string(dump_suite(suite, fmt), fmt)
    assert rebuilt == suite


# ---------------------------------------------------------------------------
# Adapter-specific property tests
# ---------------------------------------------------------------------------


def _run_adapter_property(
    adapter_factory: Callable[[], Any],
    rows: list[dict[str, Any]],
    opts: AdapterOptions,
) -> None:
    """Shared body for each per-adapter test."""
    adapter = adapter_factory()
    suite = adapter.rows_to_suite(rows, opts)
    _assert_structurally_valid(suite)
    for fmt in ("yaml", "json"):
        _assert_round_trips(suite, fmt)


@given(
    rows=st.lists(_mmlu_row(), min_size=1, max_size=10),
    opts=_adapter_options(),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_mmlu_adapter_produces_valid_suite(
    rows: list[dict[str, Any]], opts: AdapterOptions
) -> None:
    """**Validates: Requirements 3.3, 4.1, 4.3, 17.1, 17.2, 17.8**

    ``MMluAdapter.rows_to_suite`` produces a structurally valid
    :class:`EvaluationSuite` that round-trips through YAML and JSON.
    Rows whose ``subject`` does not match ``opts.subject`` are
    filtered out by the adapter; to keep the generated suite
    non-empty we force a ``subject`` match by either leaving
    ``opts.subject`` as ``None`` or injecting at least one row with
    that subject.
    """
    # Guarantee at least one row survives the subject filter.
    if opts.subject is not None and not any(r["subject"] == opts.subject for r in rows):
        # Mutate the first row's subject in place so the filter keeps
        # it. The Hypothesis-generated dict is fresh per example, so
        # in-place mutation is safe.
        rows[0] = {**rows[0], "subject": opts.subject}
    _run_adapter_property(MMluAdapter, rows, opts)


@given(
    rows=st.lists(_hellaswag_row(), min_size=1, max_size=10, unique_by=lambda r: r["ind"]),
    opts=_adapter_options(),
)
@settings(max_examples=20, deadline=None)
def test_hellaswag_adapter_produces_valid_suite(
    rows: list[dict[str, Any]], opts: AdapterOptions
) -> None:
    """**Validates: Requirements 3.3, 4.1, 4.3, 17.1, 17.2, 17.8**

    HellaSwag's ``id`` uses ``row["ind"]`` directly so duplicate
    ``ind`` values would produce duplicate ids and break Property 2.
    The strategy uses ``unique_by`` to keep ids unique — the adapter
    itself does not de-dupe.
    """
    _run_adapter_property(HellaSwagAdapter, rows, opts)


@given(
    rows=st.lists(_truthfulqa_row(), min_size=1, max_size=10),
    opts=_adapter_options(),
)
@settings(max_examples=20, deadline=None)
def test_truthfulqa_adapter_produces_valid_suite(
    rows: list[dict[str, Any]], opts: AdapterOptions
) -> None:
    """**Validates: Requirements 3.3, 4.1, 4.3, 17.1, 17.2, 17.8**

    TruthfulQA v1 supports only ``form="mc1"``; the options strategy
    pins it there.
    """
    _run_adapter_property(TruthfulQaAdapter, rows, opts)


@given(
    rows=st.lists(_gsm8k_row(), min_size=1, max_size=10),
    opts=_adapter_options(),
)
@settings(max_examples=20, deadline=None)
def test_gsm8k_adapter_produces_valid_suite(
    rows: list[dict[str, Any]], opts: AdapterOptions
) -> None:
    """**Validates: Requirements 3.3, 4.1, 4.3, 17.1, 17.2, 17.8**"""
    _run_adapter_property(Gsm8kAdapter, rows, opts)


@given(
    rows=st.lists(_humaneval_row(), min_size=1, max_size=10),
    opts=_adapter_options(),
)
@settings(max_examples=20, deadline=None)
def test_humaneval_adapter_produces_valid_suite(
    rows: list[dict[str, Any]], opts: AdapterOptions
) -> None:
    """**Validates: Requirements 3.3, 4.1, 4.3, 17.1, 17.2, 17.8**"""
    _run_adapter_property(HumanEvalAdapter, rows, opts)


# ---------------------------------------------------------------------------
# Bonus coverage: explicit degenerate cases that are not easy to hit with
# Hypothesis generation alone.
# ---------------------------------------------------------------------------


_MMLU_ROW: dict[str, Any] = {
    "question": "q",
    "choices": ["a", "b", "c", "d"],
    "answer": 0,
    "subject": "philosophy",
}
_HELLASWAG_ROW: dict[str, Any] = {
    "ctx": "x",
    "endings": ["a", "b", "c", "d"],
    "label": "2",
    "ind": 1,
    "activity_label": "Baby",
}
_TRUTHFULQA_ROW: dict[str, Any] = {
    "question": "q",
    "mc1_targets": {"choices": ["a", "b"], "labels": [0, 1]},
    "category": "Law",
}
_GSM8K_ROW: dict[str, Any] = {"question": "q", "answer": "steps\n#### 42"}
_HUMANEVAL_ROW: dict[str, Any] = {
    "prompt": "def f():",
    "canonical_solution": "    pass",
    "test": "assert True",
    "entry_point": "f",
}


@pytest.mark.parametrize(
    "adapter_factory, rows",
    [
        (MMluAdapter, [_MMLU_ROW]),
        (HellaSwagAdapter, [_HELLASWAG_ROW]),
        (TruthfulQaAdapter, [_TRUTHFULQA_ROW]),
        (Gsm8kAdapter, [_GSM8K_ROW]),
        (HumanEvalAdapter, [_HUMANEVAL_ROW]),
    ],
)
def test_single_row_adapters_round_trip(
    adapter_factory: Callable[[], Any], rows: list[dict[str, Any]]
) -> None:
    """Smoke test: every adapter handles a single hand-authored row.

    Guards against a regression where a clever optimisation special-
    cases empty input and breaks the 1-row path.
    """
    _run_adapter_property(adapter_factory, rows, AdapterOptions())
