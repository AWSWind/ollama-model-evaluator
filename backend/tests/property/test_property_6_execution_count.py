"""Property 6: Execution count and coverage.

For any :class:`RunConfig` with models ``M``, selected test cases
``K``, and ``repetitions = R``, a successful Run dispatches exactly
``|M| · |K| · R`` executions, with exactly one execution per
``(model, test_case_id, repetition)`` tuple.

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness
Properties as Property 6 and validates Requirement 5.1.

Approach
--------
Reuses the same :func:`~tests.property.test_property_4_tag_filtering._suites_and_config`
strategy scaffolding but widens ``models`` and ``repetitions`` so the
multiplication under test is non-trivial. The assertions are:

1. ``len(select_executions(...)) == |M| · |K| · R``, where ``|K|``
   is computed independently from the strategy's own reference
   selection.
2. The multiset of ``(model, test_case_id, repetition)`` tuples has
   no duplicates (every tuple appears exactly once).

``max_examples=20`` and ``deadline=None`` match the floor set in
``design.md``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.runner.selection import select_executions
from ollama_evaluator.suites.models import EvaluationSuite, MetricConfig, TestCase

_TAG_POOL = ["math", "code", "reading"]
_SUITE_NAME_POOL = ["suite-a", "suite-b"]


def _make_test_case(suite_name: str, idx: int, tags: list[str]) -> TestCase:
    """Construct a :class:`TestCase` with a suite-scoped id.

    Encoding the suite name into the id keeps ``test_case_id``
    globally unique across the generated universe, so the
    "exactly one per tuple" assertion is unambiguous even when two
    suites would otherwise share an id.
    """
    return TestCase(
        id=f"{suite_name}/tc-{idx}",
        prompt="prompt",
        tags=tags,
        metrics=[MetricConfig(name="exact-match")],
    )


@st.composite
def _suite(draw: st.DrawFn) -> EvaluationSuite:
    """Draw a small :class:`EvaluationSuite` with tagged test cases."""
    name = draw(st.sampled_from(_SUITE_NAME_POOL))
    num_cases = draw(st.integers(min_value=1, max_value=4))
    test_cases = [
        _make_test_case(
            name,
            i,
            draw(st.lists(st.sampled_from(_TAG_POOL), min_size=0, max_size=2, unique=True)),
        )
        for i in range(num_cases)
    ]
    return EvaluationSuite(name=name, test_cases=test_cases)


@st.composite
def _suites_and_config(
    draw: st.DrawFn,
) -> tuple[list[EvaluationSuite], RunConfig]:
    """Draw suites + a :class:`RunConfig` with non-trivial ``models`` / ``R``."""
    suites: list[EvaluationSuite] = draw(
        st.lists(_suite(), min_size=1, max_size=2, unique_by=lambda s: s.name)
    )
    suite_filter = draw(
        st.lists(
            st.sampled_from([s.name for s in suites]),
            min_size=1,
            max_size=len(suites),
            unique=True,
        )
    )
    models = draw(
        st.lists(
            st.sampled_from(["m-a", "m-b", "m-c"]),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    repetitions = draw(st.integers(min_value=1, max_value=3))
    tag_filter = draw(
        st.lists(
            st.sampled_from(_TAG_POOL), min_size=0, max_size=2, unique=True
        )
    )
    config = RunConfig(
        models=models,
        suites=suite_filter,
        tag_filter=tag_filter,
        repetitions=repetitions,
    )
    return suites, config


def _selected_test_case_ids(
    suites: list[EvaluationSuite], config: RunConfig
) -> set[str]:
    """Compute ``K`` independently from :func:`select_executions`."""
    suite_names = set(config.suites)
    tag_filter = set(config.tag_filter)
    ids: set[str] = set()
    for s in suites:
        if s.name not in suite_names:
            continue
        for tc in s.test_cases:
            if tag_filter and not (set(tc.tags) & tag_filter):
                continue
            ids.add(tc.id)
    return ids


@given(data=_suites_and_config())
@settings(max_examples=20, deadline=None)
def test_execution_count_equals_m_k_r(
    data: tuple[list[EvaluationSuite], RunConfig]
) -> None:
    """**Validates: Requirement 5.1**

    ``len(select_executions(...)) == |M| · |K| · R``.
    """
    suites, config = data
    selected_ids = _selected_test_case_ids(suites, config)
    expected = len(config.models) * len(selected_ids) * config.repetitions
    assert len(select_executions(suites, config)) == expected


@given(data=_suites_and_config())
@settings(max_examples=20, deadline=None)
def test_every_tuple_appears_exactly_once(
    data: tuple[list[EvaluationSuite], RunConfig]
) -> None:
    """**Validates: Requirement 5.1**

    Every ``(model, test_case_id, repetition)`` tuple appears
    exactly once in the selection. ``len(set) == len(list)`` is the
    standard Python idiom for "no duplicates".
    """
    suites, config = data
    tuples = [
        (model, tc.id, rep) for model, tc, rep in select_executions(suites, config)
    ]
    assert len(tuples) == len(set(tuples))


@given(data=_suites_and_config())
@settings(max_examples=20, deadline=None)
def test_selection_matches_cross_product(
    data: tuple[list[EvaluationSuite], RunConfig]
) -> None:
    """**Validates: Requirement 5.1**

    The selected tuple set equals the cross product
    ``{(m, tc_id, r) | m ∈ M, tc_id ∈ K, r ∈ 1..R}``. This is the
    "exactly one call per tuple" invariant stated set-wise rather
    than length-wise.
    """
    suites, config = data
    selected_ids = _selected_test_case_ids(suites, config)
    expected = {
        (m, tc_id, r)
        for m in config.models
        for tc_id in selected_ids
        for r in range(1, config.repetitions + 1)
    }
    actual = {
        (m, tc.id, r) for m, tc, r in select_executions(suites, config)
    }
    assert actual == expected
