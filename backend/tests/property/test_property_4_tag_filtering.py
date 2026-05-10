"""Property 4: Tag and name filtering.

For every :class:`EvaluationSuite` ``s`` and any
:class:`RunConfig.tag_filter` / :class:`RunConfig.suites`, the test
cases selected for execution equal::

    { tc ∈ s.test_cases
      | s.name ∈ config.suites
      ∧ (config.tag_filter == [] ∨ tc.tags ∩ config.tag_filter ≠ ∅) }

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness
Properties as Property 4 and validates Requirement 3.6.

Approach
--------
A Hypothesis strategy builds a small universe of suites plus a
``RunConfig`` whose filters are drawn *after* the suites so the
generator can reuse the suites' tag and name pools. The property
test reconstructs the reference selection set via a plain
set-comprehension and asserts equality against the (deduplicated)
set of :class:`TestCase` ids in the output of
:func:`select_executions`.

``max_examples=20`` and ``deadline=None`` match the floor set in
``design.md``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.config import RunConfig
from ollama_evaluator.runner.selection import select_executions
from ollama_evaluator.suites.models import EvaluationSuite, MetricConfig, TestCase

# Reuse a small tag alphabet so the generator actually produces
# overlapping tag filters — fully-random tag pools would make
# ``tc.tags ∩ tag_filter`` empty almost every time, and the
# positive-filter branch would rarely run.
_TAG_POOL = ["math", "code", "reading", "truthful"]
_SUITE_NAME_POOL = ["suite-a", "suite-b", "suite-c"]


def _make_test_case(idx: int, tags: list[str]) -> TestCase:
    """Construct a minimal valid :class:`TestCase` with the given ``tags``."""
    return TestCase(
        id=f"tc-{idx}",
        prompt="prompt",
        tags=tags,
        metrics=[MetricConfig(name="exact-match")],
    )


@st.composite
def _suite(draw: st.DrawFn) -> EvaluationSuite:
    """Draw a small :class:`EvaluationSuite` with tagged test cases."""
    name = draw(st.sampled_from(_SUITE_NAME_POOL))
    num_cases = draw(st.integers(min_value=1, max_value=4))
    # Draw test cases with unique ids so the suite-level validator is
    # satisfied without filtering.
    test_cases = [
        _make_test_case(
            idx=i,
            tags=draw(
                st.lists(st.sampled_from(_TAG_POOL), min_size=0, max_size=3, unique=True)
            ),
        )
        for i in range(num_cases)
    ]
    return EvaluationSuite(name=name, test_cases=test_cases)


@st.composite
def _suites_and_config(
    draw: st.DrawFn,
) -> tuple[list[EvaluationSuite], RunConfig]:
    """Draw a universe of suites plus a :class:`RunConfig` that filters them."""
    # Unique by name so the selection reference is well-defined; the
    # suite-name pool is small, so sampling with replacement would
    # make ``by_name`` collapse.
    suites: list[EvaluationSuite] = draw(
        st.lists(_suite(), min_size=1, max_size=3, unique_by=lambda s: s.name)
    )
    # Filter may reference known and unknown suite names so the
    # "unknown suite gets silently skipped" branch is covered.
    suite_filter = draw(
        st.lists(
            st.sampled_from(_SUITE_NAME_POOL),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    # ``tag_filter`` can be empty (trivial accept) or a non-empty
    # sample from the tag pool.
    tag_filter = draw(
        st.lists(
            st.sampled_from(_TAG_POOL), min_size=0, max_size=3, unique=True
        )
    )
    config = RunConfig(
        models=["model-x"],
        suites=suite_filter,
        tag_filter=tag_filter,
        repetitions=1,
    )
    return suites, config


def _reference_selection(
    suites: list[EvaluationSuite], config: RunConfig
) -> set[tuple[str, str]]:
    """Compute the reference ``(suite_name, test_case_id)`` selection set.

    This mirrors Property 4's set comprehension exactly so the test
    body's assertion is a pure set-equality check.
    """
    suite_names = set(config.suites)
    tag_filter = set(config.tag_filter)
    reference: set[tuple[str, str]] = set()
    for s in suites:
        if s.name not in suite_names:
            continue
        for tc in s.test_cases:
            if tag_filter and not (set(tc.tags) & tag_filter):
                continue
            reference.add((s.name, tc.id))
    return reference


@given(data=_suites_and_config())
@settings(max_examples=20, deadline=None)
def test_tag_and_name_filtering(
    data: tuple[list[EvaluationSuite], RunConfig]
) -> None:
    """**Validates: Requirement 3.6**

    The ``(suite_name, test_case_id)`` set produced by
    :func:`select_executions` equals the reference set computed
    directly from Property 4's specification.
    """
    suites, config = data
    expected = _reference_selection(suites, config)
    # ``select_executions`` returns ``(model, tc, repetition)`` tuples.
    # Because the generator pins ``repetitions=1`` and a single model,
    # the *set* of selected test cases is exactly the property's
    # reference set. We key the actual set by Python-object identity
    # of the :class:`TestCase` so the lookup back to the enclosing
    # suite is unambiguous even when two suites happen to generate
    # the same ``tc.id`` string.
    suite_by_tc_identity: dict[int, str] = {
        id(tc): s.name for s in suites for tc in s.test_cases
    }
    actual: set[tuple[str, str]] = {
        (suite_by_tc_identity[id(tc)], tc.id)
        for _model, tc, _rep in select_executions(suites, config)
    }
    assert actual == expected
