"""Property 44: HuggingFace field-map totality and injectivity.

For any :class:`HFSuiteSpec` ``spec`` whose field map resolves on
every row in a list ``R``:

1. **Totality.** :func:`materialise_hf` produces exactly
   ``min(|R|, spec.limit or |R|)`` :class:`TestCase` objects with no
   row silently dropped.
2. **Determinism.** Under a fixed ``spec.seed`` and fixed ``R``, two
   calls to :func:`materialise_hf` yield byte-identical
   :class:`TestCase` payloads.
3. **Totality failure mode.** For any ``R`` where a declared path
   fails to resolve on at least one row, :func:`materialise_hf`
   raises :class:`FieldMapError` without producing a partial suite.

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness
Properties as Property 44 and validates Requirements 3.3, 3.5, 17.2,
17.7.

Approach
--------
A Hypothesis composite ``_row_pair`` generates a row plus the
corresponding field map paths that *will* resolve on every draw. We
draw rows from this pair-strategy so the adapter's totality path is
exercised without needing to rediscover valid shapes for every
example.

For the failure mode we take a well-formed row stream, pick one row,
and either delete or null out a declared field — then assert that
:func:`materialise_hf` raises :class:`FieldMapError` without
returning any suite.

Determinism exercises the deterministic-sub-sample branch by pinning
a ``limit`` smaller than ``len(rows)`` and a concrete ``seed``.

``max_examples=20`` and ``deadline=None`` match the testing-strategy
floor set in ``design.md``.
"""

from __future__ import annotations

import string
from copy import deepcopy
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.suites.adapter_base import HFRef
from ollama_evaluator.suites.huggingface import (
    FieldMapError,
    HFFieldMap,
    HFSuiteSpec,
    materialise_hf,
)
from ollama_evaluator.suites.models import MetricConfig

# ASCII-only text alphabet for generated field values. Avoids JSON/YAML
# escaping concerns that belong to upstream library tests.
_TEXT = st.text(
    alphabet=string.ascii_letters + string.digits + " -_",
    min_size=1,
    max_size=12,
)


@st.composite
def _well_formed_row(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a row that satisfies the fixed field map declared in ``_FIELD_MAP``.

    The shape matches the one documented in ``design.md`` §Generic
    HuggingFace loader: a top-level ``question`` (the prompt source),
    a nested ``answers.text[0]`` (the expected output source), a
    ``category`` (tag source), and an optional ``system`` prompt
    that the field map does *not* reference — its purpose is to
    verify that extra row fields are silently ignored.
    """
    # Draw 1..3 strings for ``answers.text`` so ``answers.text[0]``
    # always resolves; indices 1 and 2 exist only on some draws to
    # exercise the list-bounds branch inside the resolver.
    answers = [draw(_TEXT) for _ in range(draw(st.integers(min_value=1, max_value=3)))]
    return {
        "question": draw(_TEXT),
        "answers": {"text": answers},
        "category": draw(_TEXT),
        "system": draw(_TEXT),
    }


_FIELD_MAP = HFFieldMap(
    prompt="question",
    expected_output="answers.text[0]",
    tags_from=["category"],
)

_HF_REF = HFRef(repo_id="demo/qa", config="plain_text", split="validation")


def _spec(
    *, limit: int | None = None, seed: int | None = None
) -> HFSuiteSpec:
    """Build a canonical :class:`HFSuiteSpec` for this property's tests."""
    return HFSuiteSpec(
        name="demo",
        hf_ref=_HF_REF,
        field_map=_FIELD_MAP,
        limit=limit,
        seed=seed,
        metrics=[MetricConfig(name="exact-match")],
    )


# ---------------------------------------------------------------------------
# Totality: every row produces exactly one TestCase (up to ``limit``)
# ---------------------------------------------------------------------------


@given(rows=st.lists(_well_formed_row(), min_size=1, max_size=10))
@settings(max_examples=20, deadline=None)
def test_materialise_hf_is_total_without_limit(
    rows: list[dict[str, Any]],
) -> None:
    """**Validates: Requirements 3.3, 3.5, 17.2, 17.7**

    With no ``limit``, the output has exactly ``len(rows)`` test
    cases and no row is silently dropped.
    """
    suite = materialise_hf(_spec(), rows=rows)
    assert len(suite.test_cases) == len(rows)


@given(
    rows=st.lists(_well_formed_row(), min_size=1, max_size=10),
    limit=st.integers(min_value=1, max_value=15),
)
@settings(max_examples=20, deadline=None)
def test_materialise_hf_respects_limit(
    rows: list[dict[str, Any]], limit: int
) -> None:
    """**Validates: Requirements 3.3, 3.5, 17.2, 17.7**

    With a ``limit``, the output has exactly ``min(|R|, limit)``
    test cases. The test sweeps ``limit`` both below and above
    ``len(rows)`` so the saturation branch is covered too.
    """
    spec = _spec(limit=limit)
    suite = materialise_hf(spec, rows=rows)
    assert len(suite.test_cases) == min(len(rows), limit)


# ---------------------------------------------------------------------------
# Determinism: same (spec, rows, seed) → byte-identical test cases
# ---------------------------------------------------------------------------


@given(
    rows=st.lists(_well_formed_row(), min_size=2, max_size=10),
    seed=st.integers(min_value=0, max_value=2**30),
)
@settings(
    max_examples=20,
    deadline=None,
    # ``len(rows) >= 2`` filtering + the ``limit < len(rows)`` guard
    # below reject a small fraction of draws; suppress the health
    # check so the test stays stable across generator tweaks.
    suppress_health_check=[HealthCheck.filter_too_much],
)
def test_materialise_hf_is_deterministic_under_seed(
    rows: list[dict[str, Any]], seed: int
) -> None:
    """**Validates: Requirements 3.3, 3.5, 17.2, 17.7**

    Two :func:`materialise_hf` calls with the same ``spec`` (same
    ``limit``/``seed``) and the same row list return equal suites.
    Testing at ``limit = len(rows) - 1`` exercises the
    seeded-shuffle branch that determinism most relies on.
    """
    limit = len(rows) - 1
    spec = _spec(limit=limit, seed=seed)
    first = materialise_hf(spec, rows=rows)
    second = materialise_hf(spec, rows=rows)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


# ---------------------------------------------------------------------------
# Failure mode: unresolvable path → FieldMapError, no partial suite
# ---------------------------------------------------------------------------


@given(
    rows=st.lists(_well_formed_row(), min_size=1, max_size=5),
    target_index=st.integers(min_value=0, max_value=10_000),
    mutation=st.sampled_from(["delete", "set_none", "wrong_type"]),
)
@settings(max_examples=20, deadline=None)
def test_materialise_hf_raises_fieldmap_error_on_missing_path(
    rows: list[dict[str, Any]], target_index: int, mutation: str
) -> None:
    """**Validates: Requirements 3.3, 3.5, 17.2, 17.7**

    Mutating the ``answers.text[0]`` source on one row in one of
    three ways (remove ``text``, set it to ``None``, or replace it
    with the wrong type) causes :func:`materialise_hf` to raise
    :class:`FieldMapError` without producing a partial suite.
    """
    idx = target_index % len(rows)
    broken_rows = deepcopy(rows)
    if mutation == "delete":
        # Remove the ``text`` key so ``answers.text`` fails to resolve.
        del broken_rows[idx]["answers"]["text"]
    elif mutation == "set_none":
        broken_rows[idx]["answers"]["text"] = None
    else:
        # Wrong type: strings can't be indexed with ``[0]`` as a list.
        broken_rows[idx]["answers"]["text"] = "not-a-list"

    try:
        materialise_hf(_spec(), rows=broken_rows)
    except FieldMapError as err:
        assert err.row_index == idx
        # The two mutation modes address different fields on the
        # resolver; the exact reason string differs but both include
        # ``answers.text`` in the ``field`` attribute.
        assert "answers" in err.field
        return
    # Falling off the try means the function succeeded — that is a
    # totality failure and falsifies the property.
    raise AssertionError(
        "materialise_hf should have raised FieldMapError on the broken row"
    )


def test_materialise_hf_has_no_partial_state_on_failure(tmp_path: object) -> None:
    """**Validates: Requirements 3.3, 3.5, 17.2, 17.7**

    A single example-level assertion that, when :func:`materialise_hf`
    raises :class:`FieldMapError`, no :class:`EvaluationSuite` is
    returned (even a partial one). The generator-based test above
    already covers this via Hypothesis; this example-level test
    keeps the intent legible.
    """
    del tmp_path  # unused; kept so pytest fixture names stay stable
    good_row = {
        "question": "q",
        "answers": {"text": ["a"]},
        "category": "c",
    }
    broken_row = {"question": "q2", "answers": {"text": []}, "category": "c2"}
    try:
        materialise_hf(_spec(), rows=[good_row, broken_row])
    except FieldMapError as err:
        assert err.row_index == 1
    else:
        raise AssertionError("expected FieldMapError for the out-of-range path")
