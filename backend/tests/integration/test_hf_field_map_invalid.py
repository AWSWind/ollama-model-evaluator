"""Task 26.3 — HF field-map validation.

Requirement 17.7: when a declared :class:`HFFieldMap` field cannot be
resolved on a row (missing key, out-of-range index, wrong type), the
loader raises :class:`FieldMapError` with the offending row index and
field-path reason. The scheduler's preflight will eventually
translate this into a terminal ``run-failed`` event with
``error_code=="field_map_invalid"``.

As of the current code base the scheduler does not yet accept
``kind: huggingface`` suite files (see the module docstring on
``test_remote_mode_preflight_abort.py``). The component that
enforces the "declared field missing → structured error" half of
Requirement 17.7 is :func:`ollama_evaluator.suites.huggingface.materialise_hf`
itself; this test pins its behaviour so the future scheduler hook
has a stable contract to depend on.

The task description explicitly permits this narrowing:

    Use whichever path is exercised by the current scheduler. If
    the scheduler doesn't support ``kind: huggingface`` suite files
    yet, test ``materialise_hf`` directly against a broken row.

Requirements traced: 17.7.
"""

from __future__ import annotations

import pytest

from ollama_evaluator.suites.adapter_base import HFRef
from ollama_evaluator.suites.huggingface import (
    FieldMapError,
    HFFieldMap,
    HFSuiteSpec,
    materialise_hf,
)
from ollama_evaluator.suites.models import MetricConfig


def _spec_with_first_element_path() -> HFSuiteSpec:
    """Spec that declares ``answers.text[0]`` as the expected-output path.

    Mirrors the most common SQuAD / TriviaQA shape: the declared path
    assumes at least one answer is present. A row where
    ``answers.text == []`` violates the declaration and the loader
    must raise :class:`FieldMapError` rather than silently producing
    a malformed :class:`TestCase`.
    """
    return HFSuiteSpec(
        kind="huggingface",
        name="hf-broken",
        hf_ref=HFRef(repo_id="fake/dataset", split="train"),
        field_map=HFFieldMap(
            prompt="question",
            expected_output="answers.text[0]",
        ),
        limit=None,
        seed=None,
        metrics=[MetricConfig(name="exact-match")],
    )


def _spec_with_missing_key_path() -> HFSuiteSpec:
    """Spec that declares a ``prompt`` path that does not exist on the row."""
    return HFSuiteSpec(
        kind="huggingface",
        name="hf-broken",
        hf_ref=HFRef(repo_id="fake/dataset", split="train"),
        field_map=HFFieldMap(prompt="missing_prompt_field"),
        limit=None,
        seed=None,
        metrics=[MetricConfig(name="exact-match")],
    )


def test_empty_list_for_indexed_path_raises_field_map_error() -> None:
    """Declared ``answers.text[0]`` with an empty ``answers.text`` list.

    :class:`FieldMapError` must carry:

    * ``row_index`` — the 0-based position of the offending row
      inside the materialised list.
    * ``field`` — the full declared path, so the preflight error
      envelope can echo it verbatim.
    * ``reason`` — free-form human message suitable for UI display.
    """
    spec = _spec_with_first_element_path()
    rows = [
        {"question": "What is Q?", "answers": {"text": []}},
    ]

    with pytest.raises(FieldMapError) as info:
        materialise_hf(spec, rows)

    err = info.value
    assert err.row_index == 0
    assert err.field == "answers.text[0]"
    # Reason mentions the [0] index and the empty list — callers
    # branch on the structured fields, not the string, so we assert
    # loosely here.
    assert "[0]" in err.reason or "range" in err.reason.lower()


def test_missing_top_level_prompt_key_raises_field_map_error() -> None:
    """A row without the declared ``prompt`` key raises a ``missing key`` error."""
    spec = _spec_with_missing_key_path()
    rows = [{"question": "nope, wrong key", "answer": "x"}]

    with pytest.raises(FieldMapError) as info:
        materialise_hf(spec, rows)

    err = info.value
    assert err.row_index == 0
    assert err.field == "missing_prompt_field"
    assert "missing" in err.reason.lower()


def test_expected_type_mismatch_raises_field_map_error() -> None:
    """A resolved value of the wrong type is surfaced with a typed message.

    Declaring ``expected_output: answers`` where ``answers`` is a
    dict (not a string) is a user error the loader must reject.
    """
    spec = HFSuiteSpec(
        kind="huggingface",
        name="hf-broken",
        hf_ref=HFRef(repo_id="fake/dataset", split="train"),
        field_map=HFFieldMap(
            prompt="question",
            expected_output="answers",
        ),
        limit=None,
        seed=None,
        metrics=[MetricConfig(name="exact-match")],
    )
    rows = [
        {"question": "Q?", "answers": {"text": ["A"], "answer_start": [0]}},
    ]

    with pytest.raises(FieldMapError) as info:
        materialise_hf(spec, rows)

    err = info.value
    assert err.row_index == 0
    assert err.field == "expected_output"
    # The loader's _resolve_str path checks for ``str`` and reports
    # the actual type. We only assert the type-name appears so the
    # test tolerates message-tweaking.
    assert "dict" in err.reason or "str" in err.reason


def test_error_fires_on_first_broken_row_not_partial_suite() -> None:
    """Property 44 — the loader raises on the first bad row, not a partial suite.

    Given a list of rows where the *second* row is broken, the loader
    must raise :class:`FieldMapError` carrying ``row_index == 1`` and
    must not return a partial :class:`EvaluationSuite` containing
    only the first row.
    """
    spec = _spec_with_first_element_path()
    rows = [
        {"question": "ok", "answers": {"text": ["A"]}},
        {"question": "bad", "answers": {"text": []}},
    ]

    with pytest.raises(FieldMapError) as info:
        materialise_hf(spec, rows)

    assert info.value.row_index == 1
    assert info.value.field == "answers.text[0]"
