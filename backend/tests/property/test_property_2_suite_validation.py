"""Property 2: Evaluation_Suite validation.

Validates the structural-invariant side of the suite loader: loading a
serialised suite succeeds *if and only if* every required invariant
holds (non-empty ``name``, non-empty ``prompt``, non-empty ``metrics``,
unique ``test_cases[i].id``), and when it fails the raised
:class:`~ollama_evaluator.suites.loader.SuiteValidationError` carries
structured diagnostics identifying the offending path, test-case id,
and missing/invalid field.

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness Properties
as Property 2 and is driven by:

* **Requirement 3.3** — every suite must have a ``name``, every
  ``TestCase`` must have a non-empty ``prompt``, a non-empty
  ``metrics`` list, and an ``id`` that is unique within the suite.
* **Requirement 3.5** — when validation fails, the loader must raise
  an error that names the offending test case (when the failure is
  scoped to one) and the missing/invalid field.

Approach
--------
Each test is a Hypothesis generative pair: first draw a *valid* suite
from :func:`~tests.property.generators.evaluation_suites`, then mutate
the JSON-dumped form in exactly one targeted way (remove / blank / set
an offending value) and assert the loader raises
:class:`SuiteValidationError` with the expected ``missing_field``,
``test_case_id``, and/or message contents.

The malformed mutations are serialised with :func:`json.dumps` rather
than ``dump_suite`` so:

1. Parsing is deterministic — ``json.loads`` surfaces any non-JSON
   syntax issue as a :class:`json.JSONDecodeError`, which is already
   tested elsewhere. Here we want the Pydantic validation layer, not
   the YAML / JSON lexer.
2. The writer's canonicalisation pass (sorted keys, etc.) does not
   sneak in while a test expects a *specific* malformed shape.

``max_examples=20`` matches the floor set by the testing strategy in
``design.md``; ``deadline=None`` avoids flaky timing failures on slow
CI workers.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.suites import (
    SuiteValidationError,
    load_suite_from_string,
)
from ollama_evaluator.suites.models import EvaluationSuite

from .generators import evaluation_suites


def _as_json_dict(suite: EvaluationSuite) -> dict[str, Any]:
    """Return the JSON-mode ``model_dump`` of ``suite`` as a mutable dict.

    Using ``mode="json"`` converts any non-JSON-native types (e.g.
    ``None`` stays as ``None``; enums / paths would be stringified)
    into values that round-trip through :func:`json.dumps` without a
    custom encoder. The returned dict is an independent deep-copy safe
    to mutate — :meth:`BaseModel.model_dump` always returns fresh
    containers.
    """
    return suite.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Positive direction: valid input succeeds
# ---------------------------------------------------------------------------


@given(suite=evaluation_suites())
@settings(max_examples=20, deadline=None)
def test_valid_suite_loads_successfully(suite: EvaluationSuite) -> None:
    """**Validates: Requirements 3.3, 3.5**

    For every valid :class:`EvaluationSuite` drawn from the strategy,
    :func:`load_suite_from_string` returns a model equal to the input.

    This restates Property 1's direction over JSON only to give the
    validation side of the biconditional a named, explicit test — the
    failure-direction tests below all *mutate* a valid suite, so if
    this test ever regresses we know the mutations are meaningless
    (they'd be mutating an already-broken base).
    """
    text = json.dumps(_as_json_dict(suite))
    rebuilt = load_suite_from_string(text, "json")
    assert rebuilt == suite


# ---------------------------------------------------------------------------
# Negative direction: each invariant violation raises with diagnostics
# ---------------------------------------------------------------------------


@given(suite=evaluation_suites(), data=st.data())
@settings(max_examples=20, deadline=None)
def test_missing_prompt_raises_with_diagnostics(
    suite: EvaluationSuite, data: st.DataObject
) -> None:
    """**Validates: Requirements 3.3, 3.5**

    Removing the ``prompt`` key from any single ``TestCase`` produces
    a :class:`SuiteValidationError` whose ``missing_field`` is
    ``"prompt"`` and whose ``test_case_id`` is the id of that
    ``TestCase``. The offending test case is picked by Hypothesis so
    the property covers every index across generated suites, including
    the first and last positions.
    """
    n = len(suite.test_cases)
    idx = data.draw(st.integers(min_value=0, max_value=n - 1))
    payload = _as_json_dict(suite)
    tc = cast(dict[str, Any], payload["test_cases"][idx])
    offending_id = cast(str, tc["id"])
    del tc["prompt"]
    text = json.dumps(payload)

    with pytest.raises(SuiteValidationError) as excinfo:
        load_suite_from_string(text, "json")
    err = excinfo.value
    # Structured diagnostics (Requirement 3.5): the loader exposes the
    # dotted path of the offending field via ``missing_field`` and the
    # offending case's id via ``test_case_id``. The human-readable
    # message is the raw Pydantic ``msg`` ("Field required") so
    # downstream tooling can render it verbatim; the dotted path does
    # the job of naming the location.
    assert err.missing_field == f"test_cases.{idx}.prompt"
    assert err.test_case_id == offending_id
    assert "Field required" in err.message


@given(suite=evaluation_suites(), data=st.data())
@settings(max_examples=20, deadline=None)
def test_missing_metrics_raises_with_diagnostics(
    suite: EvaluationSuite, data: st.DataObject
) -> None:
    """**Validates: Requirements 3.3, 3.5**

    Removing the ``metrics`` key from any single ``TestCase`` produces
    a :class:`SuiteValidationError` whose ``missing_field`` is
    ``"metrics"`` and whose ``test_case_id`` identifies the offending
    test case.
    """
    n = len(suite.test_cases)
    idx = data.draw(st.integers(min_value=0, max_value=n - 1))
    payload = _as_json_dict(suite)
    tc = cast(dict[str, Any], payload["test_cases"][idx])
    offending_id = cast(str, tc["id"])
    del tc["metrics"]
    text = json.dumps(payload)

    with pytest.raises(SuiteValidationError) as excinfo:
        load_suite_from_string(text, "json")
    err = excinfo.value
    # Structured diagnostics (Requirement 3.5): ``missing_field``
    # carries the full dotted path of the offending location. The
    # loader surfaces the raw Pydantic ``msg`` in ``message``.
    assert err.missing_field == f"test_cases.{idx}.metrics"
    assert err.test_case_id == offending_id
    assert "Field required" in err.message


@given(suite=evaluation_suites(), data=st.data())
@settings(max_examples=20, deadline=None)
def test_empty_prompt_references_offending_test_case(
    suite: EvaluationSuite, data: st.DataObject
) -> None:
    """**Validates: Requirements 3.3, 3.5**

    Setting ``prompt`` to the empty string on any test case produces a
    :class:`SuiteValidationError` whose ``test_case_id`` identifies
    that case and whose ``missing_field`` dotted-path pinpoints the
    offending location — so users can jump straight to the right line
    in a multi-case suite file.

    An empty prompt is a *value* violation rather than a missing-key
    violation, but the loader still reports a dotted path because
    Pydantic's ``loc`` traverses the same ``test_cases[idx].prompt``
    chain.
    """
    n = len(suite.test_cases)
    idx = data.draw(st.integers(min_value=0, max_value=n - 1))
    payload = _as_json_dict(suite)
    tc = cast(dict[str, Any], payload["test_cases"][idx])
    offending_id = cast(str, tc["id"])
    tc["prompt"] = ""
    text = json.dumps(payload)

    with pytest.raises(SuiteValidationError) as excinfo:
        load_suite_from_string(text, "json")
    err = excinfo.value
    assert err.test_case_id == offending_id
    assert err.missing_field == f"test_cases.{idx}.prompt"
    # The message is the raw Pydantic ``msg``; our ``TestCase``
    # validator raises ``TestCase.prompt must be a non-empty string``.
    assert "non-empty" in err.message


@given(suite=evaluation_suites(), data=st.data())
@settings(max_examples=20, deadline=None)
def test_empty_metrics_references_metrics_field(
    suite: EvaluationSuite, data: st.DataObject
) -> None:
    """**Validates: Requirements 3.3, 3.5**

    Setting ``metrics`` to an empty list produces a
    :class:`SuiteValidationError` whose ``missing_field`` dotted-path
    references the ``metrics`` list of the offending test case.
    """
    n = len(suite.test_cases)
    idx = data.draw(st.integers(min_value=0, max_value=n - 1))
    payload = _as_json_dict(suite)
    tc = cast(dict[str, Any], payload["test_cases"][idx])
    offending_id = cast(str, tc["id"])
    tc["metrics"] = []
    text = json.dumps(payload)

    with pytest.raises(SuiteValidationError) as excinfo:
        load_suite_from_string(text, "json")
    err = excinfo.value
    assert err.test_case_id == offending_id
    assert err.missing_field == f"test_cases.{idx}.metrics"


@given(
    suite=evaluation_suites().filter(lambda s: len(s.test_cases) >= 2),
    data=st.data(),
)
@settings(
    max_examples=20,
    deadline=None,
    # ``evaluation_suites`` generates 1..4 cases; the ``>= 2`` filter
    # rejects ~25–35% of draws, well under Hypothesis' default limit.
    # Suppress the ``filter_too_much`` health check to keep the test
    # stable if future generator tweaks shift the size distribution.
    suppress_health_check=[HealthCheck.filter_too_much],
)
def test_duplicate_test_case_id_raises_and_names_duplicate(
    suite: EvaluationSuite, data: st.DataObject
) -> None:
    """**Validates: Requirements 3.3, 3.5**

    Introducing a duplicate ``TestCase.id`` across two cases produces
    a :class:`SuiteValidationError` whose message contains the
    duplicated id (quoted, as ``'the-id'``, matching the ``!r`` format
    used by the model's ``_unique_test_case_ids`` validator).

    The duplication is performed by copying one case's id onto a
    different index, so the resulting document has exactly two cases
    sharing the same id and the rest unchanged.
    """
    n = len(suite.test_cases)
    # Draw two distinct indices so the mutation always introduces a
    # real duplicate regardless of the sampled positions.
    indices: list[int] = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=2,
            max_size=2,
            unique=True,
        )
    )
    src, dst = indices[0], indices[1]
    payload = _as_json_dict(suite)
    test_cases = cast(list[dict[str, Any]], payload["test_cases"])
    duplicated_id = cast(str, test_cases[src]["id"])
    test_cases[dst]["id"] = duplicated_id
    text = json.dumps(payload)

    with pytest.raises(SuiteValidationError) as excinfo:
        load_suite_from_string(text, "json")
    err = excinfo.value
    # The ``EvaluationSuite._unique_test_case_ids`` validator formats
    # the duplicate id with ``!r``; the loader then wraps that text
    # into ``err.message``. Assert the id is surfaced quoted.
    assert repr(duplicated_id) in err.message


@given(suite=evaluation_suites())
@settings(max_examples=20, deadline=None)
def test_missing_top_level_name_reports_name(suite: EvaluationSuite) -> None:
    """**Validates: Requirements 3.3, 3.5**

    Removing the top-level ``name`` field produces a
    :class:`SuiteValidationError` whose ``missing_field`` is
    ``"name"`` and whose ``test_case_id`` is ``None`` — because the
    failure is at the suite level, not a per-test-case violation.
    """
    payload = _as_json_dict(suite)
    del payload["name"]
    text = json.dumps(payload)

    with pytest.raises(SuiteValidationError) as excinfo:
        load_suite_from_string(text, "json")
    err = excinfo.value
    assert err.missing_field == "name"
    assert err.test_case_id is None
