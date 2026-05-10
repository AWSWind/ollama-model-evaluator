"""Property 1: Evaluation_Suite round-trip.

For every valid :class:`~ollama_evaluator.suites.models.EvaluationSuite`
``s`` and every ``fmt ∈ {"yaml", "json"}``::

    load_suite_from_string(dump_suite(s, fmt), fmt) == s

The round-trip invariant is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness Properties
as Property 1 and is directly driven by:

* **Requirement 3.2** — suites are persisted as YAML or JSON files, so
  the serialiser must emit a document the loader can reparse losslessly.
* **Requirement 3.4** — ``extra="forbid"`` forbids silent loss of
  authored fields; a round-trip that dropped any field would regress
  this guarantee.
* **Requirement 4.1** — YAML and JSON are both supported first-class
  authoring formats, which is why the property is parametrised over
  ``fmt``.
* **Requirement 4.2** — the writer's canonicalisation rules (sorted
  keys, 2-space indent, block-style YAML, no comments) exist so the
  loader, running in ``ruamel.yaml(typ="rt")`` mode, always sees a
  document it can reconstruct into the original model.
* **Requirement 4.3** — round-trip equivalence is specified at the
  *Pydantic model* level, not byte-for-byte, which is what this test
  asserts.

The Hypothesis strategy in :mod:`tests.property.generators.evaluation_suites`
is constrained to ASCII identifiers (``_SIMPLE_ALPHABET``) and finite
floats. Per the Task 3.3 constraints this is acceptable: YAML and JSON
*string escaping* is tested by the upstream ``ruamel.yaml`` and stdlib
``json`` projects, whereas Property 1 is about suite-level round-trip
fidelity. The strategy still exercises:

* All optional fields on :class:`~ollama_evaluator.suites.models.TestCase`
  via ``one_of(none(), ...)`` arms.
* The three-way distinction on ``stop_sequences`` — ``None`` (inherit),
  ``[]`` (clear), and non-empty list (override).
* Nested ``reference_data`` dicts up to two levels deep.
* Varying :class:`~ollama_evaluator.suites.models.GenerationDefaults`
  so the suite-level defaults branch is covered.
* 0..3 metric-specific ``params`` per ``MetricConfig``.

``max_examples=20`` matches the floor set by the testing strategy in
``design.md``; ``deadline=None`` avoids false positives from
``ruamel.yaml``'s slightly variable dump-time under load.
"""

from __future__ import annotations

from typing import Literal

import pytest
from hypothesis import given, settings

from ollama_evaluator.suites import dump_suite, load_suite_from_string
from ollama_evaluator.suites.models import EvaluationSuite

from .generators import evaluation_suites


@pytest.mark.parametrize("fmt", ["yaml", "json"])
@given(suite=evaluation_suites())
@settings(max_examples=20, deadline=None)
def test_evaluation_suite_round_trips_through_dump_and_load(
    suite: EvaluationSuite, fmt: Literal["yaml", "json"]
) -> None:
    """**Validates: Requirements 3.2, 3.4, 4.1, 4.2, 4.3**

    ``load_suite_from_string(dump_suite(s, fmt), fmt) == s`` for every
    valid ``s`` drawn from :func:`evaluation_suites` and every
    ``fmt ∈ {"yaml", "json"}``.
    """
    serialised = dump_suite(suite, fmt)
    rebuilt = load_suite_from_string(serialised, fmt)
    assert rebuilt == suite
