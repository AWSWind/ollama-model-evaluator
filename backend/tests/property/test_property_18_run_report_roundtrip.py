"""Property 18: Run_Report round-trip.

For every valid :class:`~ollama_evaluator.models.RunReport` ``r``::

    RunReport.model_validate_json(r.model_dump_json()) == r

This is Requirement 8.5 from the Ollama Model Evaluator spec: Run_Reports
must survive serialisation to disk and the REST API without losing
information. The round-trip equality is load-bearing because:

* The ``History_Store`` (Requirement 8.1) reads reports from disk with
  ``model_validate_json`` and clients compare them to the in-memory
  instances produced during a Run.
* The REST API (Requirement 13) relies on pydantic's JSON round-trip
  for every ``GET /api/runs/{id}``.
* The error-summary rendering code (Requirement 11.3) iterates over a
  re-loaded report and must see identical structure.

The test here delegates generation to :mod:`tests.property.generators`,
which exercises all 5 ``RunReport.status`` literals, both
``ended_at=None`` and ``ended_at >= started_at`` branches, both missing
and populated performance fields, all 4 ``TestCaseResult.status``
values, and both ``MetricResult.error=None`` / ``error="..."`` paths
— everything listed in Task 2.5 of ``.kiro/specs/ollama-model-evaluator/tasks.md``.

``max_examples`` is set to ``100`` to meet the spec's testing-strategy
floor for property tests; ``deadline=None`` is needed because pydantic
model construction can exceed Hypothesis' default 200 ms deadline on
heavily populated drawings without indicating a real problem.
"""

from __future__ import annotations

from hypothesis import given, settings

from ollama_evaluator.models import RunReport

from .generators import run_reports


@given(report=run_reports())
@settings(max_examples=20, deadline=None)
def test_run_report_round_trips_through_json(report: RunReport) -> None:
    """**Validates: Requirement 8.5**

    ``RunReport.model_validate_json(r.model_dump_json()) == r`` for every
    valid ``r``.
    """
    rebuilt = RunReport.model_validate_json(report.model_dump_json())
    assert rebuilt == report
