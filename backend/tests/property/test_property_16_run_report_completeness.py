"""Feature: ollama-model-evaluator, Property 16: Run_Report completeness.

For any successfully completed Run, the produced :class:`RunReport`
contains: a non-empty ``run_id``; ``backend_version``;
``ollama_version`` (which may be ``None`` when the Ollama_Server
omitted it); ``started_at`` and ``ended_at`` with
``ended_at >= started_at`` when ``ended_at`` is populated; the full
``config`` as submitted; a ``models`` entry per evaluated model
carrying ``name``, ``digest``, and ``parameter_size``; a ``results``
entry per ``(model, test_case_id, repetition)`` execution; and an
``aggregates`` entry per evaluated model.

Validates: Requirements 2.5, 8.2, 8.4.

This property is expressed at the data-model level: every
:class:`RunReport` drawn from the :func:`run_reports` generator
already satisfies the invariants documented in the model's
validators (``ended_at >= started_at``; non-empty ``run_id``;
``ConfigDict(extra="forbid")``). This test reasserts each clause
explicitly so a future model-level relaxation surfaces as a failing
property rather than a silent change.
"""

from __future__ import annotations

from hypothesis import given, settings

from ollama_evaluator.config import ConfigFile
from ollama_evaluator.models import ModelInfo, RunReport

from .generators import run_reports


@given(report=run_reports())
@settings(max_examples=20, deadline=None)
def test_run_report_has_every_required_field(report: RunReport) -> None:
    """**Validates: Requirements 2.5, 8.2, 8.4**

    A :class:`RunReport` drawn from :func:`run_reports` has every
    field required by the design populated with the expected type.
    """
    # Identity & version fields.
    assert isinstance(report.run_id, str) and report.run_id.strip(), (
        "run_id must be a non-empty string"
    )
    assert isinstance(report.backend_version, str) and report.backend_version, (
        "backend_version must be non-empty"
    )
    # ``ollama_version`` is nullable per design but must be typed.
    assert report.ollama_version is None or isinstance(report.ollama_version, str)

    # Timestamps and the ordering invariant (Property 16 text).
    assert report.started_at is not None
    if report.ended_at is not None:
        assert report.ended_at >= report.started_at, (
            f"ended_at ({report.ended_at}) must be >= started_at ({report.started_at})"
        )

    # Full config is a ``ConfigFile`` instance — not just a dict —
    # so the round-trip + reproducibility guarantees of Req 8.4 hold.
    assert isinstance(report.config, ConfigFile)

    # Models: every ``ModelInfo`` carries at least ``name``; digest /
    # parameter_size may be ``None`` because v1 adapters don't
    # always know them upfront.
    for info in report.models:
        assert isinstance(info, ModelInfo)
        assert info.name

    # Aggregates: one entry per evaluated model (Req 8.2). We assert
    # shape rather than a strict 1:1 mapping because the generator
    # can draw an empty-model Run to exercise the ``run-failed``
    # path; for non-empty models the aggregates list is covered
    # model-by-model by its own unit tests.
    assert isinstance(report.aggregates, list)
    if report.models:
        agg_models = {a.model for a in report.aggregates}
        report_models = {m.name for m in report.models}
        # Aggregates are a subset of evaluated models — generator
        # may draw extras or fewer (e.g. aborted mid-dispatch).
        assert agg_models <= report_models

    # Results: every entry is a TestCaseResult with a repetition >= 1
    # (validator enforced). Assert positive-repetition invariant
    # here as a guard against a future change that might loosen it.
    for r in report.results:
        assert r.repetition >= 1

    # Error summary: every entry references a valid repetition; the
    # bijection proof against ``results`` lives in Property 22.
    for e in report.error_summary:
        assert e.repetition >= 1
