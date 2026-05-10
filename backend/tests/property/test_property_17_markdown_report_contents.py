"""Feature: ollama-model-evaluator, Property 17: Markdown report contents.

For any :class:`RunReport` ``r``, the Markdown rendering of ``r``
contains each model name in ``r.models``, each suite name in
``r.config.run.suites``, and the column headers ``Model``,
``Passed``, ``Failed``, ``Mean tokens/s``, and ``Mean total ms``.

Validates: Requirement 8.3.
"""

from __future__ import annotations

from hypothesis import given, settings

from ollama_evaluator.models import RunReport
from ollama_evaluator.runner.reports import render_markdown

from .generators import run_reports


_REQUIRED_HEADERS = ("Model", "Passed", "Failed", "Mean tokens/s", "Mean total ms")


@given(report=run_reports())
@settings(max_examples=20, deadline=None)
def test_markdown_contains_models_suites_and_headers(report: RunReport) -> None:
    """**Validates: Requirement 8.3**

    Rendered Markdown contains every model name, every configured
    suite name, and the five required column headers.
    """
    md = render_markdown(report)

    for info in report.models:
        assert info.name in md, f"model {info.name!r} missing from rendered Markdown"

    for suite in report.config.run.suites:
        assert suite in md, f"suite {suite!r} missing from rendered Markdown"

    for header in _REQUIRED_HEADERS:
        assert header in md, f"header {header!r} missing from rendered Markdown"
