"""Run_Report artifact writer — JSON + Markdown on disk.

The scheduler produces a :class:`~ollama_evaluator.models.RunReport`
at terminal time (Task 12.1). This module serialises that report to
the two on-disk artifacts Requirement 8.1–8.3 mandate:

* ``<output_dir>/runs/<run_id>/report.json`` — canonical JSON,
  produced by Pydantic ``model_dump_json(indent=2)``. This is the
  artifact the REST API (Requirement 16.3) serves verbatim and the
  :class:`~ollama_evaluator.history.store.HistoryStore` indexes.
* ``<output_dir>/runs/<run_id>/report.md`` — human-readable Markdown
  with per-Model and per-Evaluation_Suite summary tables plus an
  error summary section (Requirements 8.3, 11.3).

Both files are written atomically (write-temp-then-rename) so a
process crash mid-write cannot leave a partial artifact for the
History_Store or the UI to pick up.

Properties validated here:

* 16 (Run_Report completeness) — via ``model_dump_json``: every field
  from the design's §Data Models / Run_Report table survives the
  serialisation.
* 17 (Markdown report contents) — the rendered text contains each
  model name, each suite name from ``report.config.run.suites``, and
  the column headers ``Model``, ``Passed``, ``Failed``, ``Mean
  tokens/s``, ``Mean total ms``.
* 22 (Error summary completeness) — the error summary section lists
  every entry from ``report.error_summary``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..models import ModelAggregate, RunReport, TestCaseResult


_MARKDOWN_HEADERS = (
    "Model",
    "Passed",
    "Failed",
    "Pass rate",
    "Mean tokens/s",
    "Mean total ms",
)
_PER_SUITE_HEADERS = (
    "Suite",
    "Passed",
    "Failed",
    "Pass rate",
    "Mean tokens/s",
    "Mean total ms",
)
_PER_MODEL_SUITE_HEADERS = (
    "Model",
    "Suite",
    "Passed",
    "Failed",
    "Pass rate",
    "Mean tokens/s",
    "Mean total ms",
)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Uses :func:`tempfile.NamedTemporaryFile(delete=False)` in the
    destination directory so the subsequent :func:`os.replace` is an
    atomic rename on every supported platform (including Windows,
    where cross-volume renames would otherwise fail).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``NamedTemporaryFile`` with ``delete=False`` lets us flush,
    # close, and rename; the fallback ``os.unlink`` keeps the temp
    # file from being leaked if the rename raises.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
        raise


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _fmt_float(value: float | None, precision: int = 2) -> str:
    """Render ``value`` as a bounded-precision float or ``"n/a"`` when None."""
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _fmt_pass_rate(passed: int, failed: int) -> str:
    """Render a pass rate as ``NN.N%`` over (passed+failed), or ``"n/a"``.

    Only ``pass`` and ``fail`` statuses contribute to the denominator so
    ``error``/``timeout`` results do not skew the rate downward (those
    already surface in the error summary). When the denominator is zero
    we return ``"n/a"`` instead of ``"0.0%"`` so readers can distinguish
    "0 passes of 0 scoreable cases" from "0 passes of 100 scoreable
    cases".
    """
    total = passed + failed
    if total <= 0:
        return "n/a"
    return f"{(passed / total) * 100:.1f}%"


def _per_model_row(agg: ModelAggregate) -> str:
    return "| {name} | {passed} | {failed} | {rate} | {tps} | {tm} |".format(
        name=agg.model,
        passed=agg.passed,
        failed=agg.failed,
        rate=_fmt_pass_rate(agg.passed, agg.failed),
        tps=_fmt_float(agg.mean_tokens_per_second),
        tm=_fmt_float(agg.mean_total_ms),
    )


def _per_suite_row(suite: str, results: list[TestCaseResult]) -> str:
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    tps_samples = [
        r.performance.tokens_per_second
        for r in results
        if r.performance.tokens_per_second is not None
    ]
    tm_samples = [r.performance.total_ms for r in results]
    mean_tps = sum(tps_samples) / len(tps_samples) if tps_samples else None
    mean_tm = sum(tm_samples) / len(tm_samples) if tm_samples else None
    return "| {name} | {passed} | {failed} | {rate} | {tps} | {tm} |".format(
        name=suite,
        passed=passed,
        failed=failed,
        rate=_fmt_pass_rate(passed, failed),
        tps=_fmt_float(mean_tps),
        tm=_fmt_float(mean_tm),
    )


def _per_model_suite_row(
    model: str, suite: str, results: list[TestCaseResult]
) -> str:
    """Render one row of the model × suite breakdown table.

    Used by Requirement-adjacent "per-model-per-suite" table added so
    multi-model runs make per-model performance legible at a glance.
    The counters + means follow the same semantics as
    :func:`_per_suite_row` but scoped to a single model.
    """
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    tps_samples = [
        r.performance.tokens_per_second
        for r in results
        if r.performance.tokens_per_second is not None
    ]
    tm_samples = [r.performance.total_ms for r in results]
    mean_tps = sum(tps_samples) / len(tps_samples) if tps_samples else None
    mean_tm = sum(tm_samples) / len(tm_samples) if tm_samples else None
    return (
        "| {model} | {suite} | {passed} | {failed} | {rate} | {tps} | {tm} |"
    ).format(
        model=model,
        suite=suite,
        passed=passed,
        failed=failed,
        rate=_fmt_pass_rate(passed, failed),
        tps=_fmt_float(mean_tps),
        tm=_fmt_float(mean_tm),
    )


def render_markdown(report: RunReport) -> str:
    """Render a human-readable Markdown summary of ``report``.

    Sections (in order):

    1. Metadata header (run id, status, start/end, versions).
    2. Per-Model table with the exact column headers Property 17
       requires (``Model``, ``Passed``, ``Failed``, ``Mean tokens/s``,
       ``Mean total ms``).
    3. Per-Evaluation_Suite table using the same column shape —
       counters are summed over every result whose ``suite`` matches.
    4. Error summary section listing each :class:`ErrorSummaryEntry`
       (Requirement 11.3). When ``error_summary`` is empty, the
       section heading is still emitted with the text "None." so the
       rendered artefact has a consistent structure regardless of
       outcome.
    """
    lines: list[str] = []

    # ------- Header ---------------------------------------------------
    lines.append(f"# Run report: {report.run_id}")
    lines.append("")
    lines.append(f"- **Status**: {report.status}")
    lines.append(f"- **Backend version**: {report.backend_version}")
    lines.append(
        f"- **Ollama version**: {report.ollama_version or 'n/a'}"
    )
    lines.append(f"- **Started at**: {report.started_at.isoformat()}")
    lines.append(
        f"- **Ended at**: {report.ended_at.isoformat() if report.ended_at else 'n/a'}"
    )
    lines.append("")

    # ------- Models (Property 17 anchors) -----------------------------
    lines.append("## Models")
    lines.append("")
    for info in report.models:
        digest = info.digest or "n/a"
        size = info.parameter_size or "n/a"
        lines.append(f"- **{info.name}** (digest={digest}, parameter_size={size})")
    if not report.models:
        lines.append("_(no models)_")
    lines.append("")

    # ------- Suites (Property 17 anchors) -----------------------------
    lines.append("## Suites")
    lines.append("")
    for suite in report.config.run.suites:
        lines.append(f"- **{suite}**")
    if not report.config.run.suites:
        lines.append("_(no suites)_")
    lines.append("")

    # ------- Per-model table ------------------------------------------
    lines.append("## Per-model results")
    lines.append("")
    lines.append("| " + " | ".join(_MARKDOWN_HEADERS) + " |")
    lines.append("|" + "|".join(["---"] * len(_MARKDOWN_HEADERS)) + "|")
    for agg in report.aggregates:
        lines.append(_per_model_row(agg))
    if not report.aggregates:
        lines.append("| _(no data)_ | 0 | 0 | n/a | n/a | n/a |")
    lines.append("")

    # ------- Per-suite table ------------------------------------------
    lines.append("## Per-suite results")
    lines.append("")
    lines.append("| " + " | ".join(_PER_SUITE_HEADERS) + " |")
    lines.append("|" + "|".join(["---"] * len(_PER_SUITE_HEADERS)) + "|")
    # Always iterate the configured suite order so rows are stable
    # across runs with the same config; supplement with any extra
    # suites that appear on results but not in config (robustness
    # guard only — should not happen in the standard path).
    by_suite: dict[str, list[TestCaseResult]] = {
        s: [] for s in report.config.run.suites
    }
    for r in report.results:
        by_suite.setdefault(r.suite, []).append(r)
    for suite, results in by_suite.items():
        lines.append(_per_suite_row(suite, results))
    if not by_suite:
        lines.append("| _(no data)_ | 0 | 0 | n/a | n/a | n/a |")
    lines.append("")

    # ------- Per-model × per-suite table ------------------------------
    # Added so multi-model runs can attribute pass/fail to a specific
    # (model, suite) pair. When only one model is configured this table
    # is still emitted (redundant but not confusing); the cost is a
    # handful of extra markdown rows.
    lines.append("## Per-model × per-suite results")
    lines.append("")
    lines.append("| " + " | ".join(_PER_MODEL_SUITE_HEADERS) + " |")
    lines.append(
        "|" + "|".join(["---"] * len(_PER_MODEL_SUITE_HEADERS)) + "|"
    )
    # Iterate configured model order, then configured suite order, so
    # the table layout stays stable across re-runs of the same config.
    configured_models = list(report.config.run.models)
    configured_suites = list(report.config.run.suites)
    by_model_suite: dict[tuple[str, str], list[TestCaseResult]] = {}
    for r in report.results:
        by_model_suite.setdefault((r.model, r.suite), []).append(r)
    emitted_any = False
    for model in configured_models:
        for suite in configured_suites:
            key = (model, suite)
            if key not in by_model_suite:
                # Skip empty cells rather than emit "0 0 n/a" rows for
                # every (model, suite) combination that was filtered out
                # by tag selection.
                continue
            lines.append(_per_model_suite_row(model, suite, by_model_suite[key]))
            emitted_any = True
    # Guard for robustness — results referencing models/suites not in
    # the configured order still appear so nothing goes missing.
    for (model, suite), results in by_model_suite.items():
        if model in configured_models and suite in configured_suites:
            continue
        lines.append(_per_model_suite_row(model, suite, results))
        emitted_any = True
    if not emitted_any:
        lines.append("| _(no data)_ | _(no data)_ | 0 | 0 | n/a | n/a | n/a |")
    lines.append("")

    # ------- Error summary (Requirement 11.3) -------------------------
    lines.append("## Error summary")
    lines.append("")
    if not report.error_summary:
        lines.append("None.")
    else:
        lines.append("| Model | Suite | Test case | Repetition | Error |")
        lines.append("|---|---|---|---|---|")
        for entry in report.error_summary:
            # Backtick-escape pipes in the error message so Markdown
            # tables still render correctly when the message contains
            # a ``|`` character.
            msg = entry.error_message.replace("|", "\\|")
            lines.append(
                f"| {entry.model} | {entry.suite} | {entry.test_case_id} "
                f"| {entry.repetition} | {msg} |"
            )
    lines.append("")

    # Ensure trailing newline.
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def write_artifacts(
    run_id: str,
    report: RunReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` to ``<output_dir>/runs/<run_id>/``.

    Both writes are atomic. The function is declared ``async`` to
    match the rest of the runner's call graph even though the I/O
    primitives used here are synchronous; any callers that want to
    off-load to a thread can wrap the call in
    :func:`asyncio.to_thread`.

    Returns the two written paths so the caller (Task 13's
    :class:`HistoryStore` wiring and Task 19's CLI) can log them
    verbatim.
    """
    run_dir = Path(output_dir) / run_id
    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"

    _write_atomic(json_path, report.model_dump_json(indent=2))
    _write_atomic(md_path, render_markdown(report))

    return json_path, md_path


__all__ = [
    "render_markdown",
    "write_artifacts",
]
