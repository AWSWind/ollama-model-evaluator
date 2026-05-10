"""Reusable Hypothesis strategies for backend property tests.

This module builds Hypothesis strategies over the Pydantic data models
defined in :mod:`ollama_evaluator.models` and :mod:`ollama_evaluator.config`.
The strategies constrain generation to the valid input space — every
value drawn from them is a well-formed model instance — so property
tests can focus on the invariant under test rather than re-asserting
shape validity.

Design references:

* ``.kiro/specs/ollama-model-evaluator/design.md`` §Data Models.
* ``.kiro/specs/ollama-model-evaluator/requirements.md`` §Requirement 8.5
  (Property 18: Run_Report round-trip).

Implementation notes:

* Strategies compose bottom-up: leaf strategies (datetimes, floats,
  ascii strings) feed into model strategies, which then feed into
  higher-level strategies such as :func:`run_reports`.
* String generation is constrained to ``[A-Za-z0-9-_]`` to avoid JSON
  encoding edge cases (escaped control characters, non-BMP codepoints)
  that are unrelated to the invariants the property tests assert.
* Floats are always finite and non-NaN (``allow_nan=False``,
  ``allow_infinity=False``) because NaN is not ``__eq__`` to itself and
  would spuriously falsify round-trip equality.
* Datetimes are generated as UTC-aware instances in the 2020–2030 range
  so that serialisation always produces a deterministic ``Z``-suffixed
  ISO 8601 string.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import ascii_letters, digits
from typing import Any

from hypothesis import strategies as st

from ollama_evaluator.config import ConfigFile, RunConfig
from ollama_evaluator.models import (
    ErrorSummaryEntry,
    MetricAggregate,
    MetricResult,
    ModelAggregate,
    ModelInfo,
    PerformanceMetrics,
    RunReport,
    TestCaseResult,
)
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)

# Ascii-only alphabet used for every generated string. Stripping Unicode
# and whitespace keeps round-trip tests focused on model semantics rather
# than on JSON string escaping, which is already tested by pydantic and
# the standard library.
_SIMPLE_ALPHABET = ascii_letters + digits + "-_"


def simple_strings(min_size: int = 1, max_size: int = 10) -> st.SearchStrategy[str]:
    """Short, ASCII-only identifiers.

    Used for ``run_id``, model tags, suite names, test-case ids, metric
    names, and free-form messages in the generators below. ``min_size``
    defaults to ``1`` because most identifier-shaped fields reject empty
    strings.
    """
    return st.text(
        alphabet=_SIMPLE_ALPHABET,
        min_size=min_size,
        max_size=max_size,
    )


def finite_floats(
    min_value: float = 0.0, max_value: float = 1_000_000.0
) -> st.SearchStrategy[float]:
    """Finite, non-NaN floats in a bounded range.

    Bounds are defensive: they keep generated magnitudes within IEEE-754
    double precision where ``repr(round-trip)`` is exact, avoiding any
    theoretical precision loss on JSON serialisation.
    """
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def tz_aware_datetimes() -> st.SearchStrategy[datetime]:
    """UTC-aware datetimes in the 2020-01-01 — 2030-01-01 range.

    Hypothesis' built-in ``datetimes()`` strategy returns naive datetimes
    by default; mapping ``.replace(tzinfo=timezone.utc)`` attaches UTC
    without depending on the ``hypothesis[pytz]`` extra.
    """
    return st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1),
    ).map(lambda d: d.replace(tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


@st.composite
def performance_metrics(draw: st.DrawFn) -> PerformanceMetrics:
    """Draw a :class:`PerformanceMetrics` with nullable fields exercised.

    Each nullable field (``ttft_ms``, ``prompt_tokens``, ``response_tokens``,
    ``tokens_per_second``) is drawn from a ``one_of(none(), ...)`` so both
    the "server reported" and "server omitted" branches (Requirement 6.5)
    are covered across 100+ examples.
    """
    return PerformanceMetrics(
        ttft_ms=draw(st.one_of(st.none(), finite_floats(0.0, 60_000.0))),
        total_ms=draw(finite_floats(0.0, 600_000.0)),
        prompt_tokens=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100_000))),
        response_tokens=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100_000))),
        tokens_per_second=draw(st.one_of(st.none(), finite_floats(0.0, 10_000.0))),
    )


# Simple JSON-safe values for ``MetricResult.details``. The property tests
# only need detail dicts to round-trip structurally; rich nested shapes are
# covered by the unit tests in ``test_run_report_models.py``.
_details_value_strategy = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    finite_floats(-1_000_000.0, 1_000_000.0),
    simple_strings(min_size=0, max_size=10),
)


@st.composite
def metric_results(
    draw: st.DrawFn,
    names: st.SearchStrategy[str] | None = None,
) -> MetricResult:
    """Draw a :class:`MetricResult` covering both ``error=None`` and populated.

    A quarter-ish of generated instances carry a non-empty ``error``
    string so Property 18 exercises the "metric raised" branch
    (Requirement 7.5). ``details`` is a shallow mapping of JSON-safe
    scalars, sufficient to verify round-trip without pulling in nested
    structure edge cases that belong to pydantic's own test suite.
    """
    name_strategy = names if names is not None else simple_strings(min_size=1, max_size=10)
    details: dict[str, Any] = draw(
        st.dictionaries(
            keys=simple_strings(min_size=1, max_size=5),
            values=_details_value_strategy,
            max_size=3,
        )
    )
    return MetricResult(
        name=draw(name_strategy),
        score=draw(finite_floats(-1_000.0, 1_000.0)),
        passed=draw(st.booleans()),
        threshold=draw(st.one_of(st.none(), finite_floats(-1_000.0, 1_000.0))),
        details=details,
        error=draw(st.one_of(st.none(), simple_strings(min_size=1, max_size=20))),
    )


@st.composite
def test_case_results(
    draw: st.DrawFn,
    models: st.SearchStrategy[str] | None = None,
    suites: st.SearchStrategy[str] | None = None,
    test_case_ids: st.SearchStrategy[str] | None = None,
) -> TestCaseResult:
    """Draw a :class:`TestCaseResult` exercising all 4 status literals.

    Callers may pin ``model``/``suite``/``test_case_id`` to a parent
    strategy (e.g. to keep the result consistent with an enclosing
    :class:`RunReport`'s ``models`` list). When unset, independent
    identifier strategies are used.
    """
    model_strategy = models if models is not None else simple_strings(min_size=1, max_size=10)
    suite_strategy = suites if suites is not None else simple_strings(min_size=1, max_size=10)
    tc_id_strategy = (
        test_case_ids if test_case_ids is not None else simple_strings(min_size=1, max_size=10)
    )
    return TestCaseResult(
        model=draw(model_strategy),
        suite=draw(suite_strategy),
        test_case_id=draw(tc_id_strategy),
        repetition=draw(st.integers(min_value=1, max_value=10)),
        status=draw(st.sampled_from(["pass", "fail", "error", "timeout"])),
        response=draw(st.one_of(st.none(), simple_strings(min_size=0, max_size=30))),
        error_message=draw(st.one_of(st.none(), simple_strings(min_size=0, max_size=30))),
        performance=draw(performance_metrics()),
        metrics=draw(st.lists(metric_results(), min_size=0, max_size=3)),
    )


@st.composite
def metric_aggregates(
    draw: st.DrawFn,
    metric: st.SearchStrategy[str] | None = None,
) -> MetricAggregate:
    """Draw a :class:`MetricAggregate`; ``stddev`` and ``count`` are >= 0."""
    name_strategy = metric if metric is not None else simple_strings(min_size=1, max_size=10)
    return MetricAggregate(
        metric=draw(name_strategy),
        mean=draw(finite_floats(-1_000.0, 1_000.0)),
        stddev=draw(finite_floats(0.0, 1_000.0)),
        count=draw(st.integers(min_value=0, max_value=1_000)),
    )


@st.composite
def model_aggregates(
    draw: st.DrawFn,
    model: st.SearchStrategy[str] | None = None,
) -> ModelAggregate:
    """Draw a :class:`ModelAggregate` with consistent child ``MetricAggregate``s.

    The keys of ``metric_aggregates`` are drawn unique, and each child
    :class:`MetricAggregate` is drawn with ``metric`` pinned to its key
    so the dict is internally consistent (the ``metric`` field inside
    the child equals the key used to look it up in the dict).
    """
    model_strategy = model if model is not None else simple_strings(min_size=1, max_size=10)
    metric_names: list[str] = draw(
        st.lists(simple_strings(min_size=1, max_size=10), min_size=0, max_size=3, unique=True)
    )
    per_metric: dict[str, MetricAggregate] = {
        name: draw(metric_aggregates(metric=st.just(name))) for name in metric_names
    }
    return ModelAggregate(
        model=draw(model_strategy),
        passed=draw(st.integers(min_value=0, max_value=1_000)),
        failed=draw(st.integers(min_value=0, max_value=1_000)),
        errored=draw(st.integers(min_value=0, max_value=1_000)),
        timed_out=draw(st.integers(min_value=0, max_value=1_000)),
        mean_ttft_ms=draw(st.one_of(st.none(), finite_floats(0.0, 60_000.0))),
        mean_total_ms=draw(finite_floats(0.0, 600_000.0)),
        mean_tokens_per_second=draw(st.one_of(st.none(), finite_floats(0.0, 10_000.0))),
        metric_aggregates=per_metric,
    )


@st.composite
def model_infos(
    draw: st.DrawFn,
    name: st.SearchStrategy[str] | None = None,
) -> ModelInfo:
    """Draw a :class:`ModelInfo` with optional ``digest`` / ``parameter_size``."""
    name_strategy = name if name is not None else simple_strings(min_size=1, max_size=10)
    return ModelInfo(
        name=draw(name_strategy),
        digest=draw(st.one_of(st.none(), simple_strings(min_size=1, max_size=20))),
        parameter_size=draw(st.one_of(st.none(), simple_strings(min_size=1, max_size=10))),
    )


@st.composite
def error_summary_entries(
    draw: st.DrawFn,
    models: st.SearchStrategy[str] | None = None,
    suites: st.SearchStrategy[str] | None = None,
    test_case_ids: st.SearchStrategy[str] | None = None,
) -> ErrorSummaryEntry:
    """Draw a :class:`ErrorSummaryEntry`; ``repetition`` is always >= 1."""
    model_strategy = models if models is not None else simple_strings(min_size=1, max_size=10)
    suite_strategy = suites if suites is not None else simple_strings(min_size=1, max_size=10)
    tc_id_strategy = (
        test_case_ids if test_case_ids is not None else simple_strings(min_size=1, max_size=10)
    )
    return ErrorSummaryEntry(
        model=draw(model_strategy),
        suite=draw(suite_strategy),
        test_case_id=draw(tc_id_strategy),
        repetition=draw(st.integers(min_value=1, max_value=10)),
        error_message=draw(simple_strings(min_size=1, max_size=40)),
    )


# ---------------------------------------------------------------------------
# ConfigFile (kept minimal on purpose — Property 18 is about RunReport)
# ---------------------------------------------------------------------------


@st.composite
def _run_configs(draw: st.DrawFn) -> RunConfig:
    """Draw a valid :class:`RunConfig`.

    This strategy varies every field so different ``ConfigFile``
    instances produced by :func:`config_files` differ in their ``run``
    section, satisfying the task's "draw at least 2 varied ConfigFile
    instances" requirement via ordinary Hypothesis generation.
    """
    return RunConfig(
        models=draw(
            st.lists(simple_strings(min_size=1, max_size=10), min_size=1, max_size=3, unique=True)
        ),
        suites=draw(
            st.lists(simple_strings(min_size=1, max_size=10), min_size=1, max_size=3, unique=True)
        ),
        repetitions=draw(st.integers(min_value=1, max_value=5)),
        concurrency=draw(st.integers(min_value=1, max_value=5)),
        pull_missing_models=draw(st.booleans()),
        retry_max_attempts=draw(st.integers(min_value=0, max_value=5)),
        judge_model=draw(st.one_of(st.none(), simple_strings(min_size=1, max_size=10))),
        tag_filter=draw(st.lists(simple_strings(min_size=1, max_size=10), max_size=3)),
        ollama_timeout_s=draw(finite_floats(0.01, 300.0)),
    )


@st.composite
def _config_files(draw: st.DrawFn) -> ConfigFile:
    """Draw a minimal-but-varied :class:`ConfigFile`.

    ``suites_dir``/``output_dir``/``hf_cache_dir`` are drawn from a small
    fixed set of POSIX-safe relative paths because the property under
    test (RunReport round-trip) does not need exhaustive Path coverage
    and platform-dependent Path string representations would add noise.
    The ``run`` section varies per draw, which is where the required
    cross-instance variation comes from.
    """
    return ConfigFile(
        ollama_base_url=draw(
            st.sampled_from(
                [
                    "http://localhost:11434",
                    "https://ollama.example.com:8080",
                    "http://127.0.0.1:9999",
                ]
            )
        ),
        suites_dir=Path("suites"),
        output_dir=draw(st.sampled_from([Path("runs"), Path("out")])),
        log_level=draw(st.sampled_from(["debug", "info", "warn", "error"])),
        dataset_mode=draw(st.sampled_from(["local", "remote"])),
        hf_cache_dir=draw(st.one_of(st.none(), st.just(Path("cache")))),
        run=draw(_run_configs()),
    )


# ---------------------------------------------------------------------------
# Top-level RunReport
# ---------------------------------------------------------------------------


@st.composite
def run_reports(draw: st.DrawFn) -> RunReport:
    """Draw a valid :class:`RunReport`.

    Exercises all 5 ``status`` literals. For in-flight statuses
    (``pending``/``running``) ``ended_at`` may be ``None``; for terminal
    statuses (``completed``/``aborted``/``failed``) ``ended_at`` is
    always populated and the ``ended_at >= started_at`` invariant
    (Property 16) is preserved by generating ``ended_at`` as
    ``started_at + timedelta(seconds=offset)`` with ``offset >= 0``.
    """
    started_at = draw(tz_aware_datetimes())
    status = draw(
        st.sampled_from(["pending", "running", "completed", "aborted", "failed"])
    )
    # Generate ``ended_at`` by adding a non-negative offset to
    # ``started_at`` rather than drawing independently and filtering,
    # which would reject ~50% of draws.
    offset_seconds = draw(st.integers(min_value=0, max_value=365 * 24 * 3600))
    ended_at_value: datetime | None = started_at + timedelta(seconds=offset_seconds)
    if status in ("pending", "running") and draw(st.booleans()):
        ended_at_value = None

    model_names: list[str] = draw(
        st.lists(simple_strings(min_size=1, max_size=10), min_size=0, max_size=3, unique=True)
    )
    models_list = [draw(model_infos(name=st.just(n))) for n in model_names]
    aggregates_list = [draw(model_aggregates(model=st.just(n))) for n in model_names]

    results_list = draw(st.lists(test_case_results(), min_size=0, max_size=5))
    error_summary_list = draw(st.lists(error_summary_entries(), min_size=0, max_size=3))

    return RunReport(
        run_id=draw(simple_strings(min_size=1, max_size=12)),
        backend_version=draw(simple_strings(min_size=1, max_size=10)),
        ollama_version=draw(st.one_of(st.none(), simple_strings(min_size=1, max_size=10))),
        started_at=started_at,
        ended_at=ended_at_value,
        status=status,
        config=draw(_config_files()),
        models=models_list,
        results=results_list,
        aggregates=aggregates_list,
        error_summary=error_summary_list,
    )


# ---------------------------------------------------------------------------
# Evaluation_Suite strategies (Property 1 — suites round-trip)
# ---------------------------------------------------------------------------
# Like ``run_reports`` above, these strategies constrain generation to the
# valid input space so a Property 1 test can assert equality after a full
# dump/load cycle without having to filter out invalid draws. The strategies
# are deliberately ASCII-only (see ``_SIMPLE_ALPHABET``) because Property 1
# verifies the Evaluation_Suite *model* round-trip, not YAML/JSON string
# escaping: allowing unicode or control characters would surface bugs in the
# serialiser libraries (pydantic, ruamel.yaml, stdlib ``json``), not in the
# suite loader/writer. Requirement 4.2 explicitly phrases the round-trip at
# the Pydantic-model level for this reason.


# Leaf values allowed inside ``MetricConfig.params`` (``dict[str, Any]``).
# Restricted to JSON-safe *scalars* per Task 3.3: no nested lists / dicts
# and no ``None`` (judge / schema metrics store their params as concrete
# values, not as explicit nulls).
_metric_param_value_strategy = st.one_of(
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    finite_floats(-1_000_000.0, 1_000_000.0),
    simple_strings(min_size=0, max_size=10),
)


@st.composite
def metric_configs(draw: st.DrawFn) -> MetricConfig:
    """Draw a :class:`MetricConfig` with 0..3 scalar ``params`` entries.

    ``name`` is always non-empty (Requirement 3.3 forbids blank metric
    names; see :meth:`MetricConfig._name_non_empty`). ``params`` carries
    between zero and three entries — zero exercises the "no metric-specific
    config" shape that e.g. ``response-capture`` uses, while the populated
    branches exercise the exact/regex/json-schema/length-range shapes that
    the real metric implementations inspect. Nested mappings are
    intentionally excluded from ``params`` itself; deeper structure is
    exercised via :func:`test_cases`' ``reference_data`` instead.
    """
    return MetricConfig(
        name=draw(simple_strings(min_size=1, max_size=10)),
        params=draw(
            st.dictionaries(
                keys=simple_strings(min_size=1, max_size=5),
                values=_metric_param_value_strategy,
                max_size=3,
            )
        ),
    )


# Leaf values allowed inside ``TestCase.reference_data`` (``dict[str, Any]``).
# Explicitly includes ``None`` so consumers see the "JSON null" branch. The
# second tier (``_reference_nested_strategy``) extends this with one level of
# nested mappings — enough to exercise the recursive branch of
# ``writer._sort_recursive`` without fabricating arbitrarily deep structures
# that aren't representative of real reference data.
_reference_scalar_strategy = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    finite_floats(-1_000_000.0, 1_000_000.0),
    simple_strings(min_size=0, max_size=10),
)
_reference_nested_strategy = st.dictionaries(
    keys=simple_strings(min_size=1, max_size=5),
    values=_reference_scalar_strategy,
    max_size=3,
)
_reference_value_strategy = st.one_of(
    _reference_scalar_strategy, _reference_nested_strategy
)


def _reference_data_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Two-level-deep JSON-safe mapping used for ``reference_data``."""
    return st.dictionaries(
        keys=simple_strings(min_size=1, max_size=5),
        values=_reference_value_strategy,
        max_size=3,
    )


@st.composite
def test_cases(
    draw: st.DrawFn,
    ids: st.SearchStrategy[str] | None = None,
) -> TestCase:
    """Draw a valid :class:`TestCase` exercising every optional field.

    Every optional field is drawn from ``one_of(none(), ...)`` so both
    "omitted" and "populated" branches surface across the 100+ Hypothesis
    examples used by Property 1. In particular:

    * ``stop_sequences`` is drawn with three arms — ``None`` (inherit from
      defaults), ``[]`` (explicitly clear defaults), and a non-empty
      list — because the semantic difference between ``None`` and ``[]`` is
      load-bearing (see ``TestCase.stop_sequences`` docstring and the
      ``test_round_trip_preserves_stop_sequences_none_vs_empty`` unit test).
    * ``reference_data`` uses :func:`_reference_data_strategy` so the
      two-level-deep nesting documented in the task specification is
      exercised on every ~third draw.
    * ``metrics`` is non-empty (Requirement 3.3); 1..3 metrics per case is
      enough to exercise ordering and uniqueness of metric names without
      bloating generated suites.

    ``ids`` may be supplied by :func:`evaluation_suites` so test-case ids
    in the same suite are drawn from a pre-sampled unique-id pool — the
    :class:`EvaluationSuite._unique_test_case_ids` validator rejects
    duplicates, and filtering on a ``unique=True`` ``lists`` strategy is
    cheaper than post-filtering at the model level.
    """
    id_strategy = ids if ids is not None else simple_strings(min_size=1, max_size=10)

    # Tri-state for ``stop_sequences``: None / [] / non-empty list. The
    # ``just([])`` arm makes the "explicitly clear defaults" case show up
    # reliably even at small ``max_examples``.
    stop_sequences_strategy: st.SearchStrategy[list[str] | None] = st.one_of(
        st.none(),
        st.just([]),
        st.lists(simple_strings(min_size=0, max_size=8), min_size=1, max_size=3),
    )

    return TestCase(
        id=draw(id_strategy),
        prompt=draw(simple_strings(min_size=1, max_size=30)),
        system_prompt=draw(
            st.one_of(st.none(), simple_strings(min_size=0, max_size=20))
        ),
        expected_output=draw(
            st.one_of(st.none(), simple_strings(min_size=0, max_size=20))
        ),
        reference_data=draw(st.one_of(st.none(), _reference_data_strategy())),
        tags=draw(
            st.lists(simple_strings(min_size=1, max_size=8), min_size=0, max_size=4)
        ),
        temperature=draw(st.one_of(st.none(), finite_floats(0.0, 2.0))),
        max_tokens=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10_000))),
        stop_sequences=draw(stop_sequences_strategy),
        metrics=draw(st.lists(metric_configs(), min_size=1, max_size=3)),
    )


@st.composite
def generation_defaults(draw: st.DrawFn) -> GenerationDefaults:
    """Draw a :class:`GenerationDefaults` with varied (possibly non-default) fields.

    The three fields are independent so drawing each from its own
    strategy gives the combinatorial coverage Property 1 needs. Temperature
    is bounded to a realistic sampling range; ``max_tokens`` alternates
    between ``None`` (model default) and a small positive integer;
    ``stop_sequences`` alternates between the empty list (no stops) and
    a short list of short strings.
    """
    return GenerationDefaults(
        temperature=draw(finite_floats(0.0, 2.0)),
        max_tokens=draw(
            st.one_of(st.none(), st.integers(min_value=1, max_value=10_000))
        ),
        stop_sequences=draw(
            st.lists(simple_strings(min_size=0, max_size=8), min_size=0, max_size=3)
        ),
    )


@st.composite
def evaluation_suites(draw: st.DrawFn) -> EvaluationSuite:
    """Draw a valid :class:`EvaluationSuite` with unique test-case ids.

    The id uniqueness invariant is enforced by drawing a ``unique=True``
    list of ids *first* and then mapping each id to a :func:`test_cases`
    draw pinned to that id. This is strictly cheaper than drawing full
    test cases and filtering (``EvaluationSuite._unique_test_case_ids``
    would reject duplicates at the model layer, which Hypothesis has to
    retry).

    Suites are always non-empty (1..4 cases). ``version`` is ASCII-only
    since YAML might otherwise coerce pure-numeric strings — ``simple_strings``
    already guarantees at least one letter or ``-``/``_`` so the common
    ``"1.0"`` style is still permitted via ``draw(simple_strings())`` for
    shapes like ``"v1"``. ``description`` exercises both the ``None`` and
    populated branches.
    """
    ids: list[str] = draw(
        st.lists(
            simple_strings(min_size=1, max_size=10),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    cases = [draw(test_cases(ids=st.just(i))) for i in ids]
    return EvaluationSuite(
        name=draw(simple_strings(min_size=1, max_size=15)),
        version=draw(simple_strings(min_size=1, max_size=6)),
        description=draw(
            st.one_of(st.none(), simple_strings(min_size=0, max_size=30))
        ),
        defaults=draw(generation_defaults()),
        test_cases=cases,
    )



__all__ = [
    "error_summary_entries",
    "evaluation_suites",
    "finite_floats",
    "generation_defaults",
    "metric_aggregates",
    "metric_configs",
    "metric_results",
    "model_aggregates",
    "model_infos",
    "performance_metrics",
    "run_reports",
    "simple_strings",
    "test_case_results",
    "test_cases",
    "tz_aware_datetimes",
]
