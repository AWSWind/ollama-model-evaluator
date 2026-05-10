"""Plan ``(model, TestCase, repetition)`` executions for a Run.

This module owns the pure function the scheduler calls at the start of
a Run to turn ``(suites, RunConfig)`` into the fully-expanded list of
executions to dispatch. It is deliberately side-effect-free so
Properties 4, 6, and 7 can assert over it directly without spinning up
an event loop or an Ollama fixture.

Public API:

* :func:`select_executions` — return the list of
  ``(model, TestCase, repetition)`` tuples produced by applying every
  filter and cross-product rule declared in ``design.md`` §Properties
  4 / 6 / 7.
* :func:`resolve_generate_options` — compute the effective
  :class:`ollama_evaluator.ollama.types.GenerateOptions` for a
  single :class:`TestCase` given the run-level
  :class:`GenerationDefaults` (Property 7).

Filtering contract (Property 4)
-------------------------------
Only suites whose ``name`` appears in ``config.suites`` contribute any
test cases. Within each surviving suite, a test case is included iff

* ``config.tag_filter == []`` (no filter, everything passes), or
* ``set(tc.tags) & set(config.tag_filter) != set()``.

Ordering contract (Property 6)
------------------------------
The returned list iterates:

1. Models in ``config.models`` order.
2. Suites in the order ``config.suites`` lists them (intersected
   with the suites the caller actually discovered — suites named in
   the config but not present in ``suites`` are skipped silently).
3. Test cases within each suite in suite-declaration order.
4. Repetition 1..R in ascending order.

The order is deterministic so the ``run-progress`` event stream and
the Run_Report results table reconstruct the same sequence on replay.

Generation-parameter resolution (Property 7)
--------------------------------------------
For each of the three generation fields, the resolved value is
``tc.<field> if tc.<field> is not None else run_defaults.<field>``.
``stop_sequences`` gets a three-way treatment because
:class:`~ollama_evaluator.suites.models.TestCase` defines ``None``
(inherit from defaults) as a distinct state from ``[]`` (explicit
"no stop sequences"). The :class:`GenerationDefaults` default is
``[]``, so "inherit" from the common case still yields an empty
list, but a user who sets ``defaults.stop_sequences=["END"]`` and
leaves ``tc.stop_sequences=None`` sees ``["END"]`` reach Ollama.
"""

from __future__ import annotations

from ..config import RunConfig
from ..ollama.types import GenerateOptions
from ..suites.models import EvaluationSuite, GenerationDefaults, TestCase


def select_executions(
    suites: list[EvaluationSuite],
    config: RunConfig,
) -> list[tuple[str, TestCase, int]]:
    """Plan the fully-expanded list of executions for a Run.

    Args:
        suites: The :class:`EvaluationSuite` objects discovered from
            the suites directory. The function filters and orders
            them against ``config``; extra suites not referenced in
            ``config.suites`` are silently skipped, and entries in
            ``config.suites`` that have no matching discovered suite
            are silently skipped (the caller has already validated
            this at preflight when it matters).
        config: The :class:`RunConfig` driving the Run.

    Returns:
        A list of ``(model, test_case, repetition)`` tuples in the
        order described in the module docstring. The same list
        satisfies:

        * Property 4 — tag/name filtering,
        * Property 6 — execution count and coverage.
    """
    by_name: dict[str, EvaluationSuite] = {s.name: s for s in suites}
    tag_filter = set(config.tag_filter)

    executions: list[tuple[str, TestCase, int]] = []
    for model in config.models:
        for suite_name in config.suites:
            suite = by_name.get(suite_name)
            if suite is None:
                # Caller has already validated presence at preflight
                # when that matters; silently skip here so this pure
                # function has no reason to raise.
                continue
            for tc in suite.test_cases:
                if not _passes_tag_filter(tc, tag_filter):
                    continue
                for repetition in range(1, config.repetitions + 1):
                    executions.append((model, tc, repetition))
    return executions


def _passes_tag_filter(tc: TestCase, tag_filter: set[str]) -> bool:
    """Return ``True`` iff ``tc`` satisfies the tag filter.

    The filter's semantics match Property 4: an empty filter admits
    every test case, a non-empty filter admits cases whose tags
    intersect the filter (set-intersection semantics, not
    subset-containment).
    """
    if not tag_filter:
        return True
    return bool(set(tc.tags) & tag_filter)


def resolve_generate_options(
    tc: TestCase,
    run_defaults: GenerationDefaults,
) -> GenerateOptions:
    """Compute the effective :class:`GenerateOptions` for ``tc``.

    Implements Property 7 directly: for each field, the resolved
    value is ``tc.<field> if tc.<field> is not None else
    run_defaults.<field>``.

    * ``temperature`` — :class:`GenerationDefaults.temperature` is a
      plain ``float`` (non-nullable) so the fallback is always a
      concrete value.
    * ``max_tokens`` — both ``tc.max_tokens`` and
      ``run_defaults.max_tokens`` may be ``None``; the resolved
      :class:`GenerateOptions.num_predict` mirrors that.
    * ``stop_sequences`` — see the module docstring for the
      three-way treatment.

    The resulting :class:`GenerateOptions` uses Ollama-native field
    names (``num_predict`` for ``max_tokens``, ``stop`` for
    ``stop_sequences``); the scheduler dumps it via
    ``model_dump(exclude_none=True)`` when building the Ollama
    request payload.
    """
    temperature = (
        tc.temperature if tc.temperature is not None else run_defaults.temperature
    )
    max_tokens = (
        tc.max_tokens if tc.max_tokens is not None else run_defaults.max_tokens
    )
    stop_sequences: list[str] = (
        tc.stop_sequences if tc.stop_sequences is not None else run_defaults.stop_sequences
    )

    return GenerateOptions(
        temperature=temperature,
        num_predict=max_tokens,
        stop=list(stop_sequences),
    )


__all__ = [
    "resolve_generate_options",
    "select_executions",
]
