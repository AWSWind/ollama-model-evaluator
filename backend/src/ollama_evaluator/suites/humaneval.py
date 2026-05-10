"""HumanEval adapter: ``openai_humaneval`` rows → :class:`EvaluationSuite`.

See ``.kiro/specs/ollama-model-evaluator/design.md`` §Dataset sources
and §HumanEval execution mode. Key design decisions:

* **Metric.** ``response-capture`` only. HumanEval's canonical
  ``pass@k`` metric requires executing untrusted LLM output inside a
  sandbox (design §HumanEval execution mode, Requirement 17.9); that
  is a v2+ feature. Until then the adapter records every response
  verbatim in ``MetricResult.details.response`` so an external grader
  can score later.

* **``reference_data``.** ``{"test": test, "entry_point": entry_point}``
  — the fields a future sandboxed grader needs. Stored verbatim so a
  Run_Report is self-contained.

* **``expected_output``.** ``canonical_solution`` from the dataset.
  Even though no v1 metric reads it, storing it on the test case
  keeps the Run_Report informative for human inspection and makes
  the same suites reusable when the sandboxed metric lands.

* **Tags.** ``["humaneval", "code"]``.

* **``id``.** ``f"humaneval/{row_index}"``. HumanEval rows ship with
  a ``task_id`` field of the form ``"HumanEval/42"``; we mirror that
  scheme using the filtered-list index so sub-sampling produces
  stable, contiguous ids instead of reusing the source task ids
  (which would collide if two suites both contain ``HumanEval/42``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

from .adapter_base import AdapterOptions, HFRef
from .mmlu import _apply_limit, _load_rows, _require_str
from .models import EvaluationSuite, GenerationDefaults, MetricConfig, TestCase

_PROMPT_TEMPLATE = (
    "Complete the following Python function. Return only the function "
    "body.\n{prompt}"
)


class HumanEvalAdapter:
    """Turn HumanEval rows into a first-class :class:`EvaluationSuite`.

    v1 uses the ``response-capture`` metric only; see module docstring
    and ``design.md`` §HumanEval execution mode.
    """

    ADAPTER_NAME: ClassVar[str] = "humaneval"
    DEFAULT_HF_REF: ClassVar[HFRef] = HFRef(
        repo_id="openai_humaneval",
        config=None,
        split="test",
    )

    def rows_to_suite(
        self, rows: Iterable[dict], opts: AdapterOptions
    ) -> EvaluationSuite:
        """Convert HumanEval rows to an :class:`EvaluationSuite`.

        Input rows carry ``prompt`` (str, the function signature and
        docstring), ``canonical_solution`` (str, the reference
        implementation), ``test`` (str, the unit-test code), and
        ``entry_point`` (str, the function name).
        """
        materialised = _apply_limit(list(rows), opts.limit, opts.seed)

        test_cases: list[TestCase] = []
        for row_index, row in enumerate(materialised):
            row_prompt = _require_str(row, "prompt")
            canonical_solution = _require_str(row, "canonical_solution")
            test_code = _require_str(row, "test")
            entry_point = _require_str(row, "entry_point")

            test_cases.append(
                TestCase(
                    id=f"humaneval/{row_index}",
                    prompt=_PROMPT_TEMPLATE.format(prompt=row_prompt),
                    expected_output=canonical_solution,
                    reference_data={
                        "test": test_code,
                        "entry_point": entry_point,
                    },
                    tags=["humaneval", "code"],
                    metrics=[MetricConfig(name="response-capture")],
                )
            )

        return EvaluationSuite(
            name="humaneval",
            description=(
                "OpenAI HumanEval: Python function-writing tasks. Uses "
                "response-capture in v1; execution-based pass@1 scoring "
                "needs an external sandbox runner."
            ),
            defaults=GenerationDefaults(temperature=0.0),
            test_cases=test_cases,
        )

    def materialise(
        self,
        mode: Literal["local", "remote"],
        opts: AdapterOptions,
        cache_dir: Path | None,
    ) -> EvaluationSuite:
        """Load HumanEval rows from ``cache_dir`` or the Hub, then build a suite."""
        rows = _load_rows(
            adapter_name=self.ADAPTER_NAME,
            ref=self.DEFAULT_HF_REF,
            mode=mode,
            cache_dir=cache_dir,
        )
        return self.rows_to_suite(rows, opts)


__all__ = ["HumanEvalAdapter"]
