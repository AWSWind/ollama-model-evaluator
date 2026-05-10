"""GSM8K adapter: ``openai/gsm8k`` rows → :class:`EvaluationSuite`.

See ``.kiro/specs/ollama-model-evaluator/design.md`` §Dataset sources
for the canonical mapping. Key design decisions:

* **Prompt template.** Ask the model to end its response with
  ``"Final answer: <number>"`` so the regex can anchor on a known
  token. GSM8K's own ``answer`` strings end with ``#### N`` which the
  regex also accepts so a model that copies the dataset's conventions
  still scores.

* **Regex.** ``(?i)(?:final answer:\\s*|####\\s*)(-?\\d[\\d,]*(?:\\.\\d+)?)``.
  Matches either the instruction-compliant final-answer line or the
  dataset's own ``#### N`` suffix. The captured group strips the
  prefix and leaves the raw numeric string (with optional thousands
  separators and decimal).

* **``expected_output``.** The gold numeric string as it appears after
  ``#### `` in the reference ``answer`` field, with thousands
  separators normalised out so a model that answers ``1,000`` is
  scored against the normalised ``1000`` target. The metric doesn't
  perform the normalisation — it's done here so ``expected_output``
  remains a simple string and the pattern match is effectively a
  decimal-equality check.

* **Tags.** ``["gsm8k", "math"]``.

Decimal handling. GSM8K answers are always integers in the v1.0
release, but the regex accepts decimals for robustness. When
normalising, we strip thousands separators (``,``) and preserve the
original decimal representation via :class:`decimal.Decimal` so
trailing zeros after a decimal point are preserved (``"1.20"`` ≠
``"1.2"`` mathematically are equal but textually different; using
Decimal means we compare numerically).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import ClassVar, Literal

from .adapter_base import AdapterOptions, HFRef
from .mmlu import _apply_limit, _load_rows, _require_str
from .models import EvaluationSuite, GenerationDefaults, MetricConfig, TestCase

_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")

# The metric-side pattern, stored in params so the regex-match metric
# extracts the candidate answer from the model response. Shared as a
# module constant so tests can pin their expectations to the string the
# adapter emits.
_METRIC_PATTERN = r"(?i)(?:final answer:\s*|####\s*)(-?\d[\d,]*(?:\.\d+)?)"

_PROMPT_TEMPLATE = (
    "Solve the following problem. End your response with "
    "'Final answer: <number>'.\n"
    "Problem: {question}\n"
    "Solution:"
)


class Gsm8kAdapter:
    """Turn GSM8K rows into a first-class :class:`EvaluationSuite`."""

    ADAPTER_NAME: ClassVar[str] = "gsm8k"
    DEFAULT_HF_REF: ClassVar[HFRef] = HFRef(
        repo_id="openai/gsm8k",
        config="main",
        split="test",
    )

    def rows_to_suite(
        self, rows: Iterable[dict], opts: AdapterOptions
    ) -> EvaluationSuite:
        """Convert GSM8K rows to an :class:`EvaluationSuite`.

        Input rows carry ``question`` (str) and ``answer`` (str, the
        full worked-out solution ending in ``#### N``). Rows whose
        ``answer`` does not end in a ``#### N`` block are malformed and
        raise :class:`ValueError`.
        """
        materialised = _apply_limit(list(rows), opts.limit, opts.seed)

        test_cases: list[TestCase] = []
        for row_index, row in enumerate(materialised):
            question = _require_str(row, "question")
            answer = _require_str(row, "answer")
            expected = _extract_gold_answer(answer, row_index)

            test_cases.append(
                TestCase(
                    id=f"gsm8k/{row_index}",
                    prompt=_PROMPT_TEMPLATE.format(question=question),
                    expected_output=expected,
                    tags=["gsm8k", "math"],
                    metrics=[
                        MetricConfig(
                            name="regex-match",
                            params={"pattern": _METRIC_PATTERN},
                        )
                    ],
                )
            )

        return EvaluationSuite(
            name="gsm8k",
            description=(
                "GSM8K: grade-school math word problems. Tests multi-"
                "step arithmetic reasoning; answers are exact numbers."
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
        """Load GSM8K rows from ``cache_dir`` or the Hub, then build a suite."""
        rows = _load_rows(
            adapter_name=self.ADAPTER_NAME,
            ref=self.DEFAULT_HF_REF,
            mode=mode,
            cache_dir=cache_dir,
        )
        return self.rows_to_suite(rows, opts)


def _extract_gold_answer(answer: str, row_index: int) -> str:
    """Pull the gold numeric string out of a GSM8K ``answer`` field.

    Matches the final ``#### N`` block (the canonical marker in the
    v1.0 dataset). Thousands separators are stripped so the string
    stored on the test case is directly comparable to a model
    response that emits the number without commas. The
    :class:`decimal.Decimal` parse ensures the stored string is a
    valid number (the regex already enforces shape).
    """
    match = _ANSWER_REGEX.search(answer)
    if match is None:
        raise ValueError(
            f"gsm8k row {row_index} 'answer' does not end with a '#### N' block"
        )
    raw = match.group(1)
    normalised = raw.replace(",", "")
    try:
        Decimal(normalised)
    except InvalidOperation as exc:  # pragma: no cover - regex guarantees validity
        raise ValueError(
            f"gsm8k row {row_index} gold answer {raw!r} is not a valid number"
        ) from exc
    return normalised


__all__ = ["Gsm8kAdapter"]
