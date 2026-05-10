"""TruthfulQA MC1 adapter: ``truthful_qa`` rows → :class:`EvaluationSuite`.

See ``.kiro/specs/ollama-model-evaluator/design.md`` §Dataset sources
for the canonical mapping. Key design decisions:

* **Prompt template.** Enumerated choices with A/B/C/... labels, one
  label per MC1 option. The number of options varies per row (MC1
  typically has 4–13), so we enumerate dynamically rather than
  hard-coding four slots.

* **Test-case id.** ``f"truthfulqa/mc1/{row_index}"``. TruthfulQA
  rows don't ship stable ids; using the index within the filtered,
  sub-sampled list is consistent with MMLU's fallback scheme.

* **Tags.** ``["truthfulqa", "mc1", category]`` so users can filter
  per category or across the benchmark as a whole.

* **``expected_output``.** The letter corresponding to the index where
  ``mc1_targets.labels == 1``. Exactly one entry must be ``1``
  (TruthfulQA's MC1 contract); otherwise the row is malformed and
  we surface a :class:`ValueError` naming the row.

* **Form.** Only ``opts.form == "mc1"`` is supported in v1 (design
  §Dataset sources). Callers that pass ``form="mc2"`` get a clear
  :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

from .adapter_base import AdapterOptions, HFRef
from .mmlu import _apply_limit, _load_rows, _require_str
from .models import EvaluationSuite, GenerationDefaults, MetricConfig, TestCase


def _optional_category(row: dict) -> str:
    """Return ``row['category']`` if present and non-empty, else ``''``.

    The ``truthful_qa`` HuggingFace release no longer ships the
    ``category`` column with the ``multiple_choice`` config
    (observed 2026-05-10: keys are ``mc1_targets``, ``mc2_targets``,
    ``question`` only). Earlier releases included it, so the adapter
    reads the field when available but no longer requires it. Tags
    still include ``"truthfulqa"`` and ``"mc1"`` so filtering by
    benchmark works; per-category filtering is only possible when the
    upstream dataset provides the field.
    """
    value = row.get("category")
    if isinstance(value, str) and value:
        return value
    return ""

# Upper bound on MC1 options. TruthfulQA MC1 tops out at 13 in the v1.0
# release; 26 (A..Z) gives headroom for future growth without forcing
# us to switch to multi-letter labels. Strictly a safety rail — any row
# that exceeds this bound is clearly malformed.
_ANSWER_LETTERS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TruthfulQaAdapter:
    """Turn TruthfulQA MC1 rows into a first-class :class:`EvaluationSuite`."""

    ADAPTER_NAME: ClassVar[str] = "truthfulqa"
    DEFAULT_HF_REF: ClassVar[HFRef] = HFRef(
        repo_id="truthful_qa",
        config="multiple_choice",
        split="validation",
    )

    def rows_to_suite(
        self, rows: Iterable[dict], opts: AdapterOptions
    ) -> EvaluationSuite:
        """Convert TruthfulQA MC1 rows to an :class:`EvaluationSuite`.

        Input rows carry ``question`` (str), ``mc1_targets`` (a dict
        with ``choices: list[str]`` and ``labels: list[int]`` where
        exactly one label is ``1``), and ``category`` (str).
        """
        if opts.form != "mc1":
            raise NotImplementedError(
                f"truthfulqa adapter v1 supports only form='mc1' "
                f"(got {opts.form!r})"
            )

        materialised = _apply_limit(list(rows), opts.limit, opts.seed)

        test_cases: list[TestCase] = []
        for row_index, row in enumerate(materialised):
            question = _require_str(row, "question")
            category = _optional_category(row)
            choices, labels = _require_mc1_targets(row)

            if len(choices) > len(_ANSWER_LETTERS):
                raise ValueError(
                    f"truthfulqa row {row_index} has {len(choices)} choices; "
                    f"only up to {len(_ANSWER_LETTERS)} are supported"
                )

            answer_index = _exactly_one_correct_index(labels, row_index)
            expected = _ANSWER_LETTERS[answer_index]

            enumerated = "\n".join(
                f"{_ANSWER_LETTERS[i]}) {choices[i]}" for i in range(len(choices))
            )
            prompt = f"Question: {question}\n{enumerated}\nAnswer:"

            tags = ["truthfulqa", "mc1"]
            if category:
                tags.append(category)
            test_cases.append(
                TestCase(
                    id=f"truthfulqa/mc1/{row_index}",
                    prompt=prompt,
                    expected_output=expected,
                    tags=tags,
                    metrics=[
                        MetricConfig(
                            name="regex-match",
                            params={"pattern": r"^\s*([A-Z])\b"},
                        )
                    ],
                )
            )

        return EvaluationSuite(
            name="truthfulqa-mc1",
            description=(
                "TruthfulQA (MC1): multiple-choice questions where the "
                "wrong answers match popular misconceptions. Probes "
                "resistance to hallucinating plausible-sounding falsehoods."
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
        """Load TruthfulQA rows from ``cache_dir`` or the Hub, then build a suite."""
        rows = _load_rows(
            adapter_name=self.ADAPTER_NAME,
            ref=self.DEFAULT_HF_REF,
            mode=mode,
            cache_dir=cache_dir,
        )
        return self.rows_to_suite(rows, opts)


def _require_mc1_targets(row: dict) -> tuple[list[str], list[int]]:
    """Extract ``mc1_targets.choices`` and ``mc1_targets.labels`` with validation."""
    targets = row.get("mc1_targets")
    if not isinstance(targets, dict):
        raise ValueError(
            "truthfulqa row 'mc1_targets' must be a dict "
            f"(got {type(targets).__name__})"
        )
    choices = targets.get("choices")
    labels = targets.get("labels")
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError(
            "truthfulqa row 'mc1_targets.choices' must be a non-empty list"
        )
    if not isinstance(labels, list) or len(labels) != len(choices):
        raise ValueError(
            "truthfulqa row 'mc1_targets.labels' must be a list the same "
            "length as 'mc1_targets.choices'"
        )
    for idx, choice in enumerate(choices):
        if not isinstance(choice, str):
            raise ValueError(
                f"truthfulqa row 'mc1_targets.choices[{idx}]' must be a str "
                f"(got {type(choice).__name__})"
            )
    for idx, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(
                f"truthfulqa row 'mc1_targets.labels[{idx}]' must be an int "
                f"(got {type(label).__name__})"
            )
    return list(choices), list(labels)


def _exactly_one_correct_index(labels: list[int], row_index: int) -> int:
    """Find the unique index where ``labels[i] == 1`` (MC1 contract).

    Raises :class:`ValueError` if zero or more than one label is ``1``
    so malformed rows surface immediately with a useful location.
    """
    correct_indices = [i for i, v in enumerate(labels) if v == 1]
    if len(correct_indices) != 1:
        raise ValueError(
            f"truthfulqa row {row_index} must have exactly one correct "
            f"label in 'mc1_targets.labels' (got {len(correct_indices)})"
        )
    return correct_indices[0]


__all__ = ["TruthfulQaAdapter"]
