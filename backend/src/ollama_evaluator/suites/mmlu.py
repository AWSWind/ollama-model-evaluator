"""MMLU adapter: ``cais/mmlu`` rows → :class:`EvaluationSuite`.

See ``.kiro/specs/ollama-model-evaluator/design.md`` §Dataset sources
for the full table. Key design decisions codified here:

* **Prompt template.** The template asks for a single-letter answer
  (``A``, ``B``, ``C``, or ``D``) on its own line so a simple
  ``regex-match`` on ``^\\s*([ABCD])\\b`` can extract the prediction
  from any reasonable response.

* **Test-case id.** ``f"mmlu/{subject}/{row_index}"`` — the source
  dataset doesn't ship with stable ids, so we use the dataset
  ``subject`` (from the row's ``"subject"`` column) plus the row's
  position within the filtered, sub-sampled list. Including the
  subject keeps ids disambiguated when a suite spans subjects.

* **Tags.** ``[subject, "mmlu"]`` so users can filter per-subject
  (``tag_filter=["abstract_algebra"]``) or across every MMLU question
  (``tag_filter=["mmlu"]``).

* **One suite per subject.** When ``opts.subject`` is set, only rows
  with the matching ``"subject"`` column are included; the resulting
  suite's ``name`` encodes the subject. When ``opts.subject`` is
  ``None``, every subject in ``rows`` is included in a single suite
  named ``"mmlu"`` — the CLI's ``convert mmlu`` flow typically splits
  by subject at the command layer, so the adapter stays flexible.

* **``expected_output``.** The raw answer letter (``"A"``/``"B"``/
  ``"C"``/``"D"``). The pairing with the ``regex-match`` metric means
  the match itself is 0/1 presence of the letter anywhere in the
  response; exact-letter accuracy is checked by the metric's regex
  compared against ``expected_output`` at score time.

Sub-sampling. ``opts.limit`` / ``opts.seed`` apply *after* the
``subject`` filter so ``limit=100`` with ``subject="high_school_biology"``
means "100 high-school-biology rows", not "100 rows of which
potentially zero are biology".
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

from .adapter_base import AdapterOptions, HFRef
from .models import EvaluationSuite, GenerationDefaults, MetricConfig, TestCase

_ANSWER_LETTERS: tuple[str, str, str, str] = ("A", "B", "C", "D")

_PROMPT_TEMPLATE = (
    "Answer with a single letter A, B, C, or D.\n"
    "Question: {question}\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}\n"
    "Answer:"
)


class MMluAdapter:
    """Turn MMLU rows into a first-class :class:`EvaluationSuite`.

    A class (rather than a free function) so the registry can hold a
    singleton and so the adapter satisfies the
    :class:`BenchmarkAdapter` protocol via class-level attributes.
    """

    ADAPTER_NAME: ClassVar[str] = "mmlu"
    DEFAULT_HF_REF: ClassVar[HFRef] = HFRef(
        repo_id="cais/mmlu",
        # ``cais/mmlu`` requires a config — each subject is a separate
        # config, but ``"all"`` loads the combined 14k-question set
        # across all 57 subjects. Callers who want a single subject
        # can override the ref via the ``convert hf`` CLI flow or
        # supply a pre-materialised local cache.
        config="all",
        split="test",
    )

    def rows_to_suite(
        self, rows: Iterable[dict], opts: AdapterOptions
    ) -> EvaluationSuite:
        """Convert MMLU rows to an :class:`EvaluationSuite`.

        Input rows are expected to have the fields documented in
        ``.kiro/specs/ollama-model-evaluator/design.md`` §Dataset
        sources — ``question`` (str), ``choices`` (list[str] of length
        4), ``answer`` (int in ``0..3``), and ``subject`` (str).

        Args:
            rows: Iterable of MMLU row dicts.
            opts: Adapter options. ``subject``, ``limit``, and
                ``seed`` are honoured; ``form`` is ignored
                (TruthfulQA-specific).

        Returns:
            A validated :class:`EvaluationSuite` whose test cases are
            in the same order as the (filtered, sub-sampled) input
            rows.
        """
        materialised = list(rows)
        if opts.subject is not None:
            materialised = [
                r for r in materialised if r.get("subject") == opts.subject
            ]
        materialised = _apply_limit(materialised, opts.limit, opts.seed)

        suite_name = f"mmlu-{opts.subject}" if opts.subject is not None else "mmlu"

        test_cases: list[TestCase] = []
        for row_index, row in enumerate(materialised):
            question = _require_str(row, "question")
            choices = _require_choices(row, "choices")
            answer_idx = _require_answer_index(row, "answer")
            subject = _require_str(row, "subject")

            prompt = _PROMPT_TEMPLATE.format(
                question=question,
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
            )
            answer_letter = _ANSWER_LETTERS[answer_idx]

            test_cases.append(
                TestCase(
                    id=f"mmlu/{subject}/{row_index}",
                    prompt=prompt,
                    expected_output=answer_letter,
                    tags=[subject, "mmlu"],
                    metrics=[
                        MetricConfig(
                            name="regex-match",
                            params={"pattern": r"^\s*([ABCD])\b"},
                        )
                    ],
                )
            )

        return EvaluationSuite(
            name=suite_name,
            description=(
                "MMLU: 57-subject academic multiple-choice (sciences, "
                "humanities, social sciences, professional exams). Broad "
                "knowledge benchmark; gold-standard coverage indicator."
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
        """Load MMLU rows from ``cache_dir`` or the Hub, then build a suite.

        See :class:`BenchmarkAdapter.materialise` for the general
        contract.
        """
        rows = _load_rows(
            adapter_name=self.ADAPTER_NAME,
            ref=self.DEFAULT_HF_REF,
            mode=mode,
            cache_dir=cache_dir,
        )
        return self.rows_to_suite(rows, opts)


def _apply_limit(
    rows: list[dict], limit: int | None, seed: int | None
) -> list[dict]:
    """Apply optional ``limit`` / ``seed`` sub-sampling in a deterministic way.

    * ``limit is None`` → return ``rows`` unchanged.
    * ``limit >= len(rows)`` → return ``rows`` unchanged (no padding).
    * ``seed is None`` → take the first ``limit`` rows in source
      order. This keeps deterministic sub-sampling cheap and
      reproducible across runs without forcing the caller to
      remember a seed.
    * ``seed is not None`` → shuffle a copy using
      :class:`random.Random(seed)` and take the first ``limit`` rows.
    """
    if limit is None or limit >= len(rows):
        return rows
    if seed is None:
        return rows[:limit]
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def _require_str(row: dict, field: str) -> str:
    """Pull a required string field out of ``row`` with a helpful error."""
    value = row.get(field)
    if not isinstance(value, str) or value == "":
        raise ValueError(
            f"mmlu row is missing required string field {field!r} "
            f"(got {type(value).__name__})"
        )
    return value


def _require_choices(row: dict, field: str) -> list[str]:
    """Pull a length-4 string list out of ``row``; raise with a clear message."""
    value = row.get(field)
    if not isinstance(value, list) or len(value) != 4:
        length_description = len(value) if isinstance(value, list) else "n/a"
        raise ValueError(
            f"mmlu row {field!r} must be a list of 4 strings "
            f"(got {type(value).__name__} of length {length_description})"
        )
    for idx, choice in enumerate(value):
        if not isinstance(choice, str):
            raise ValueError(
                f"mmlu row {field!r}[{idx}] must be a str "
                f"(got {type(choice).__name__})"
            )
    return list(value)


def _require_answer_index(row: dict, field: str) -> int:
    """Pull an integer answer index in ``0..3`` out of ``row``."""
    value = row.get(field)
    # ``bool`` is a subclass of ``int``; exclude it to avoid
    # ``answer=True`` silently becoming index 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"mmlu row {field!r} must be an int in 0..3 "
            f"(got {type(value).__name__})"
        )
    if not 0 <= value <= 3:
        raise ValueError(
            f"mmlu row {field!r} must be an int in 0..3 (got {value})"
        )
    return value


def _load_rows(
    *,
    adapter_name: str,
    ref: HFRef,
    mode: Literal["local", "remote"],
    cache_dir: Path | None,
) -> list[dict]:
    """Shared row-loader used by adapters that rely on the HF loader.

    Importing :mod:`.huggingface` at call time avoids a circular
    import: adapter modules are imported by :mod:`.adapters`, which in
    turn needs to be importable from tests that do not want to also
    pull in :mod:`.huggingface` (and its ``datasets`` dependency).
    """
    from .huggingface import stream_rows

    return list(stream_rows(ref, mode=mode, cache_dir=cache_dir, adapter_name=adapter_name))


__all__ = ["MMluAdapter"]
