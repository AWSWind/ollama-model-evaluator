"""HellaSwag adapter: ``Rowan/hellaswag`` rows → :class:`EvaluationSuite`.

See ``.kiro/specs/ollama-model-evaluator/design.md`` §Dataset sources
for the canonical mapping. Key design decisions:

* **Prompt template.** Four-way multiple choice, one letter per
  continuation. Mirrors the MMLU template so a single regex-match
  metric works across both adapters.

* **Test-case id.** ``f"hellaswag/{ind}"`` — HellaSwag rows ship with
  a dataset-unique ``ind`` field, so we use it verbatim instead of
  the row's position in the filtered list. Deterministic sub-sampling
  still selects rows from the source stream; the selected rows'
  ``ind`` values carry through to the suite.

* **Tags.** ``["hellaswag", activity_label]`` so users can filter per
  activity (``tag_filter=["Baby"]``) or across every row
  (``tag_filter=["hellaswag"]``).

* **``expected_output``.** ``"ABCD"[int(label)]`` — HellaSwag stores
  the gold label as a string (``"0"`` / ``"1"`` / ``"2"`` / ``"3"``)
  in the validation split, so we coerce via ``int`` before indexing.
  Accepting an integer value too makes the adapter robust to
  future HF schema evolution without breaking existing tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

from .adapter_base import AdapterOptions, HFRef
from .mmlu import _apply_limit, _load_rows, _require_choices, _require_str
from .models import EvaluationSuite, GenerationDefaults, MetricConfig, TestCase

_ANSWER_LETTERS: tuple[str, str, str, str] = ("A", "B", "C", "D")

_PROMPT_TEMPLATE = (
    "Choose the most likely continuation.\n"
    "Context: {ctx}\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}\n"
    "Answer:"
)


class HellaSwagAdapter:
    """Turn HellaSwag rows into a first-class :class:`EvaluationSuite`."""

    ADAPTER_NAME: ClassVar[str] = "hellaswag"
    DEFAULT_HF_REF: ClassVar[HFRef] = HFRef(
        repo_id="Rowan/hellaswag",
        config=None,
        split="validation",
    )

    def rows_to_suite(
        self, rows: Iterable[dict], opts: AdapterOptions
    ) -> EvaluationSuite:
        """Convert HellaSwag rows to an :class:`EvaluationSuite`.

        Input rows carry ``ctx`` (the context string), ``endings``
        (list of 4 continuations), ``label`` (index as str or int),
        ``ind`` (dataset-unique id), and ``activity_label`` (the
        "theme" of the context — e.g. ``"Baby"`` or
        ``"Home & Garden"``).
        """
        materialised = _apply_limit(list(rows), opts.limit, opts.seed)

        test_cases: list[TestCase] = []
        for row in materialised:
            ctx = _require_str(row, "ctx")
            endings = _require_choices(row, "endings")
            label = _require_label(row, "label")
            ind = _require_ind(row, "ind")
            activity_label = _require_str(row, "activity_label")

            prompt = _PROMPT_TEMPLATE.format(
                ctx=ctx,
                A=endings[0],
                B=endings[1],
                C=endings[2],
                D=endings[3],
            )
            expected = _ANSWER_LETTERS[label]

            test_cases.append(
                TestCase(
                    id=f"hellaswag/{ind}",
                    prompt=prompt,
                    expected_output=expected,
                    tags=["hellaswag", activity_label],
                    metrics=[
                        MetricConfig(
                            name="regex-match",
                            params={"pattern": r"^\s*([ABCD])\b"},
                        )
                    ],
                )
            )

        return EvaluationSuite(
            name="hellaswag",
            description=(
                "HellaSwag: sentence-completion commonsense. Given a "
                "context, pick the most plausible continuation out of "
                "four. Probes everyday-world inference."
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
        """Load HellaSwag rows from ``cache_dir`` or the Hub, then build a suite."""
        rows = _load_rows(
            adapter_name=self.ADAPTER_NAME,
            ref=self.DEFAULT_HF_REF,
            mode=mode,
            cache_dir=cache_dir,
        )
        return self.rows_to_suite(rows, opts)


def _require_label(row: dict, field: str) -> int:
    """Pull the gold label out of ``row`` as an int in ``0..3``.

    HellaSwag stores the label as a *string* in the validation split,
    so we accept strings that parse to an int too. An integer value is
    also accepted so future schema evolution does not break the
    adapter. Any other type raises :class:`ValueError` with a message
    that names the offending field.
    """
    value = row.get(field)
    if isinstance(value, bool):
        # ``bool`` is an ``int``; exclude explicitly to avoid
        # ``label=True`` silently becoming index 1.
        raise ValueError(
            f"hellaswag row {field!r} must be an int or decimal str "
            f"(got {type(value).__name__})"
        )
    if isinstance(value, int):
        idx = value
    elif isinstance(value, str):
        try:
            idx = int(value)
        except ValueError as exc:
            raise ValueError(
                f"hellaswag row {field!r} must be an int or decimal str "
                f"(got {value!r})"
            ) from exc
    else:
        raise ValueError(
            f"hellaswag row {field!r} must be an int or decimal str "
            f"(got {type(value).__name__})"
        )
    if not 0 <= idx <= 3:
        raise ValueError(
            f"hellaswag row {field!r} must be in 0..3 (got {idx})"
        )
    return idx


def _require_ind(row: dict, field: str) -> int | str:
    """Pull the dataset id out of ``row``. Accept int or non-empty str."""
    value = row.get(field)
    if isinstance(value, bool) or (
        not isinstance(value, int) and not isinstance(value, str)
    ):
        raise ValueError(
            f"hellaswag row {field!r} must be an int or non-empty str "
            f"(got {type(value).__name__})"
        )
    if isinstance(value, str) and value == "":
        raise ValueError(f"hellaswag row {field!r} must be non-empty")
    return value


__all__ = ["HellaSwagAdapter"]
