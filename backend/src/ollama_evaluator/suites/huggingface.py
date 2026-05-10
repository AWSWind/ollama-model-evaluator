"""Generic HuggingFace Datasets adapter.

This module is the building block every public-benchmark adapter uses
in ``remote`` mode and the direct entry point for user-authored
``kind: huggingface`` suite files.

Public API:

* :class:`HFFieldMap` — declarative projection from dataset row to
  :class:`TestCase` fields. Supports dotted paths
  (``"answers.text"``) and bracketed list indices
  (``"answers.text[0]"``).
* :class:`HFSuiteSpec` — full user-authored spec combining an
  :class:`HFRef`, :class:`HFFieldMap`, and the metrics/defaults that
  the resulting :class:`EvaluationSuite` carries.
* :func:`materialise_hf` — pure ``(spec, rows) → EvaluationSuite``
  transformation.
* :func:`stream_rows` — adapter-facing row loader. In ``local`` mode
  reads JSONL/Parquet from ``cache_dir``; in ``remote`` mode delegates
  to :func:`datasets.load_dataset`.
* :class:`FieldMapError` — raised when a declared field fails to
  resolve on a row. Carries the row index, offending field path, and
  a reason.

Design references: ``.kiro/specs/ollama-model-evaluator/design.md``
§3a "Generic HuggingFace loader" and Requirements 17.2, 17.3, 17.4,
17.7.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .adapter_base import HFRef
from .models import EvaluationSuite, GenerationDefaults, MetricConfig, TestCase

# A single path segment can be either a bare identifier or an identifier
# immediately followed by one or more ``[N]`` indexers. The pattern is
# applied to each ``"."``-separated fragment individually.
_SEGMENT_PATTERN = re.compile(r"^([^.\[\]]+)((?:\[\d+\])*)$")
_INDEX_PATTERN = re.compile(r"\[(\d+)\]")


class FieldMapError(Exception):
    """Raised when a declared field map path fails to resolve on a row.

    The three attributes let the caller render a precise diagnostic:
    the row index (0-based within the materialised list), the field
    map path that failed, and a human-readable reason. The reason is
    stored verbatim so the scheduler's preflight can surface it
    through the ``field_map_invalid`` error envelope (Requirement
    17.7).
    """

    def __init__(self, row_index: int, field: str, reason: str) -> None:
        super().__init__(f"row {row_index}: field {field!r}: {reason}")
        self.row_index = row_index
        self.field = field
        self.reason = reason


class HFFieldMap(BaseModel):
    """Declarative row-to-:class:`TestCase` projection.

    Every field is a dotted path into the row, with optional bracketed
    list indices. ``None`` means "do not populate this field on the
    :class:`TestCase`"; the constructed test case leaves the
    corresponding attribute at its model default. ``prompt`` is
    required (a :class:`TestCase` with no prompt cannot pass
    validation).
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        ...,
        description=(
            "Dotted path to the prompt field. Required; empty or "
            "blank values are rejected."
        ),
    )
    expected_output: str | None = Field(
        default=None,
        description="Dotted path to the canonical answer. ``None`` omits the field.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Dotted path to an optional system prompt.",
    )
    choices: str | None = Field(
        default=None,
        description=(
            "Dotted path to a list of strings for multiple-choice "
            "datasets. Used only to populate ``reference_data.choices``."
        ),
    )
    tags_from: list[str] = Field(
        default_factory=list,
        description=(
            "List of column names whose *stringified* values are "
            "appended to ``TestCase.tags`` for every row."
        ),
    )

    @field_validator("prompt")
    @classmethod
    def _prompt_non_empty(cls, value: str) -> str:
        """Reject blank ``prompt`` paths."""
        if not value or not value.strip():
            raise ValueError("HFFieldMap.prompt must be a non-empty path")
        return value


class HFSuiteSpec(BaseModel):
    """Full user-authored spec for a ``kind: huggingface`` suite file."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["huggingface"] = Field(
        default="huggingface",
        description="Discriminator tag; always ``'huggingface'`` for this kind.",
    )
    name: str = Field(
        ...,
        description="Suite name (used by ``RunConfig.suites``).",
    )
    hf_ref: HFRef = Field(
        ...,
        description="Reference to the source HuggingFace dataset split.",
    )
    field_map: HFFieldMap = Field(
        ...,
        description="Row → TestCase field projection.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of rows to include. ``None`` means 'use every row'.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Sampling seed for deterministic sub-sampling when "
            "``limit`` is smaller than ``len(rows)``. ``None`` means "
            "'take the first ``limit`` rows in source order'."
        ),
    )
    dataset_mode: Literal["local", "remote"] | None = Field(
        default=None,
        description=(
            "Per-suite dataset mode override. ``None`` means 'inherit "
            "from ``ConfigFile.dataset_mode``'."
        ),
    )
    metrics: list[MetricConfig] = Field(
        ...,
        description=(
            "Scoring metrics applied to every generated TestCase. "
            "Must be non-empty so the resulting suite itself is valid."
        ),
    )
    defaults: GenerationDefaults = Field(
        default_factory=GenerationDefaults,
        description="Run-level generation defaults attached to the suite.",
    )

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("HFSuiteSpec.name must be a non-empty string")
        return value

    @field_validator("limit")
    @classmethod
    def _limit_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("HFSuiteSpec.limit must be >= 1 when set")
        return value

    @field_validator("metrics")
    @classmethod
    def _metrics_non_empty(cls, value: list[MetricConfig]) -> list[MetricConfig]:
        if len(value) == 0:
            raise ValueError("HFSuiteSpec.metrics must contain at least one MetricConfig")
        return value


# ---------------------------------------------------------------------------
# Field-map resolution
# ---------------------------------------------------------------------------


def resolve_path(row: dict, path: str, row_index: int) -> Any:
    """Resolve ``path`` into ``row``; raise :class:`FieldMapError` on failure.

    ``path`` is a ``.``-separated sequence of segments, each of which
    is a bare identifier optionally followed by one or more ``[N]``
    indexers. Examples:

    * ``"question"`` → ``row["question"]``.
    * ``"answers.text[0]"`` → ``row["answers"]["text"][0]``.
    * ``"mc1_targets.labels[2]"`` →
      ``row["mc1_targets"]["labels"][2]``.

    Missing keys, out-of-range indices, and type mismatches are all
    surfaced as :class:`FieldMapError` with a concrete reason.
    """
    current: Any = row
    for segment in path.split("."):
        match = _SEGMENT_PATTERN.match(segment)
        if match is None:
            raise FieldMapError(
                row_index, path, f"invalid path segment {segment!r}"
            )
        key, indexers = match.group(1), match.group(2)
        if not isinstance(current, dict):
            raise FieldMapError(
                row_index,
                path,
                f"expected dict at segment {key!r}, got {type(current).__name__}",
            )
        if key not in current:
            raise FieldMapError(
                row_index, path, f"missing key {key!r}"
            )
        current = current[key]
        for idx_match in _INDEX_PATTERN.finditer(indexers):
            idx = int(idx_match.group(1))
            if not isinstance(current, list):
                raise FieldMapError(
                    row_index,
                    path,
                    f"expected list for index [{idx}], got {type(current).__name__}",
                )
            if idx >= len(current) or idx < -len(current):
                raise FieldMapError(
                    row_index,
                    path,
                    f"index [{idx}] out of range (list length {len(current)})",
                )
            current = current[idx]
    if current is None:
        raise FieldMapError(row_index, path, "resolved value is None")
    return current


def _resolve_str(row: dict, path: str, row_index: int, field: str) -> str:
    """Like :func:`resolve_path` but require the result to be a non-empty ``str``."""
    value = resolve_path(row, path, row_index)
    if not isinstance(value, str):
        raise FieldMapError(
            row_index, field, f"expected str, got {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# Pure transformation
# ---------------------------------------------------------------------------


def materialise_hf(
    spec: HFSuiteSpec,
    rows: Iterable[dict] | None = None,
    *,
    mode: Literal["local", "remote"] = "local",
    cache_dir: Path | None = None,
) -> EvaluationSuite:
    """Convert an :class:`HFSuiteSpec` + rows into an :class:`EvaluationSuite`.

    Two calling patterns:

    * **Pure**: pass an explicit ``rows`` iterable. No I/O. This is
      the pattern used by Property 44 to verify field-map totality
      and injectivity without touching the HuggingFace Hub.
    * **I/O-aware**: leave ``rows`` as ``None`` and the function
      calls :func:`stream_rows` with ``spec.hf_ref`` to populate the
      row list. ``mode`` and ``cache_dir`` are forwarded.

    The function raises :class:`FieldMapError` immediately on the
    first unresolvable row, without producing a partial suite — that
    is the failure-mode contract of Property 44.
    """
    if rows is None:
        rows_iter: Iterable[dict] = stream_rows(
            spec.hf_ref, mode=mode, cache_dir=cache_dir
        )
    else:
        rows_iter = rows

    materialised = _apply_limit(list(rows_iter), spec.limit, spec.seed)

    test_cases: list[TestCase] = []
    for row_index, row in enumerate(materialised):
        prompt = _resolve_str(row, spec.field_map.prompt, row_index, "prompt")

        expected_output: str | None = None
        if spec.field_map.expected_output is not None:
            expected_output = _resolve_str(
                row,
                spec.field_map.expected_output,
                row_index,
                "expected_output",
            )

        system_prompt: str | None = None
        if spec.field_map.system_prompt is not None:
            system_prompt = _resolve_str(
                row, spec.field_map.system_prompt, row_index, "system_prompt"
            )

        reference_data: dict[str, Any] | None = None
        if spec.field_map.choices is not None:
            choices_value = resolve_path(
                row, spec.field_map.choices, row_index
            )
            if not isinstance(choices_value, list):
                raise FieldMapError(
                    row_index,
                    "choices",
                    f"expected list, got {type(choices_value).__name__}",
                )
            reference_data = {"choices": list(choices_value)}

        tags: list[str] = []
        for tag_column in spec.field_map.tags_from:
            tag_value = resolve_path(row, tag_column, row_index)
            tags.append(str(tag_value))

        test_cases.append(
            TestCase(
                id=f"{spec.name}/{row_index}",
                prompt=prompt,
                system_prompt=system_prompt,
                expected_output=expected_output,
                reference_data=reference_data,
                tags=tags,
                metrics=list(spec.metrics),
            )
        )

    return EvaluationSuite(
        name=spec.name,
        defaults=spec.defaults,
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# Row loading (local + remote)
# ---------------------------------------------------------------------------


def stream_rows(
    ref: HFRef,
    *,
    mode: Literal["local", "remote"] = "local",
    cache_dir: Path | None = None,
    adapter_name: str | None = None,
) -> Iterator[dict]:
    """Yield rows from an HF dataset reference.

    In ``local`` mode, read from the cache directory using a shallow
    naming convention: files live under
    ``cache_dir / <adapter_name or repo_id-slug> / <config>/<split>.jsonl``
    (or ``.parquet``). The adapter modules set ``adapter_name`` when
    delegating here; the generic HF loader falls back to slugifying
    the ``repo_id``.

    In ``remote`` mode, delegate to :func:`datasets.load_dataset`
    (the HuggingFace library). Test suites never call the library
    directly — they patch :func:`stream_rows` at the call site or
    patch the ``datasets.load_dataset`` symbol, per Requirement 17.4.

    Network errors in ``remote`` mode propagate verbatim so the
    scheduler's preflight can surface them as
    ``error_code=dataset_fetch_failed`` (Requirement 17.7).
    """
    if mode == "local":
        yield from _read_local_rows(ref, cache_dir=cache_dir, adapter_name=adapter_name)
        return
    yield from _stream_remote_rows(ref)


def _read_local_rows(
    ref: HFRef,
    *,
    cache_dir: Path | None,
    adapter_name: str | None,
) -> Iterator[dict]:
    """Read pre-cached JSONL/Parquet rows from disk.

    ``cache_dir`` must be supplied in ``local`` mode — there is no
    sensible system-wide default for a local cache. The function
    raises :class:`FileNotFoundError` when the resolved path does not
    exist, which matches the design's "disk-level only" failure mode
    for local materialisation.
    """
    if cache_dir is None:
        raise FileNotFoundError(
            "local mode requires cache_dir to be set"
        )
    base = Path(cache_dir) / (adapter_name or _slugify_repo_id(ref.repo_id))
    if ref.config is not None:
        base = base / ref.config
    split = ref.split or "train"
    jsonl_path = base / f"{split}.jsonl"
    parquet_path = base / f"{split}.parquet"
    if jsonl_path.exists():
        yield from _iter_jsonl(jsonl_path)
        return
    if parquet_path.exists():
        yield from _iter_parquet(parquet_path)
        return
    raise FileNotFoundError(
        f"no local cache for {ref.repo_id!r} split {split!r} "
        f"under {base} (looked for {jsonl_path.name} and {parquet_path.name})"
    )


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield dicts from a JSONL file, one per line."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_parquet(path: Path) -> Iterator[dict]:
    """Yield dicts from a Parquet file via ``pyarrow`` (lazy import)."""
    import pyarrow.parquet as pq  # Lazy import: pyarrow is heavy.

    table = pq.read_table(path)
    yield from table.to_pylist()


def _stream_remote_rows(ref: HFRef) -> Iterator[dict]:
    """Stream rows from the HuggingFace Hub via :func:`datasets.load_dataset`.

    Lazy-imported so tests can patch ``datasets.load_dataset`` before
    the first call, and so the module's import time stays small for
    users who never exercise ``remote`` mode.
    """
    from datasets import load_dataset  # Lazy: datasets is heavy.

    kwargs: dict[str, Any] = {"streaming": True}
    if ref.config is not None:
        kwargs["name"] = ref.config
    if ref.split is not None:
        kwargs["split"] = ref.split
    if ref.revision is not None:
        kwargs["revision"] = ref.revision
    stream = load_dataset(ref.repo_id, **kwargs)
    for row in stream:
        yield dict(row)


def _slugify_repo_id(repo_id: str) -> str:
    """Turn ``owner/dataset`` into a filesystem-safe directory name."""
    return repo_id.replace("/", "__")


def _apply_limit(
    rows: list[dict], limit: int | None, seed: int | None
) -> list[dict]:
    """Apply optional sub-sampling deterministically.

    Mirrors the implementation in :mod:`ollama_evaluator.suites.mmlu`.
    Duplicated (rather than imported) so the generic HF loader stays
    independent of the public-benchmark adapters — tests exercise
    both without coupling the modules.
    """
    if limit is None or limit >= len(rows):
        return rows
    if seed is None:
        return rows[:limit]
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


__all__ = [
    "FieldMapError",
    "HFFieldMap",
    "HFSuiteSpec",
    "materialise_hf",
    "resolve_path",
    "stream_rows",
]
