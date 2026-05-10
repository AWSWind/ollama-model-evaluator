"""Shared data contracts for public-benchmark and HuggingFace adapters.

This module defines the three pieces every adapter under
:mod:`ollama_evaluator.suites` shares:

* :class:`AdapterOptions` — Pydantic-validated per-materialisation knobs
  (row limit, sampling seed, MMLU subject filter, TruthfulQA answer
  form). Individual adapters ignore the fields they do not use; the
  model is shared rather than per-adapter so the CLI and the test
  fixtures can build one instance and hand it to any adapter.

* :class:`HFRef` — strongly-typed reference to a HuggingFace dataset
  split (``repo_id[:config][:split][@revision]``) plus a
  :meth:`HFRef.parse` classmethod that accepts the short-form
  ``repo_id[:config][:split]`` string used in suite files and on the
  CLI (``--hf-ref``).

* :class:`BenchmarkAdapter` — a :class:`typing.Protocol` capturing the
  two-method interface every public-benchmark adapter in
  :mod:`ollama_evaluator.suites` exposes. The protocol carries two
  ``ClassVar`` attributes (``ADAPTER_NAME``, ``DEFAULT_HF_REF``) plus
  the pure :meth:`BenchmarkAdapter.rows_to_suite` transformation and
  the I/O-aware :meth:`BenchmarkAdapter.materialise` dispatcher.

Design references: ``.kiro/specs/ollama-model-evaluator/design.md``
§Dataset sources (Requirements 17.1, 17.3, 17.4) and §Components /
§3. Benchmark adapters.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import EvaluationSuite


class AdapterOptions(BaseModel):
    """Per-materialisation knobs for every public-benchmark adapter.

    The model is intentionally the union of every field any v1 adapter
    needs, not the intersection: a single shape keeps the CLI entry
    points and test fixtures uniform. Adapters that do not consume a
    field (for example, HellaSwag does not read ``subject``) simply
    ignore it; Pydantic's ``extra="forbid"`` still rejects typos at the
    boundary.

    Fields:

    * ``limit`` — maximum number of rows to include in the resulting
      :class:`EvaluationSuite`. ``None`` (default) means "use every
      row". When set, sampling is deterministic under ``seed``.
    * ``seed`` — random seed driving the deterministic sub-sample when
      ``limit`` is smaller than the source row count. ``None`` means
      "take the first ``limit`` rows in source order" so local mode
      without a seed is fully reproducible.
    * ``subject`` — MMLU subject filter (e.g. ``"abstract_algebra"``).
      ``None`` means "include every subject present in ``rows``".
    * ``form`` — TruthfulQA answer form selector. Only ``"mc1"`` is
      implemented in v1 (Requirement 17.10 notes MC1 as the v1 form);
      ``"mc2"`` is reserved so a future adapter can add MC2 without a
      config-schema break.
    """

    model_config = ConfigDict(extra="forbid")

    limit: int | None = Field(
        default=None,
        description=(
            "Maximum number of rows to emit. ``None`` means 'use every "
            "row'. Must be >= 1 when set."
        ),
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Sampling seed applied when ``limit`` is smaller than "
            "``len(rows)``. ``None`` means 'take the first ``limit`` "
            "rows in source order'."
        ),
    )
    subject: str | None = Field(
        default=None,
        description=(
            "MMLU subject filter. ``None`` means 'include every "
            "subject present in the source rows'."
        ),
    )
    form: Literal["mc1", "mc2"] = Field(
        default="mc1",
        description=(
            "TruthfulQA answer form. Only ``mc1`` is implemented in "
            "v1; ``mc2`` is reserved for a future adapter."
        ),
    )

    @field_validator("limit")
    @classmethod
    def _limit_positive(cls, value: int | None) -> int | None:
        """Reject ``limit=0``: "zero rows" is a degenerate empty suite."""
        if value is not None and value < 1:
            raise ValueError(
                f"AdapterOptions.limit must be >= 1 when set (got {value})"
            )
        return value


class HFRef(BaseModel):
    """Strongly-typed reference to a HuggingFace dataset split.

    :meth:`HFRef.parse` accepts the short-form string
    ``repo_id[:config][:split]`` used in suite files and on the CLI,
    so users rarely need to build an :class:`HFRef` by hand.

    ``revision`` is not part of the short-form string but is modelled
    explicitly so suite files can pin a dataset to a specific commit
    hash or branch for reproducibility. The default ``None`` defers
    the choice to the HuggingFace ``datasets`` library, which selects
    ``"main"`` at load time.
    """

    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(
        ...,
        description='Dataset repository id (e.g. ``"cais/mmlu"``). Must be non-empty.',
    )
    config: str | None = Field(
        default=None,
        description='Dataset config name (e.g. ``"abstract_algebra"``).',
    )
    split: str | None = Field(
        default=None,
        description='Split name (``"train"``, ``"validation"``, ``"test"``, ...).',
    )
    revision: str | None = Field(
        default=None,
        description=(
            "Commit hash or branch pinning. ``None`` defers to the "
            "``datasets`` library default (``'main'``)."
        ),
    )

    @field_validator("repo_id")
    @classmethod
    def _repo_id_non_empty(cls, value: str) -> str:
        """Reject blank ``repo_id``; ``datasets.load_dataset`` would fail obscurely."""
        if not value or not value.strip():
            raise ValueError("HFRef.repo_id must be a non-empty string")
        return value

    @classmethod
    def parse(cls, spec: str) -> HFRef:
        """Parse ``repo_id[:config][:split]`` into a validated :class:`HFRef`.

        The short form mirrors the one used in hand-authored suite
        files (see the design document's §Dataset sources example). A
        ``repo_id`` with no trailing colons yields ``config=None`` and
        ``split=None``. Supplying four or more colon-separated segments
        is a :class:`ValueError`; ``revision`` is never accepted via
        the short form because revisions usually contain ``:``
        characters themselves (unrelated to the separator).

        Args:
            spec: The short-form reference string. Must be non-empty.

        Returns:
            A :class:`HFRef` with ``revision=None``.

        Raises:
            ValueError: ``spec`` is empty, has too many colon-separated
                segments, or has an empty ``repo_id`` segment.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("HFRef.parse requires a non-empty spec string")
        parts = spec.split(":")
        if len(parts) > 3:
            raise ValueError(
                "HFRef.parse expected 'repo_id[:config][:split]'; "
                f"got {len(parts)} colon-separated segments in {spec!r}"
            )
        # Pad to length 3 so the unpack below is total; ``None``
        # carries through to the optional fields.
        while len(parts) < 3:
            parts.append("")
        repo_id, config, split = parts
        return cls(
            repo_id=repo_id,
            config=config or None,
            split=split or None,
            revision=None,
        )


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Structural contract every public-benchmark adapter satisfies.

    Every adapter module (``mmlu``, ``hellaswag``, ``truthfulqa``,
    ``gsm8k``, ``humaneval``) exposes a class or module-level object
    satisfying this protocol. The protocol has two class-level
    attributes and two methods:

    * :attr:`ADAPTER_NAME` — stable identifier used as the registry
      key (see :mod:`ollama_evaluator.suites.adapters`). Matches the
      CLI subcommand name ``convert <adapter_name>``.

    * :attr:`DEFAULT_HF_REF` — canonical :class:`HFRef` used in
      ``remote`` mode when the user does not override it. The test
      suite uses this to pin expectations for Property 43 (local/
      remote equivalence).

    * :meth:`rows_to_suite` — pure function ``(rows, opts) ->
      EvaluationSuite``. No I/O, no global state. Called by both
      ``local`` and ``remote`` mode inside :meth:`materialise`.

    * :meth:`materialise` — I/O-aware dispatcher. In ``local`` mode
      reads pre-cached files from ``cache_dir`` and delegates to
      :meth:`rows_to_suite`; in ``remote`` mode streams rows from the
      HuggingFace Hub via :func:`ollama_evaluator.suites.huggingface.stream_rows`
      and then delegates to the same :meth:`rows_to_suite`.

    Using a :class:`typing.Protocol` lets adapters be plain classes
    without requiring inheritance from a shared base. The
    :func:`typing.runtime_checkable` decoration allows tests and the
    registry to assert conformance via :func:`isinstance`.
    """

    ADAPTER_NAME: ClassVar[str]
    DEFAULT_HF_REF: ClassVar[HFRef]

    def rows_to_suite(
        self, rows: Iterable[dict], opts: AdapterOptions
    ) -> EvaluationSuite:
        """Transform a stream of source rows into an :class:`EvaluationSuite`.

        Implementations MUST be pure: no file or network I/O, no
        dependence on module-level mutable state. Every adapter test
        calls this method directly with a fixture-provided row list,
        so any side effect would break the pure-function contract that
        Property 42 relies on.
        """
        ...

    def materialise(
        self,
        mode: Literal["local", "remote"],
        opts: AdapterOptions,
        cache_dir: Path | None,
    ) -> EvaluationSuite:
        """Materialise the adapter's :class:`EvaluationSuite` for ``mode``.

        Dispatches on ``mode``:

        * ``local`` — read rows from the pre-cached files under
          ``cache_dir / ADAPTER_NAME / ...`` and call
          :meth:`rows_to_suite`. Fails with :class:`FileNotFoundError`
          (or a similar OS-level error) if the cache is missing.
        * ``remote`` — stream rows from the HuggingFace Hub using
          :attr:`DEFAULT_HF_REF` and call :meth:`rows_to_suite`. The
          caller in tests mocks the ``stream_rows`` function so no
          real network I/O happens.

        The two modes produce equal suites up to ``TestCase`` identity
        (Property 43): same ``name``, same ordered ``TestCase.id``
        list, and equal per-field contents.
        """
        ...


__all__ = [
    "AdapterOptions",
    "BenchmarkAdapter",
    "HFRef",
]
