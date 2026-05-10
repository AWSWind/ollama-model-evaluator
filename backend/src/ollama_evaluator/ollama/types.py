"""Pydantic v2 models mirroring the Ollama HTTP API response shapes.

These models are the wire-level types consumed by
:class:`~ollama_evaluator.ollama.client.OllamaClient`. They are
*internal* to the Ollama adapter: the Run_Report and Evaluation_Suite
models in :mod:`ollama_evaluator.models` and
:mod:`ollama_evaluator.suites.models` are the user-facing shapes. A
separate set of types here lets the client evolve independently of the
report schema, keeps ``extra="forbid"`` discipline on user-facing models
while tolerating new Ollama fields at the boundary, and keeps the list
of imports inside :mod:`ollama_evaluator.ollama.client` small.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Components and Interfaces > Ollama Client, and
§Data Models > Ollama_Server fields.

The following Ollama endpoints feed these models:

* ``GET /api/tags`` — :class:`OllamaModelInfo` (via :func:`_parse_tags_entry`)
* ``POST /api/generate`` (streaming) — :class:`GenerateChunk`
* ``POST /api/pull`` (streaming) — :class:`PullProgress`

``GenerateOptions`` is an *outbound* model: it is serialised into the
``options`` object of the ``POST /api/generate`` request body. The three
fields here are the ones the Backend surfaces to users via
``GenerationDefaults`` and ``TestCase`` overrides (Requirements 5.3, 5.4)
— Ollama supports many more options, but exposing them is out of scope
for v1 and a future task can add fields without breaking callers thanks
to Pydantic's opt-in semantics.

Key design decisions codified here:

* **``extra="ignore"`` on every model.** Ollama adds fields between
  versions (for example ``"model"`` alongside ``"name"`` in ``/api/tags``,
  or new timing fields in ``/api/generate``). The Backend must not break
  when running against a newer Ollama_Server, so unknown fields are
  silently dropped rather than rejected. This is the opposite of the
  ``extra="forbid"`` policy used on user-facing report models
  (:mod:`ollama_evaluator.models`) — that policy protects the JSON
  report schema from hand-edit typos; here we need to protect against
  upstream schema drift.

* **Every field beyond the bare minimum is ``Optional``.** Requirement
  6.5 requires the Backend to record performance fields as ``None``
  rather than fail the Test_Case when the Ollama_Server omits them; the
  simplest way to honour that at the boundary is to make those fields
  nullable at the wire type itself.

* **``_parse_tags_entry`` flattens ``details``.** ``GET /api/tags``
  nests ``parameter_size`` and ``quantization_level`` under a
  ``details`` object. The runner only cares about the flat shape used
  by :class:`~ollama_evaluator.models.ModelInfo`, so the adapter owns
  the flattening instead of forcing every caller to reach into
  ``details``.

* **Accept both ``"name"`` and ``"model"`` keys in ``/api/tags``.**
  Ollama renamed the identifier field from ``"name"`` to ``"model"``
  between v0.1.x and v0.2.x. The Backend supports both running versions
  of Ollama in the field, so :func:`_parse_tags_entry` accepts either
  key (preferring ``"name"`` when both are present because older
  deployments remain the more common setup and we want deterministic
  behaviour when both are populated).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OllamaModelInfo(BaseModel):
    """Flattened representation of one entry from ``GET /api/tags``.

    Populated by :func:`_parse_tags_entry`, not constructed directly
    from the raw Ollama response. The flattening step lifts
    ``parameter_size`` and ``quantization_level`` out of the nested
    ``details`` object so downstream code can treat model metadata as a
    flat record (Requirement 2.5).

    Every field beyond ``name`` is nullable because older Ollama_Server
    versions and locally-imported models may omit any of them; the
    Backend records what is present and skips what is not.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        description=(
            'Ollama model tag (for example ``"llama3:8b"``). Populated '
            'from the ``"name"`` key, falling back to ``"model"`` for '
            "Ollama v0.2.x+ where the field was renamed."
        ),
    )
    digest: str | None = Field(
        default=None,
        description=(
            "Content-addressed digest reported by the Ollama_Server "
            '(for example ``"sha256:..."``). ``None`` when the server '
            "omits it."
        ),
    )
    size: int | None = Field(
        default=None,
        description=(
            "On-disk size of the model artifacts, in bytes. ``None`` "
            "when the server omits it."
        ),
    )
    modified_at: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp of the most recent modification to the local "
            "model artifacts. ``None`` when the server omits it."
        ),
    )
    parameter_size: str | None = Field(
        default=None,
        description=(
            'Human-readable parameter count (for example ``"8B"``). '
            "Lifted from ``details.parameter_size``; ``None`` when the "
            "``details`` object is absent or does not carry the field."
        ),
    )
    quantization_level: str | None = Field(
        default=None,
        description=(
            'Quantisation label (for example ``"Q4_0"``). Lifted from '
            "``details.quantization_level``; ``None`` when absent."
        ),
    )


class GenerateOptions(BaseModel):
    """Subset of Ollama's ``POST /api/generate`` ``options`` object.

    Serialised into the request body as the ``options`` sub-object. Only
    the three fields the Backend exposes to Test_Cases and
    GenerationDefaults are modelled here (Requirements 5.3, 5.4); Ollama
    accepts many more knobs but exposing them is out of scope for v1.

    ``num_predict`` is Ollama's name for what most LLM APIs call
    ``max_tokens``. The Backend keeps the Ollama-native name at the
    wire boundary and translates to/from ``max_tokens`` in the caller so
    the wire payload matches Ollama's documentation verbatim.

    All three fields are optional: when a field is ``None`` it is
    excluded from the serialised request so the Ollama_Server applies
    its own default. Callers use ``model_dump(exclude_none=True)`` when
    building the outbound payload.
    """

    model_config = ConfigDict(extra="ignore")

    temperature: float | None = Field(
        default=None,
        description=(
            "Sampling temperature. ``None`` means 'use the "
            "Ollama_Server default' (typically 0.8)."
        ),
    )
    num_predict: int | None = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (Ollama's name for "
            "``max_tokens``). ``None`` means 'use the Ollama_Server "
            "default' (typically 128)."
        ),
    )
    stop: list[str] | None = Field(
        default=None,
        description=(
            "Stop sequences. When any of these strings is produced the "
            "Ollama_Server ends the response. ``None`` means 'no stop "
            "sequences'."
        ),
    )


class GenerateChunk(BaseModel):
    """One streamed chunk from ``POST /api/generate``.

    Ollama's streaming response is newline-delimited JSON where every
    chunk carries ``model``, ``created_at``, ``response``, and ``done``.
    Non-final chunks carry a partial ``response`` fragment (which may be
    empty); the final chunk carries an empty ``response`` but populates
    the timing and token-count fields.

    Every timing/token-count field is optional because the Ollama_Server
    does not always populate them, and because declaring them
    unconditionally would reject every non-final chunk. Requirement 6.5
    requires the Backend to record missing fields as ``None`` rather
    than fail the Test_Case, so nullable types at the wire boundary are
    the simplest way to honour that invariant.

    Time-to-first-token (Requirement 6.1) is not a field on this model;
    the Backend measures it as the wall-clock delta between request
    dispatch and the arrival of the first chunk with ``response != ""``.
    Total duration (Requirement 6.2) is reported by Ollama in
    nanoseconds on the final chunk; the Backend converts to milliseconds
    at the client boundary.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = Field(
        ...,
        description="Ollama model tag that produced this chunk.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the Ollama_Server emitted this chunk.",
    )
    response: str = Field(
        ...,
        description=(
            "Partial response text in this chunk. Empty string on the "
            "final chunk and on occasional keep-alive chunks."
        ),
    )
    thinking: str | None = Field(
        default=None,
        description=(
            "Reasoning-trace text emitted by Qwen-style models that "
            "stream their chain-of-thought in a separate field (Ollama "
            "0.5+). ``None`` on models that do not emit a ``thinking`` "
            "field. The scheduler treats ``thinking`` as part of the "
            "response when ``response`` is empty so metrics applied to "
            "reasoning models still see non-empty text."
        ),
    )
    done: bool = Field(
        ...,
        description=(
            "``True`` on the final chunk of a generation. The final "
            "chunk also populates the timing and token-count fields "
            "below; non-final chunks leave them ``None``."
        ),
    )

    total_duration: int | None = Field(
        default=None,
        description=(
            "Total generation duration in *nanoseconds* as reported by "
            "Ollama (Requirement 6.2). Populated on the final chunk "
            "only; ``None`` on partial chunks and when the server "
            "omits it."
        ),
    )
    load_duration: int | None = Field(
        default=None,
        description=(
            "Time in *nanoseconds* spent loading model weights for this "
            "request. Populated on the final chunk only."
        ),
    )
    prompt_eval_count: int | None = Field(
        default=None,
        description=(
            "Number of prompt tokens evaluated (Requirement 6.3). "
            "Populated on the final chunk only; ``None`` on partial "
            "chunks and when the server omits it (Requirement 6.5)."
        ),
    )
    prompt_eval_duration: int | None = Field(
        default=None,
        description=(
            "Time in *nanoseconds* spent on prompt evaluation. "
            "Populated on the final chunk only."
        ),
    )
    eval_count: int | None = Field(
        default=None,
        description=(
            "Number of response tokens generated (Requirement 6.3). "
            "Populated on the final chunk only; ``None`` on partial "
            "chunks and when the server omits it (Requirement 6.5)."
        ),
    )
    eval_duration: int | None = Field(
        default=None,
        description=(
            "Time in *nanoseconds* spent generating the response. "
            "Populated on the final chunk only."
        ),
    )


class PullProgress(BaseModel):
    """One streamed chunk from ``POST /api/pull``.

    Used by :meth:`~ollama_evaluator.ollama.client.OllamaClient.pull_model`
    to report progress to the runner when ``pull_missing_models`` is
    enabled (Requirement 2.4). Ollama emits a series of these chunks
    during a pull with varying ``status`` values (for example
    ``"pulling manifest"``, ``"pulling <digest>"``, ``"verifying sha256"``,
    ``"success"``); only per-layer chunks carry the ``digest``,
    ``total``, and ``completed`` fields.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        ...,
        description=(
            "Human-readable status message from the Ollama_Server "
            '(for example ``"pulling manifest"`` or ``"success"``).'
        ),
    )
    digest: str | None = Field(
        default=None,
        description=(
            "Content-addressed digest of the layer being pulled when "
            "this chunk reports per-layer progress. ``None`` on status "
            "chunks that do not reference a specific layer."
        ),
    )
    total: int | None = Field(
        default=None,
        description=(
            "Total size in bytes of the layer being pulled. ``None`` "
            "on status chunks that do not reference a specific layer."
        ),
    )
    completed: int | None = Field(
        default=None,
        description=(
            "Bytes already pulled for the layer. ``None`` on status "
            "chunks that do not reference a specific layer. When "
            "present, ``completed <= total`` holds at the Ollama_Server."
        ),
    )


def _parse_tags_entry(raw: dict[str, Any]) -> OllamaModelInfo:
    """Flatten one entry from ``GET /api/tags`` into :class:`OllamaModelInfo`.

    Handles two schema quirks of the Ollama ``/api/tags`` response:

    1. The identifier field was renamed from ``"name"`` to ``"model"``
       between v0.1.x and v0.2.x. Both are accepted; ``"name"`` wins
       when both are present because older deployments remain the more
       common setup in the field and we want deterministic behaviour
       when a mixed payload is produced by a proxy.
    2. ``parameter_size`` and ``quantization_level`` are nested under
       ``details``. This helper lifts them to the flat model so the
       rest of the Backend never reaches into ``details``.

    A missing ``details`` block is treated as absent — no error — so
    locally-imported models that do not report these fields still parse.
    Other unknown top-level keys are silently dropped via
    ``extra="ignore"`` on :class:`OllamaModelInfo`.

    Args:
        raw: One element of the ``"models"`` array from
            ``GET /api/tags``.

    Returns:
        A fully-validated :class:`OllamaModelInfo` with the identifier,
        digest, size, modification time, and ``details``-derived fields
        populated as available.

    Raises:
        pydantic.ValidationError: If the entry is missing both ``name``
            and ``model`` keys, or if any present field has a value of
            the wrong type.
    """
    # Prefer "name" when present; fall back to "model" for Ollama v0.2.x+.
    # If both are absent, leave it unset so Pydantic raises a helpful
    # ValidationError pointing at ``name`` as the missing required field.
    name = raw.get("name") if raw.get("name") is not None else raw.get("model")

    details = raw.get("details") or {}

    return OllamaModelInfo.model_validate(
        {
            "name": name,
            "digest": raw.get("digest"),
            "size": raw.get("size"),
            "modified_at": raw.get("modified_at"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
        }
    )


__all__ = [
    "GenerateChunk",
    "GenerateOptions",
    "OllamaModelInfo",
    "PullProgress",
    "_parse_tags_entry",
]
