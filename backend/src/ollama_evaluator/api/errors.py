"""Uniform error envelope for the Backend HTTP API.

Every non-2xx response the Backend emits carries a JSON body shaped like
:class:`ErrorEnvelope`: a stable ``error_code`` from :class:`ErrorCode`, a
human-readable ``message``, and an optional ``field`` dotted path that
points at the offending request field for validation errors.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Error Handling / §API error envelopes. Requirements 13.5, 13.6.

The helper :func:`http_error` is a thin wrapper around
:class:`fastapi.HTTPException` that packs an :class:`ErrorEnvelope` into
the exception's ``detail`` attribute as a plain ``dict``. FastAPI's
default exception handler returns ``detail`` verbatim as the JSON body,
which keeps the wire format under a single source of truth here.
"""

from __future__ import annotations

from enum import Enum

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(str, Enum):
    """Closed set of stable error-code tokens used in :class:`ErrorEnvelope`.

    ``str, Enum`` means each value is a ``str`` instance at runtime, so
    ``ErrorCode.run_not_found == "run_not_found"`` and Pydantic serialises
    the enum as its string value without extra configuration.
    """

    ollama_unreachable = "ollama_unreachable"
    model_not_found = "model_not_found"
    suite_not_found = "suite_not_found"
    run_not_found = "run_not_found"
    validation_failed = "validation_failed"
    suite_invalid = "suite_invalid"
    no_common_dimensions = "no_common_dimensions"
    dataset_fetch_failed = "dataset_fetch_failed"
    field_map_invalid = "field_map_invalid"
    run_timeout = "run_timeout"
    run_error = "run_error"
    metric_error = "metric_error"


class ErrorEnvelope(BaseModel):
    """JSON body returned with every non-2xx response (Requirement 13.5).

    ``field`` is populated only for validation-style failures where a
    single request field can be named (Requirement 13.6). For
    environmental or domain errors (``ollama_unreachable``,
    ``run_not_found``, ``no_common_dimensions``, ...) ``field`` is
    ``None`` and callers rely on ``error_code`` for branching.
    """

    model_config = ConfigDict(extra="forbid")

    error_code: ErrorCode = Field(
        ...,
        description="Stable error tag; one of :class:`ErrorCode`.",
    )
    message: str = Field(
        ...,
        description="Human-readable description suitable for rendering in the UI.",
    )
    field: str | None = Field(
        default=None,
        description=(
            "Dotted path of the offending request field for "
            "``validation_failed``/``suite_invalid``/``field_map_invalid`` "
            "responses. ``None`` for errors that do not name a field."
        ),
    )


def http_error(
    code: ErrorCode,
    message: str,
    *,
    status_code: int,
    field: str | None = None,
) -> HTTPException:
    """Build an :class:`HTTPException` whose ``detail`` is an envelope dict.

    FastAPI returns ``HTTPException.detail`` verbatim as the JSON body
    when the exception is raised inside a request handler. Using a
    :class:`dict` (produced via :meth:`BaseModel.model_dump`) rather
    than an :class:`ErrorEnvelope` instance keeps the dispatch path
    free of a second Pydantic serialisation pass and means the wire
    format is guaranteed to be the envelope's canonical form.
    """
    envelope = ErrorEnvelope(error_code=code, message=message, field=field)
    # ``mode="json"`` coerces the enum to its string value; FastAPI's
    # default encoder does the same under the hood, but doing it here
    # keeps the ``detail`` self-contained and avoids relying on the
    # implicit encoder.
    return HTTPException(status_code=status_code, detail=envelope.model_dump(mode="json"))


__all__ = [
    "ErrorCode",
    "ErrorEnvelope",
    "http_error",
]
