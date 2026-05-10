"""Pydantic data models for the user-supplied Config_File and Run configuration.

This module defines two top-level models that together describe a single Run:

* :class:`RunConfig` — the run-level knobs (which models to evaluate, which
  suites to run, how many repetitions, how much concurrency, and so on).
  ``RunConfig`` is what the Backend_API accepts when a client POSTs to
  ``/api/runs`` and is also the ``run:`` section of the on-disk Config_File.
* :class:`ConfigFile` — the full on-disk Config_File loaded via the ``--config``
  CLI flag. It wraps a :class:`RunConfig` and adds process-wide settings
  (Ollama base URL, where to discover suites, where to write reports, logging
  verbosity, and dataset-sourcing mode).

Both models forbid unknown fields (``ConfigDict(extra="forbid")``) so that
typos in user-authored YAML/JSON are reported up front rather than being
silently ignored (aligned with the suite loader in Requirement 3.5).

Default values and validation bounds come directly from the Data Models
section of ``.kiro/specs/ollama-model-evaluator/design.md`` and from the
Requirements document:

* ``ollama_base_url`` default — Requirement 1.1.
* ``ollama_timeout_s`` default (120 s) and ``> 0`` bound — Requirement 1.4.
* ``repetitions`` default and ``>= 1`` bound — Requirement 5.2.
* ``concurrency`` default and ``>= 1`` bound — Requirement 5.5.
* ``pull_missing_models`` default — Requirement 2.4.
* ``retry_max_attempts`` default (2 additional attempts) and ``>= 0`` bound
  — Requirement 11.1.
* ``log_level`` enumeration — Requirement 10.6.
* ``dataset_mode`` enumeration and default — Requirement 17.3.
* ``tag_filter`` semantics — Requirement 3.6.

This module deliberately does **not** import from ``ollama_evaluator.suites``
to avoid a circular import: ``suites/loader.py`` will eventually consume
``ConfigFile`` to discover suite files, and importing suite models here would
introduce a dependency in the wrong direction. ``RunConfig.defaults`` is
intentionally omitted from this task and will be added alongside the runner
in a later task once the ``GenerationDefaults`` import direction is settled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    """Run-level configuration for a single Run.

    ``RunConfig`` is the body the Backend_API accepts for
    ``POST /api/runs`` and is also embedded in :class:`ConfigFile` as the
    ``run:`` section. A ``RunConfig`` is serialised verbatim into every
    ``RunReport`` so runs are reproducible from their report alone
    (Requirement 8.4).

    Validators enforce the numeric ranges specified in the Requirements
    document and reject empty ``models`` / ``suites`` lists early, before
    any Ollama_Server call is made.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(
        ...,
        description=(
            "Ollama model tags to evaluate in this Run (e.g. 'llama3:8b'). "
            "Must contain at least one model (Requirement 2.2)."
        ),
    )
    suites: list[str] = Field(
        ...,
        description=(
            "Names of Evaluation_Suites to run. Must contain at least one "
            "name (Requirement 3.1)."
        ),
    )
    repetitions: int = Field(
        default=1,
        description=(
            "Number of times each Test_Case is executed per Model. Must be "
            ">= 1 (Requirement 5.2)."
        ),
    )
    concurrency: int = Field(
        default=1,
        description=(
            "Maximum number of in-flight Ollama_Server requests during this "
            "Run. Must be >= 1 (Requirement 5.5)."
        ),
    )
    pull_missing_models: bool = Field(
        default=False,
        description=(
            "If true, instruct the Ollama_Server to pull any requested Model "
            "that is not already available (Requirement 2.4)."
        ),
    )
    retry_max_attempts: int = Field(
        default=2,
        description=(
            "Number of *additional* retry attempts per Test_Case execution "
            "on retryable network errors (Requirement 11.1). Must be >= 0. "
            "A value of 0 disables retries entirely."
        ),
    )
    judge_model: str | None = Field(
        default=None,
        description=(
            "Ollama model tag used by any configured ``llm-as-judge`` "
            "metric. ``None`` means no judge model is configured; metrics "
            "that require one will raise a validation error at Run start."
        ),
    )
    tag_filter: list[str] = Field(
        default_factory=list,
        description=(
            "Optional Test_Case tag filter (Requirement 3.6). When empty, "
            "every Test_Case in the selected suites is included. When "
            "non-empty, only Test_Cases whose ``tags`` intersect this list "
            "are selected."
        ),
    )
    ollama_timeout_s: float = Field(
        default=120.0,
        description=(
            "Per-request timeout in seconds applied to every Ollama_Server "
            "call made during this Run (Requirement 1.4). Must be > 0."
        ),
    )

    @field_validator("models")
    @classmethod
    def _models_non_empty(cls, value: list[str]) -> list[str]:
        """Reject an empty ``models`` list early (Requirement 2.2)."""
        if len(value) == 0:
            raise ValueError("RunConfig.models must contain at least one model name")
        return value

    @field_validator("suites")
    @classmethod
    def _suites_non_empty(cls, value: list[str]) -> list[str]:
        """Reject an empty ``suites`` list early (Requirement 3.1)."""
        if len(value) == 0:
            raise ValueError("RunConfig.suites must contain at least one suite name")
        return value

    @field_validator("repetitions")
    @classmethod
    def _repetitions_positive(cls, value: int) -> int:
        """Enforce ``repetitions >= 1`` (Requirement 5.2)."""
        if value < 1:
            raise ValueError(
                f"RunConfig.repetitions must be >= 1 (got {value})"
            )
        return value

    @field_validator("concurrency")
    @classmethod
    def _concurrency_positive(cls, value: int) -> int:
        """Enforce ``concurrency >= 1`` (Requirement 5.5)."""
        if value < 1:
            raise ValueError(
                f"RunConfig.concurrency must be >= 1 (got {value})"
            )
        return value

    @field_validator("retry_max_attempts")
    @classmethod
    def _retry_non_negative(cls, value: int) -> int:
        """Enforce ``retry_max_attempts >= 0`` (Requirement 11.1).

        A value of 0 is explicitly allowed and means "do not retry"; it
        differs from the default of 2 (two additional attempts).
        """
        if value < 0:
            raise ValueError(
                f"RunConfig.retry_max_attempts must be >= 0 (got {value})"
            )
        return value

    @field_validator("ollama_timeout_s")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        """Enforce ``ollama_timeout_s > 0`` (Requirement 1.4).

        A zero or negative timeout would either fail every request
        immediately or be interpreted inconsistently by ``httpx``; either
        way it is not what a user means by "timeout".
        """
        if value <= 0:
            raise ValueError(
                f"RunConfig.ollama_timeout_s must be > 0 (got {value})"
            )
        return value


class ConfigFile(BaseModel):
    """The on-disk Config_File loaded via the ``--config`` CLI flag.

    A ``ConfigFile`` carries process-wide settings (Ollama base URL, where
    to discover suites, where to write reports, logging verbosity, and
    dataset-sourcing defaults) plus an embedded :class:`RunConfig`
    describing the Run to execute.

    ``hf_cache_dir`` defaults to ``None`` so the Backend can fall back to
    the HuggingFace ``datasets`` library's own cache resolution (``$HF_HOME``
    or ``~/.cache/huggingface``) without the Config_File having to encode
    platform-specific paths.
    """

    model_config = ConfigDict(extra="forbid")

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Base URL of the Ollama_Server (Requirement 1.1). Must be an "
            "absolute ``http://`` or ``https://`` URL."
        ),
    )
    suites_dir: Path = Field(
        ...,
        description=(
            "Directory containing Evaluation_Suite files to discover at Run "
            "start (Requirement 3.1). Required; there is no sensible "
            "default because evaluation content is user-supplied."
        ),
    )
    output_dir: Path = Field(
        default=Path("./runs"),
        description=(
            "Directory where Run_Reports (``runs/{run_id}/report.json`` and "
            "``report.md``) are written (Requirement 8.1)."
        ),
    )
    log_level: Literal["debug", "info", "warn", "error"] = Field(
        default="info",
        description=(
            "Global log verbosity (Requirement 10.6). The CLI ``--log-level`` "
            "flag overrides this value at runtime."
        ),
    )
    dataset_mode: Literal["local", "remote"] = Field(
        default="local",
        description=(
            "Default ``dataset_mode`` for adapter-backed Evaluation_Suites "
            "(Requirement 17.3). Individual suite files may override this "
            "per-suite. ``local`` means read from disk; ``remote`` means "
            "stream from the HuggingFace Hub at Run time."
        ),
    )
    hf_cache_dir: Path | None = Field(
        default=None,
        description=(
            "Directory used by the HuggingFace ``datasets`` library for its "
            "on-disk cache. When ``None`` the library's own default is used "
            "(``$HF_HOME`` or ``~/.cache/huggingface``)."
        ),
    )
    run: RunConfig = Field(
        ...,
        description="Run-level configuration (models, suites, concurrency, ...).",
    )

    @field_validator("ollama_base_url")
    @classmethod
    def _ollama_base_url_is_http(cls, value: str) -> str:
        """Require an absolute ``http://`` or ``https://`` URL.

        A simple ``startswith`` check is deliberate: we don't want to pull
        in a full URL parser just for this validation, and the Ollama
        client in ``ollama/client.py`` will perform deeper validation when
        it tries to open a connection. This check catches the common
        mistakes (empty string, missing scheme, typo'd scheme) and does so
        with a message the user can act on.
        """
        if not isinstance(value, str) or not (
            value.startswith("http://") or value.startswith("https://")
        ):
            raise ValueError(
                "ConfigFile.ollama_base_url must be an absolute http(s) URL "
                f"(got {value!r})"
            )
        return value


__all__ = [
    "ConfigFile",
    "RunConfig",
]
