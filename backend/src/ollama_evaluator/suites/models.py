"""Pydantic data models for Evaluation Suites.

This module defines the user-facing data contracts for authoring evaluation
suites:

* ``MetricConfig`` — configuration for a single scoring metric attached to a
  ``TestCase``. Metric-specific parameters (e.g. ``pattern`` for a regex
  metric, ``schema`` for a JSON-schema metric) are carried in a nested
  ``params`` mapping so that the enclosing model can still enforce
  ``extra="forbid"`` at the suite/test-case level.
* ``GenerationDefaults`` — run-level default generation parameters that a
  ``TestCase`` may override.
* ``TestCase`` — a single evaluation unit (prompt + metrics + optional
  metadata).
* ``EvaluationSuite`` — a named, versioned collection of ``TestCase`` objects.

All models forbid unknown fields (``ConfigDict(extra="forbid")``) so that
typos in user-authored YAML/JSON are reported up front with a useful error
message (Requirement 3.5).

Design references: `.kiro/specs/ollama-model-evaluator/design.md`
§Components / §Data Models. Requirements 3.3 and 3.4 are the direct drivers
for the validators in this module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MetricConfig(BaseModel):
    """Configuration for a single metric attached to a :class:`TestCase`.

    Different built-in metrics accept different parameters (``exact-match``
    takes ``case_sensitive`` / ``trim``; ``regex-match`` takes ``pattern`` /
    ``flags``; ``json-schema-valid`` takes ``schema``; ``length-range`` takes
    ``min`` / ``max``; ``llm-as-judge`` takes ``rubric`` / ``judge_model``;
    and so on). Rather than enumerating every metric's schema at the suite
    layer, ``MetricConfig`` exposes a ``name`` plus a generic ``params``
    mapping. The concrete metric implementations (``metrics/builtin.py``,
    ``metrics/judge.py``) are responsible for validating the contents of
    ``params`` themselves.

    ``extra="forbid"`` is still applied here — the only two top-level fields
    permitted are ``name`` and ``params``. This preserves the spec rule that
    every suite model rejects unknown fields while still letting metrics
    carry arbitrary structured configuration.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Identifier of a registered metric (e.g. 'exact-match').",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metric-specific parameters passed verbatim to the metric "
            "implementation. Validated by the metric itself, not here."
        ),
    )

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        """Reject blank metric names early with a clear message."""
        if not value or not value.strip():
            raise ValueError("MetricConfig.name must be a non-empty string")
        return value


class GenerationDefaults(BaseModel):
    """Run-level defaults for Ollama generation parameters.

    A ``TestCase`` may override any of these fields individually. When a
    ``TestCase`` leaves a field unset (``None``), the runner falls back to
    the corresponding value from ``GenerationDefaults`` (see Requirement 5.4
    and Property 7 in the design document).
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(
        default=0.0,
        description="Sampling temperature passed to Ollama (``options.temperature``).",
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum tokens to generate (``options.num_predict``). "
            "``None`` means use the model's default."
        ),
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="Default ``options.stop`` sequences applied to every Test_Case.",
    )


class TestCase(BaseModel):
    """A single evaluation unit within an :class:`EvaluationSuite`.

    Each test case carries a prompt to send to the model, optional metadata
    (system prompt, expected output, reference data, tags), optional
    per-case generation overrides, and a non-empty list of
    :class:`MetricConfig` entries describing how the response should be
    scored.

    Validators enforce:

    * ``prompt`` is non-empty (Requirement 3.3).
    * ``metrics`` contains at least one entry (Requirement 3.3).

    Unknown fields are rejected by ``ConfigDict(extra="forbid")`` so typos
    in user-authored suites surface as validation errors rather than being
    silently ignored (Requirement 3.5).
    """

    # Tell pytest not to treat this class as a test suite. Pydantic v2 leaves
    # un-annotated class-level assignments alone, so this does not become a
    # model field.
    __test__ = False

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Stable identifier, unique within the enclosing suite.",
    )
    prompt: str = Field(
        ...,
        description="User prompt sent to the Ollama model. Must be non-empty.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt prepended to the request.",
    )
    expected_output: str | None = Field(
        default=None,
        description="Optional canonical answer used by exact/regex metrics.",
    )
    reference_data: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary structured context consumed by specific metrics.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for filtering at Run time (Requirement 3.6).",
    )
    temperature: float | None = Field(
        default=None,
        description=(
            "Per-case sampling temperature override. ``None`` means fall "
            "back to the suite/run defaults."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        description="Per-case ``num_predict`` override. ``None`` means use defaults.",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description=(
            "Per-case ``stop`` override. ``None`` (not ``[]``) means fall "
            "back to the run-level defaults; ``[]`` explicitly clears them."
        ),
    )
    metrics: list[MetricConfig] = Field(
        ...,
        description="Scoring configuration. Must contain at least one entry.",
    )

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        """Disallow empty / whitespace-only test-case ids."""
        if not value or not value.strip():
            raise ValueError("TestCase.id must be a non-empty string")
        return value

    @field_validator("prompt")
    @classmethod
    def _prompt_non_empty(cls, value: str) -> str:
        """Enforce Requirement 3.3: ``prompt`` is required and non-empty."""
        if value == "":
            raise ValueError("TestCase.prompt must be a non-empty string")
        return value

    @field_validator("metrics")
    @classmethod
    def _metrics_non_empty(cls, value: list[MetricConfig]) -> list[MetricConfig]:
        """Enforce Requirement 3.3: at least one metric per test case."""
        if len(value) == 0:
            raise ValueError("TestCase.metrics must contain at least one MetricConfig")
        return value


class EvaluationSuite(BaseModel):
    """A named, versioned collection of :class:`TestCase` objects.

    A suite is the primary on-disk artifact authored by users (YAML or JSON;
    see ``suites/loader.py``) and the unit of filtering at Run time
    (Requirement 3.6). Suites must have a non-empty ``name``, a non-empty
    ``test_cases`` list, and every ``test_cases[i].id`` must be unique
    within the suite (Requirement 3.3).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Human-readable suite name, used by ``RunConfig.suites``.",
    )
    version: str = Field(
        default="1.0",
        description="Authoring version string; informational only in v1.",
    )
    description: str | None = Field(
        default=None,
        description="Optional free-form description shown in the UI.",
    )
    defaults: GenerationDefaults = Field(
        default_factory=GenerationDefaults,
        description="Default generation parameters for every TestCase in this suite.",
    )
    test_cases: list[TestCase] = Field(
        ...,
        description="Ordered, non-empty list of test cases with unique ids.",
    )

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        """Reject blank suite names; they are used as identifiers elsewhere."""
        if not value or not value.strip():
            raise ValueError("EvaluationSuite.name must be a non-empty string")
        return value

    @field_validator("test_cases")
    @classmethod
    def _test_cases_non_empty(cls, value: list[TestCase]) -> list[TestCase]:
        """A suite with zero test cases cannot produce any Run results."""
        if len(value) == 0:
            raise ValueError(
                "EvaluationSuite.test_cases must contain at least one TestCase"
            )
        return value

    @model_validator(mode="after")
    def _unique_test_case_ids(self) -> EvaluationSuite:
        """Enforce uniqueness of ``test_cases[i].id`` within the suite.

        The error message names the first duplicate id (in document order)
        so the suite loader can surface it verbatim to the user (Requirement
        3.5).
        """
        seen: set[str] = set()
        for tc in self.test_cases:
            if tc.id in seen:
                raise ValueError(
                    f"EvaluationSuite.test_cases contains duplicate id: {tc.id!r}"
                )
            seen.add(tc.id)
        return self


__all__ = [
    "EvaluationSuite",
    "GenerationDefaults",
    "MetricConfig",
    "TestCase",
]
