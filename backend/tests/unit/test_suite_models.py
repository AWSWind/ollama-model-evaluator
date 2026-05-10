"""Unit tests for the Evaluation Suite Pydantic models (Task 2.1).

These tests cover the model-level invariants enforced by
``ollama_evaluator.suites.models``:

* Unique ``TestCase.id`` within a suite (Requirement 3.3).
* Non-empty ``TestCase.prompt`` (Requirement 3.3).
* Non-empty ``TestCase.metrics`` (Requirement 3.3).
* ``extra="forbid"`` on every user-facing container model (Requirement 3.4).
* ``MetricConfig.params`` accepts arbitrary metric-specific parameters so
  different metrics can carry their own configuration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metric(name: str = "exact-match", **params: object) -> MetricConfig:
    return MetricConfig(name=name, params=dict(params))


def _test_case(
    id_: str = "case-1",
    prompt: str = "What is 2 + 2?",
    metrics: list[MetricConfig] | None = None,
) -> TestCase:
    return TestCase(
        id=id_,
        prompt=prompt,
        metrics=metrics if metrics is not None else [_metric()],
    )


# ---------------------------------------------------------------------------
# MetricConfig
# ---------------------------------------------------------------------------


class TestMetricConfig:
    def test_accepts_metric_specific_params(self) -> None:
        """``MetricConfig.params`` must allow arbitrary metric-specific keys."""
        regex_cfg = MetricConfig(
            name="regex-match",
            params={"pattern": r"^\s*([ABCD])\b", "flags": "i"},
        )
        schema_cfg = MetricConfig(
            name="json-schema-valid",
            params={"schema": {"type": "object", "required": ["answer"]}},
        )
        length_cfg = MetricConfig(
            name="length-range",
            params={"min": 1, "max": 100},
        )

        assert regex_cfg.params["pattern"] == r"^\s*([ABCD])\b"
        assert regex_cfg.params["flags"] == "i"
        assert schema_cfg.params["schema"]["required"] == ["answer"]
        assert length_cfg.params == {"min": 1, "max": 100}

    def test_params_defaults_to_empty_dict(self) -> None:
        cfg = MetricConfig(name="exact-match")
        assert cfg.params == {}

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricConfig(name="")
        assert "name" in str(excinfo.value).lower()

    def test_rejects_whitespace_only_name(self) -> None:
        with pytest.raises(ValidationError):
            MetricConfig(name="   ")

    def test_extra_top_level_fields_forbidden(self) -> None:
        """Only ``name`` and ``params`` are allowed at the top level."""
        with pytest.raises(ValidationError) as excinfo:
            MetricConfig.model_validate(
                {"name": "exact-match", "pattern": "oops-should-be-in-params"}
            )
        message = str(excinfo.value)
        assert "pattern" in message
        assert "extra" in message.lower() or "not permitted" in message.lower()


# ---------------------------------------------------------------------------
# GenerationDefaults
# ---------------------------------------------------------------------------


class TestGenerationDefaults:
    def test_defaults_match_design(self) -> None:
        defaults = GenerationDefaults()
        assert defaults.temperature == 0.0
        assert defaults.max_tokens is None
        assert defaults.stop_sequences == []

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            GenerationDefaults.model_validate(
                {"temperature": 0.2, "top_p": 0.9}  # top_p is not supported in v1
            )
        assert "top_p" in str(excinfo.value)


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------


class TestTestCase:
    def test_minimal_valid_test_case(self) -> None:
        tc = _test_case()
        assert tc.id == "case-1"
        assert tc.prompt == "What is 2 + 2?"
        assert len(tc.metrics) == 1
        # Optional fields default to None / [].
        assert tc.system_prompt is None
        assert tc.expected_output is None
        assert tc.reference_data is None
        assert tc.tags == []
        assert tc.temperature is None
        assert tc.max_tokens is None
        assert tc.stop_sequences is None

    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TestCase(id="c1", prompt="", metrics=[_metric()])
        message = str(excinfo.value)
        assert "prompt" in message
        assert "non-empty" in message

    def test_rejects_empty_metrics(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TestCase(id="c1", prompt="Hi", metrics=[])
        message = str(excinfo.value)
        assert "metrics" in message
        assert "at least one" in message

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            TestCase(id="", prompt="Hi", metrics=[_metric()])

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TestCase.model_validate(
                {
                    "id": "c1",
                    "prompt": "Hi",
                    "metrics": [{"name": "exact-match"}],
                    "typo_field": "boom",
                }
            )
        assert "typo_field" in str(excinfo.value)


# ---------------------------------------------------------------------------
# EvaluationSuite
# ---------------------------------------------------------------------------


class TestEvaluationSuite:
    def test_minimal_valid_suite(self) -> None:
        suite = EvaluationSuite(
            name="reasoning-basics",
            test_cases=[_test_case("c1"), _test_case("c2")],
        )
        assert suite.name == "reasoning-basics"
        assert suite.version == "1.0"
        assert suite.description is None
        assert isinstance(suite.defaults, GenerationDefaults)
        assert [tc.id for tc in suite.test_cases] == ["c1", "c2"]

    def test_rejects_empty_test_cases(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            EvaluationSuite(name="s", test_cases=[])
        assert "at least one" in str(excinfo.value)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationSuite(name="", test_cases=[_test_case("c1")])

    def test_duplicate_test_case_ids_rejected_with_named_duplicate(self) -> None:
        """Requirement 3.3: ids must be unique within a suite.

        The error message must name the first duplicate id so the loader
        can include it in the user-facing validation error (Requirement
        3.5).
        """
        duplicate_id = "case-dup"
        with pytest.raises(ValidationError) as excinfo:
            EvaluationSuite(
                name="dupes",
                test_cases=[
                    _test_case(duplicate_id),
                    _test_case("case-2"),
                    _test_case(duplicate_id),  # duplicate
                ],
            )
        message = str(excinfo.value)
        assert "duplicate" in message
        assert repr(duplicate_id) in message or duplicate_id in message

    def test_duplicate_detection_reports_first_duplicate(self) -> None:
        """When multiple ids collide, the first collision is reported."""
        with pytest.raises(ValidationError) as excinfo:
            EvaluationSuite(
                name="dupes",
                test_cases=[
                    _test_case("a"),
                    _test_case("b"),
                    _test_case("a"),  # first duplicate in document order
                    _test_case("b"),  # also a duplicate, but second
                ],
            )
        message = str(excinfo.value)
        assert "'a'" in message
        # The less-specific collision should not be the one reported.
        assert "'b'" not in message

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            EvaluationSuite.model_validate(
                {
                    "name": "s",
                    "test_cases": [
                        {
                            "id": "c1",
                            "prompt": "Hi",
                            "metrics": [{"name": "exact-match"}],
                        }
                    ],
                    "mystery_field": 42,
                }
            )
        assert "mystery_field" in str(excinfo.value)
