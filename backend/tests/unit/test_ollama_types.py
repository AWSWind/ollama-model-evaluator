"""Unit tests for the Ollama HTTP API wire types (Task 4.1).

Covers :mod:`ollama_evaluator.ollama.types`:

* Every model builds from a realistic Ollama JSON payload and exposes
  the declared fields (Requirements 2.5, 6.3).
* ``extra="ignore"`` silently drops unknown fields so the Backend does
  not break across Ollama_Server versions.
* :func:`_parse_tags_entry` flattens the ``details`` object, accepts
  both ``"name"`` and ``"model"`` keys (Ollama renamed the field
  between v0.1.x and v0.2.x), and tolerates a missing ``details``
  block.
* :class:`GenerateChunk` leaves the timing and token-count fields as
  ``None`` on partial (non-final) chunks, matching Requirement 6.5.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ollama_evaluator.ollama.types import (
    GenerateChunk,
    GenerateOptions,
    OllamaModelInfo,
    PullProgress,
    _parse_tags_entry,
)


# ---------------------------------------------------------------------------
# OllamaModelInfo
# ---------------------------------------------------------------------------


class TestOllamaModelInfo:
    def test_builds_from_flat_payload(self) -> None:
        """A fully populated flat payload maps 1:1 onto the model."""
        info = OllamaModelInfo.model_validate(
            {
                "name": "llama3:8b",
                "digest": "sha256:abc123",
                "size": 4_700_000_000,
                "modified_at": "2024-05-01T12:34:56Z",
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
            }
        )

        assert info.name == "llama3:8b"
        assert info.digest == "sha256:abc123"
        assert info.size == 4_700_000_000
        assert info.modified_at == datetime(2024, 5, 1, 12, 34, 56, tzinfo=timezone.utc)
        assert info.parameter_size == "8B"
        assert info.quantization_level == "Q4_0"

    def test_only_name_is_required(self) -> None:
        """Every field except ``name`` is nullable (Requirement 6.5 parallel)."""
        info = OllamaModelInfo.model_validate({"name": "mistral:7b"})

        assert info.name == "mistral:7b"
        assert info.digest is None
        assert info.size is None
        assert info.modified_at is None
        assert info.parameter_size is None
        assert info.quantization_level is None

    def test_missing_name_is_rejected(self) -> None:
        """Without ``name`` Pydantic raises — the identifier is load-bearing."""
        with pytest.raises(ValidationError):
            OllamaModelInfo.model_validate({"digest": "sha256:abc"})

    def test_ignores_unknown_fields(self) -> None:
        """``extra="ignore"`` drops fields Ollama adds in newer versions."""
        info = OllamaModelInfo.model_validate(
            {
                "name": "llama3:8b",
                # Fields that do not exist today but could in a future
                # Ollama release:
                "expires_at": "2099-01-01T00:00:00Z",
                "some_future_field": {"nested": 42},
            }
        )

        assert info.name == "llama3:8b"
        # The unknown fields must not land on the model.
        assert not hasattr(info, "expires_at")
        assert not hasattr(info, "some_future_field")


# ---------------------------------------------------------------------------
# GenerateOptions
# ---------------------------------------------------------------------------


class TestGenerateOptions:
    def test_all_fields_optional(self) -> None:
        """Empty options must validate — callers rely on ``exclude_none`` round-trips."""
        opts = GenerateOptions()

        assert opts.temperature is None
        assert opts.num_predict is None
        assert opts.stop is None

    def test_roundtrips_populated_payload(self) -> None:
        opts = GenerateOptions.model_validate(
            {
                "temperature": 0.2,
                "num_predict": 256,
                "stop": ["\n\n", "Q:"],
            }
        )

        assert opts.temperature == pytest.approx(0.2)
        assert opts.num_predict == 256
        assert opts.stop == ["\n\n", "Q:"]

    def test_ignores_unknown_options(self) -> None:
        """Ollama accepts many more options; unknown ones must not break us."""
        opts = GenerateOptions.model_validate(
            {
                "temperature": 0.0,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "mirostat": 2,
            }
        )

        assert opts.temperature == 0.0
        assert not hasattr(opts, "top_p")
        assert not hasattr(opts, "repeat_penalty")
        assert not hasattr(opts, "mirostat")


# ---------------------------------------------------------------------------
# GenerateChunk
# ---------------------------------------------------------------------------


class TestGenerateChunk:
    def test_partial_chunk_leaves_optional_fields_none(self) -> None:
        """Requirement 6.5: partial chunks do not carry timings.

        Only ``model``, ``created_at``, ``response``, ``done`` are set;
        every other field must default to ``None`` so the client can
        accumulate chunks without special-casing.
        """
        chunk = GenerateChunk.model_validate(
            {
                "model": "llama3:8b",
                "created_at": "2024-05-01T12:34:56.789Z",
                "response": "The ",
                "done": False,
            }
        )

        assert chunk.model == "llama3:8b"
        assert chunk.response == "The "
        assert chunk.done is False
        # Every timing/token-count field is None on a partial chunk.
        assert chunk.total_duration is None
        assert chunk.load_duration is None
        assert chunk.prompt_eval_count is None
        assert chunk.prompt_eval_duration is None
        assert chunk.eval_count is None
        assert chunk.eval_duration is None

    def test_final_chunk_populates_timing_fields(self) -> None:
        """The final chunk carries token counts and nanosecond durations (Req 6.3)."""
        chunk = GenerateChunk.model_validate(
            {
                "model": "llama3:8b",
                "created_at": "2024-05-01T12:34:57.000Z",
                "response": "",
                "done": True,
                "total_duration": 1_500_000_000,
                "load_duration": 100_000_000,
                "prompt_eval_count": 12,
                "prompt_eval_duration": 50_000_000,
                "eval_count": 42,
                "eval_duration": 1_350_000_000,
            }
        )

        assert chunk.done is True
        assert chunk.total_duration == 1_500_000_000
        assert chunk.load_duration == 100_000_000
        assert chunk.prompt_eval_count == 12
        assert chunk.prompt_eval_duration == 50_000_000
        assert chunk.eval_count == 42
        assert chunk.eval_duration == 1_350_000_000

    def test_final_chunk_may_omit_token_counts(self) -> None:
        """Requirement 6.5: the server may omit token counts entirely."""
        chunk = GenerateChunk.model_validate(
            {
                "model": "llama3:8b",
                "created_at": "2024-05-01T12:34:57.000Z",
                "response": "",
                "done": True,
                "total_duration": 1_500_000_000,
                # prompt_eval_count and eval_count intentionally absent.
            }
        )

        assert chunk.done is True
        assert chunk.total_duration == 1_500_000_000
        assert chunk.prompt_eval_count is None
        assert chunk.eval_count is None

    def test_ignores_unknown_fields(self) -> None:
        """New Ollama versions add fields; we must tolerate them."""
        chunk = GenerateChunk.model_validate(
            {
                "model": "llama3:8b",
                "created_at": "2024-05-01T12:34:56.789Z",
                "response": "hi",
                "done": False,
                "context": [1, 2, 3],  # Present on final chunks in some versions.
                "future_field": "x",
            }
        )

        assert chunk.response == "hi"
        assert not hasattr(chunk, "context")
        assert not hasattr(chunk, "future_field")


# ---------------------------------------------------------------------------
# PullProgress
# ---------------------------------------------------------------------------


class TestPullProgress:
    def test_status_only_chunk(self) -> None:
        """Status messages without digest/progress must validate."""
        chunk = PullProgress.model_validate({"status": "pulling manifest"})

        assert chunk.status == "pulling manifest"
        assert chunk.digest is None
        assert chunk.total is None
        assert chunk.completed is None

    def test_per_layer_chunk(self) -> None:
        chunk = PullProgress.model_validate(
            {
                "status": "pulling 4f11f4d09f0b",
                "digest": "sha256:4f11f4d09f0b",
                "total": 4_661_224_192,
                "completed": 1_230_000_000,
            }
        )

        assert chunk.status == "pulling 4f11f4d09f0b"
        assert chunk.digest == "sha256:4f11f4d09f0b"
        assert chunk.total == 4_661_224_192
        assert chunk.completed == 1_230_000_000

    def test_ignores_unknown_fields(self) -> None:
        chunk = PullProgress.model_validate(
            {
                "status": "success",
                "some_future_field": True,
            }
        )

        assert chunk.status == "success"
        assert not hasattr(chunk, "some_future_field")


# ---------------------------------------------------------------------------
# _parse_tags_entry
# ---------------------------------------------------------------------------


class TestParseTagsEntry:
    def test_accepts_name_key(self) -> None:
        """Ollama v0.1.x schema — ``"name"`` carries the tag."""
        info = _parse_tags_entry(
            {
                "name": "llama3:8b",
                "digest": "sha256:abc123",
                "size": 4_700_000_000,
                "modified_at": "2024-05-01T12:34:56Z",
                "details": {
                    "parameter_size": "8B",
                    "quantization_level": "Q4_0",
                },
            }
        )

        assert info.name == "llama3:8b"
        assert info.digest == "sha256:abc123"
        assert info.size == 4_700_000_000
        assert info.modified_at == datetime(2024, 5, 1, 12, 34, 56, tzinfo=timezone.utc)
        assert info.parameter_size == "8B"
        assert info.quantization_level == "Q4_0"

    def test_accepts_model_key(self) -> None:
        """Ollama v0.2.x+ schema — ``"model"`` is the new name."""
        info = _parse_tags_entry(
            {
                "model": "mistral:7b-instruct",
                "digest": "sha256:def456",
                "size": 3_800_000_000,
                "details": {
                    "parameter_size": "7B",
                    "quantization_level": "Q5_K_M",
                },
            }
        )

        assert info.name == "mistral:7b-instruct"
        assert info.digest == "sha256:def456"
        assert info.parameter_size == "7B"
        assert info.quantization_level == "Q5_K_M"

    def test_prefers_name_when_both_present(self) -> None:
        """Deterministic behaviour when a proxy emits both keys."""
        info = _parse_tags_entry(
            {
                "name": "llama3:8b",
                "model": "llama3:latest",  # Proxy-added alias; should lose.
                "digest": "sha256:abc",
            }
        )

        assert info.name == "llama3:8b"

    def test_handles_missing_details_block(self) -> None:
        """Locally-imported models often lack a ``details`` object."""
        info = _parse_tags_entry(
            {
                "name": "my-finetune:latest",
                "digest": "sha256:zzz",
                "size": 123_456,
            }
        )

        assert info.name == "my-finetune:latest"
        assert info.parameter_size is None
        assert info.quantization_level is None

    def test_handles_explicit_null_details(self) -> None:
        """``details: null`` is equivalent to absent — both must parse."""
        info = _parse_tags_entry(
            {
                "name": "my-finetune:latest",
                "details": None,
            }
        )

        assert info.name == "my-finetune:latest"
        assert info.parameter_size is None
        assert info.quantization_level is None

    def test_handles_partial_details(self) -> None:
        """``details`` may carry only some of the lifted fields."""
        info = _parse_tags_entry(
            {
                "name": "phi:2.7b",
                "details": {"parameter_size": "2.7B"},
            }
        )

        assert info.name == "phi:2.7b"
        assert info.parameter_size == "2.7B"
        assert info.quantization_level is None

    def test_ignores_unknown_top_level_and_details_fields(self) -> None:
        """Newer Ollama versions add fields at both levels; both must be dropped."""
        info = _parse_tags_entry(
            {
                "name": "llama3:8b",
                "expires_at": "2099-01-01T00:00:00Z",
                "details": {
                    "parameter_size": "8B",
                    "quantization_level": "Q4_0",
                    "format": "gguf",  # Present in some Ollama versions.
                    "family": "llama",  # Likewise.
                },
            }
        )

        assert info.name == "llama3:8b"
        assert info.parameter_size == "8B"
        assert info.quantization_level == "Q4_0"
        assert not hasattr(info, "expires_at")
        assert not hasattr(info, "format")
        assert not hasattr(info, "family")

    def test_missing_name_and_model_raises(self) -> None:
        """Without an identifier the entry is useless — validation must fail."""
        with pytest.raises(ValidationError):
            _parse_tags_entry({"digest": "sha256:abc"})

    def test_name_key_null_falls_back_to_model(self) -> None:
        """Explicit ``name: null`` is treated as absent, so ``model`` wins."""
        info = _parse_tags_entry(
            {
                "name": None,
                "model": "llama3:8b",
            }
        )

        assert info.name == "llama3:8b"
