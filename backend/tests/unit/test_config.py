"""Unit tests for the Config_File / RunConfig Pydantic models (Task 2.2).

These tests verify the invariants defined in
``.kiro/specs/ollama-model-evaluator/design.md`` §Data Models and the
corresponding Requirements:

* Default values match the design document for every field.
* ``extra="forbid"`` rejects unknown top-level keys on both models
  (consistent with ``suites.models``).
* Numeric bounds are enforced with helpful error messages:
  ``repetitions >= 1`` (Req 5.2), ``concurrency >= 1`` (Req 5.5),
  ``retry_max_attempts >= 0`` (Req 11.1), ``ollama_timeout_s > 0`` (Req
  1.4).
* ``models`` and ``suites`` are non-empty (Req 2.2, 3.1).
* ``dataset_mode`` only accepts ``"local"`` / ``"remote"`` (Req 17.3).
* ``log_level`` only accepts the four enumerated values (Req 10.6).
* ``ollama_base_url`` must be an absolute ``http(s)`` URL (Req 1.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ollama_evaluator.config import ConfigFile, RunConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_config(**overrides: object) -> RunConfig:
    """Build a minimal valid ``RunConfig`` with the given field overrides."""
    defaults: dict[str, object] = {
        "models": ["llama3:8b"],
        "suites": ["reasoning-basics"],
    }
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def _config_file(**overrides: object) -> ConfigFile:
    """Build a minimal valid ``ConfigFile`` with the given field overrides."""
    defaults: dict[str, object] = {
        "suites_dir": Path("./suites"),
        "run": _run_config(),
    }
    defaults.update(overrides)
    return ConfigFile(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RunConfig defaults
# ---------------------------------------------------------------------------


class TestRunConfigDefaults:
    def test_defaults_match_design(self) -> None:
        """Every default must line up with design.md §Data Models."""
        cfg = _run_config()
        assert cfg.models == ["llama3:8b"]
        assert cfg.suites == ["reasoning-basics"]
        assert cfg.repetitions == 1  # Req 5.2
        assert cfg.concurrency == 1  # Req 5.5
        assert cfg.pull_missing_models is False  # Req 2.4
        assert cfg.retry_max_attempts == 2  # Req 11.1
        assert cfg.judge_model is None
        assert cfg.tag_filter == []  # Req 3.6
        assert cfg.ollama_timeout_s == 120.0  # Req 1.4


# ---------------------------------------------------------------------------
# RunConfig validators
# ---------------------------------------------------------------------------


class TestRunConfigValidators:
    def test_rejects_empty_models(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_config(models=[])
        message = str(excinfo.value)
        assert "models" in message
        assert "at least one" in message

    def test_rejects_empty_suites(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_config(suites=[])
        message = str(excinfo.value)
        assert "suites" in message
        assert "at least one" in message

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_rejects_non_positive_repetitions(self, bad_value: int) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_config(repetitions=bad_value)
        assert "repetitions" in str(excinfo.value)

    @pytest.mark.parametrize("bad_value", [0, -1, -5])
    def test_rejects_non_positive_concurrency(self, bad_value: int) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_config(concurrency=bad_value)
        assert "concurrency" in str(excinfo.value)

    @pytest.mark.parametrize("bad_value", [-1, -2, -100])
    def test_rejects_negative_retry_max_attempts(self, bad_value: int) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_config(retry_max_attempts=bad_value)
        assert "retry_max_attempts" in str(excinfo.value)

    def test_retry_max_attempts_zero_is_allowed(self) -> None:
        """A value of 0 means 'no retries' and must be accepted."""
        cfg = _run_config(retry_max_attempts=0)
        assert cfg.retry_max_attempts == 0

    @pytest.mark.parametrize("bad_value", [0, 0.0, -0.1, -1.0])
    def test_rejects_non_positive_timeout(self, bad_value: float) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _run_config(ollama_timeout_s=bad_value)
        assert "ollama_timeout_s" in str(excinfo.value)

    def test_accepts_small_positive_timeout(self) -> None:
        cfg = _run_config(ollama_timeout_s=0.001)
        assert cfg.ollama_timeout_s == pytest.approx(0.001)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RunConfig.model_validate(
                {
                    "models": ["m"],
                    "suites": ["s"],
                    "typo_field": "boom",
                }
            )
        assert "typo_field" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ConfigFile defaults
# ---------------------------------------------------------------------------


class TestConfigFileDefaults:
    def test_defaults_match_design(self) -> None:
        cfg = _config_file()
        assert cfg.ollama_base_url == "http://localhost:11434"  # Req 1.1
        assert cfg.suites_dir == Path("./suites")
        assert cfg.output_dir == Path("./runs")
        assert cfg.log_level == "info"  # Req 10.6
        assert cfg.dataset_mode == "local"  # Req 17.3
        assert cfg.hf_cache_dir is None
        assert isinstance(cfg.run, RunConfig)

    def test_output_dir_override_respected(self) -> None:
        cfg = _config_file(output_dir=Path("/tmp/my-runs"))
        assert cfg.output_dir == Path("/tmp/my-runs")

    def test_hf_cache_dir_accepts_path(self) -> None:
        cfg = _config_file(hf_cache_dir=Path("/var/cache/hf"))
        assert cfg.hf_cache_dir == Path("/var/cache/hf")


# ---------------------------------------------------------------------------
# ConfigFile validators
# ---------------------------------------------------------------------------


class TestConfigFileValidators:
    def test_suites_dir_required(self) -> None:
        """``suites_dir`` has no default; the loader must provide one."""
        with pytest.raises(ValidationError) as excinfo:
            ConfigFile.model_validate({"run": _run_config().model_dump()})
        assert "suites_dir" in str(excinfo.value)

    def test_run_required(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ConfigFile.model_validate({"suites_dir": "./suites"})
        assert "run" in str(excinfo.value)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ConfigFile.model_validate(
                {
                    "suites_dir": "./suites",
                    "run": _run_config().model_dump(),
                    "surprise_field": "nope",
                }
            )
        assert "surprise_field" in str(excinfo.value)

    @pytest.mark.parametrize(
        "good_url",
        [
            "http://localhost:11434",
            "https://ollama.example.com",
            "http://127.0.0.1:8080",
            "https://ollama.example.com:443/v1",
        ],
    )
    def test_accepts_valid_http_base_urls(self, good_url: str) -> None:
        cfg = _config_file(ollama_base_url=good_url)
        assert cfg.ollama_base_url == good_url

    @pytest.mark.parametrize(
        "bad_url",
        [
            "",
            "localhost:11434",
            "ftp://localhost:11434",
            "file:///tmp/ollama",
            "ollama.example.com",
        ],
    )
    def test_rejects_non_http_base_urls(self, bad_url: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _config_file(ollama_base_url=bad_url)
        assert "ollama_base_url" in str(excinfo.value)

    @pytest.mark.parametrize("mode", ["local", "remote"])
    def test_dataset_mode_accepts_both_values(self, mode: str) -> None:
        cfg = _config_file(dataset_mode=mode)
        assert cfg.dataset_mode == mode

    @pytest.mark.parametrize("bad_mode", ["", "offline", "online", "LOCAL", "Remote"])
    def test_dataset_mode_rejects_other_values(self, bad_mode: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _config_file(dataset_mode=bad_mode)
        assert "dataset_mode" in str(excinfo.value)

    @pytest.mark.parametrize("level", ["debug", "info", "warn", "error"])
    def test_log_level_accepts_enumerated_values(self, level: str) -> None:
        cfg = _config_file(log_level=level)
        assert cfg.log_level == level

    @pytest.mark.parametrize(
        "bad_level",
        ["", "trace", "warning", "critical", "INFO", "Debug"],
    )
    def test_log_level_rejects_other_values(self, bad_level: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _config_file(log_level=bad_level)
        assert "log_level" in str(excinfo.value)
