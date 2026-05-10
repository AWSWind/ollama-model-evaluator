"""Feature: ollama-model-evaluator, Property 20: CLI exit code ↔ failure presence.

The CLI's ``run`` subcommand exits with code ``0`` iff every
:class:`TestCaseResult.status == 'pass'``; otherwise the exit code is
non-zero. Preflight errors surface as exit code ``2``.

Validates: Requirements 10.2, 10.3.

Approach: drive the CLI's ``run`` subcommand through
:class:`typer.testing.CliRunner` against a :class:`FakeOllamaClient`
(monkey-patched into :mod:`ollama_evaluator.cli`). Hypothesis varies
the number of test cases and a per-case outcome bitmap; the expected
exit code is derived directly from the bitmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from ollama_evaluator.metrics import register_metric
from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.models import MetricResult as MResult
from ollama_evaluator.ollama.types import OllamaModelInfo
from ollama_evaluator.suites.writer import dump_suite
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    MetricConfig,
    TestCase,
)
from tests.unit._fakes import FakeOllamaClient, make_chunks


class _ConfigurablePassMetric:
    """Metric that passes or fails based on whether the response contains ``PASS``."""

    name = "p20-configurable"

    async def score(self, response: str, ctx: MetricContext) -> MResult:
        passed = "PASS" in response
        return MResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            threshold=1.0,
            details={},
        )


register_metric(_ConfigurablePassMetric())


@pytest.fixture
def cli_workspace(tmp_path: Path):
    """Set up a minimal ConfigFile + suites directory for the CLI."""
    suites_dir = tmp_path / "suites"
    output_dir = tmp_path / "runs"
    suites_dir.mkdir()
    output_dir.mkdir()
    return {
        "root": tmp_path,
        "suites_dir": suites_dir,
        "output_dir": output_dir,
    }


def _install_fake_client(monkeypatch, outcomes: list[bool]) -> FakeOllamaClient:
    """Monkey-patch :class:`OllamaClient` in :mod:`ollama_evaluator.cli`.

    The returned :class:`FakeOllamaClient` is installed as the class
    the CLI instantiates; ``outcomes`` drives which generate calls
    emit ``"PASS"`` vs ``"FAIL"``.
    """
    fake = FakeOllamaClient()
    fake.set_version("0.2")
    fake.set_models([OllamaModelInfo(name="m1")])

    call_index = [0]

    def _chunk_factory(model, prompt, system, options):
        idx = call_index[0]
        call_index[0] += 1
        text = "PASS" if idx < len(outcomes) and outcomes[idx] else "FAIL"
        return make_chunks(text)

    fake.set_generate_chunks(_chunk_factory)

    import ollama_evaluator.cli as cli_module

    def _factory(base_url: str, timeout_s: float = 120.0, **kwargs) -> FakeOllamaClient:
        return fake

    monkeypatch.setattr(cli_module, "OllamaClient", _factory)
    return fake


def _write_suite(suites_dir: Path, n_cases: int) -> None:
    cases = [
        TestCase(
            id=f"tc{i}",
            prompt=f"p{i}",
            metrics=[MetricConfig(name="p20-configurable")],
        )
        for i in range(n_cases)
    ]
    suite = EvaluationSuite(name="suite", test_cases=cases)
    (suites_dir / "suite.yaml").write_text(dump_suite(suite, "yaml"), encoding="utf-8")


def _write_config(root: Path, suites_dir: Path, output_dir: Path) -> Path:
    path = root / "config.yaml"
    path.write_text(
        f"""
suites_dir: {suites_dir}
output_dir: {output_dir}
ollama_base_url: http://localhost:11434
run:
  models: [m1]
  suites: [suite]
  repetitions: 1
  concurrency: 1
  retry_max_attempts: 0
""",
        encoding="utf-8",
    )
    return path


@given(
    outcomes=st.lists(st.booleans(), min_size=1, max_size=3),
)
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_cli_exit_code_matches_failure_presence(
    cli_workspace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[bool],
) -> None:
    """**Validates: Requirements 10.2, 10.3**

    Exit code is 0 iff every test case passed.
    """
    from ollama_evaluator.cli import app

    _install_fake_client(monkeypatch, outcomes)
    _write_suite(cli_workspace["suites_dir"], len(outcomes))
    config_path = _write_config(
        cli_workspace["root"],
        cli_workspace["suites_dir"],
        cli_workspace["output_dir"],
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(config_path), "run"])

    if all(outcomes):
        assert result.exit_code == 0, (
            f"expected exit 0 (all pass), got {result.exit_code}\n"
            f"stdout: {result.stdout}\n"
        )
    else:
        assert result.exit_code != 0, (
            f"expected non-zero exit, got {result.exit_code}\n"
            f"stdout: {result.stdout}\n"
        )
