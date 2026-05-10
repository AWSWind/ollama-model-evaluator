"""Task 26.6 — CLI ``run`` exit codes.

Requirements 10.2, 10.3, 1.3: the ``ollama-evaluator run`` subcommand
returns well-defined exit codes so shell scripts and CI pipelines can
branch on outcome:

* ``0`` — Run completed and every :class:`TestCaseResult.status`
  equals ``"pass"``.
* ``1`` — Run completed but at least one test case did not pass, or
  a catch-all error fired during ``run``.
* ``2`` — Preflight error (``ollama_unreachable``,
  ``model_not_found``, ``dataset_fetch_failed``, ``field_map_invalid``).

Per the task notes, wiring a full FakeOllamaServer end-to-end through
Typer's :class:`CliRunner` is possible but heavy — the CLI builds its
own :class:`OllamaClient` inside :func:`_execute_run`, and swapping in
an ASGI-transport-backed one from outside means monkeypatching the
module-level symbol. The cleanest approach is to drive
:func:`_execute_run` directly with a monkeypatched
``ollama_evaluator.cli.OllamaClient`` (so whatever we return becomes
the scheduler's client) and assert the function's return value — which
is the exact value the ``run_cmd`` wrapper passes to
:class:`typer.Exit` (Property 20).

The task description permits this narrowing:

    Approach: since wiring a live FakeOllamaServer through the CLI
    is complicated, consider instead monkeypatching the scheduler
    path or using the supervisor directly. Use whatever is cleanest
    for the current code; if fully end-to-end is hard, test
    ``_execute_run(config)`` directly and assert its return value
    (the function returns the exit code).

Three scenarios are exercised:

1. All test cases pass → exit code 0.
2. At least one test case fails → non-zero exit code (``1``).
3. Ollama unreachable (``httpx.ConnectError`` on ``version``) →
   exit code 2.

Requirements traced: 10.2, 10.3, 1.3.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

import ollama_evaluator.cli as cli_mod
from ollama_evaluator.config import ConfigFile, RunConfig
from ollama_evaluator.ollama.types import (
    GenerateChunk,
    GenerateOptions,
    OllamaModelInfo,
    PullProgress,
)
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    MetricConfig,
    TestCase,
)
from ollama_evaluator.suites.writer import dump_suite


# ---------------------------------------------------------------------------
# In-memory OllamaClient substitute
# ---------------------------------------------------------------------------


class _ScriptedOllamaClient:
    """Fake :class:`OllamaClient` with scripted outcomes.

    The fake implements the four methods the scheduler touches
    (``version``, ``list_models``, ``generate``, ``aclose``). Behaviour
    is configured via three constructor kwargs:

    * ``response`` — the response string the ``generate`` stream
      emits. Drives whether the scheduler's ``exact-match`` metric
      marks the execution ``pass`` or ``fail``.
    * ``version_raises`` — optional exception raised from ``version``.
      Used to simulate an unreachable Ollama server (preflight fails
      with ``ollama_unreachable``).
    * ``models`` — the list of :class:`OllamaModelInfo` returned by
      ``list_models``. Defaults to a single ``llama3:8b`` entry so
      preflight's ``model_not_found`` check passes.
    """

    def __init__(
        self,
        *,
        response: str = "4",
        version_raises: BaseException | None = None,
        models: list[OllamaModelInfo] | None = None,
    ) -> None:
        self._response = response
        self._version_raises = version_raises
        self._models = models if models is not None else [
            OllamaModelInfo(
                name="llama3:8b",
                digest="sha256:abc",
                parameter_size="8B",
            )
        ]

    async def version(self) -> str:
        if self._version_raises is not None:
            raise self._version_raises
        return "0.1.32"

    async def list_models(self) -> list[OllamaModelInfo]:
        return list(self._models)

    async def aclose(self) -> None:
        return None

    async def pull_model(self, name: str) -> AsyncIterator[PullProgress]:
        yield PullProgress(status="success")

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        # A partial chunk carrying the scripted response followed by
        # a done chunk with populated timings so the performance
        # metrics on the TestCaseResult are non-null.
        yield GenerateChunk(
            model=model,
            created_at=datetime.now(tz=timezone.utc),
            response=self._response,
            done=False,
        )
        yield GenerateChunk(
            model=model,
            created_at=datetime.now(tz=timezone.utc),
            response="",
            done=True,
            total_duration=100_000_000,
            load_duration=0,
            prompt_eval_count=3,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=100_000_000,
        )


def _make_client_factory(**kwargs: Any):
    """Return a callable matching ``OllamaClient(base_url=..., timeout_s=...)``."""

    def factory(*_args: Any, **_kwargs: Any) -> _ScriptedOllamaClient:
        # The CLI calls ``OllamaClient(base_url=config.ollama_base_url,
        # timeout_s=config.run.ollama_timeout_s)`` positionally-by-
        # keyword; we accept arbitrary args and return our fake.
        return _ScriptedOllamaClient(**kwargs)

    return factory


def _write_suite(suites_dir: Path, *, expected_output: str = "4") -> None:
    """Drop a one-test-case suite with ``exact-match`` metric on disk."""
    suites_dir.mkdir(parents=True, exist_ok=True)
    suite = EvaluationSuite(
        name="tiny",
        test_cases=[
            TestCase(
                id="tc1",
                prompt="What is 2+2?",
                expected_output=expected_output,
                metrics=[MetricConfig(name="exact-match")],
            )
        ],
    )
    (suites_dir / "tiny.yaml").write_text(
        dump_suite(suite, "yaml"), encoding="utf-8"
    )


def _make_config(tmp_path: Path) -> ConfigFile:
    """Build a :class:`ConfigFile` matching the suite written on disk."""
    return ConfigFile(
        ollama_base_url="http://ollama",
        suites_dir=tmp_path / "suites",
        output_dir=tmp_path / "runs",
        run=RunConfig(
            models=["llama3:8b"],
            suites=["tiny"],
            repetitions=1,
            concurrency=1,
            tag_filter=[],
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 1 — all test cases pass → exit 0
# ---------------------------------------------------------------------------


async def test_exit_code_0_when_all_test_cases_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 10.3: exit 0 iff every execution's ``status == 'pass'``."""
    _write_suite(tmp_path / "suites", expected_output="4")
    monkeypatch.setattr(
        cli_mod,
        "OllamaClient",
        _make_client_factory(response="4"),
    )

    exit_code = await cli_mod._execute_run(_make_config(tmp_path))
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Scenario 2 — at least one test case fails → non-zero exit
# ---------------------------------------------------------------------------


async def test_exit_code_nonzero_when_a_test_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 10.3: a failed metric pushes the run to exit code 1."""
    # Expected "4" but model replies "nope" — ``exact-match`` fails.
    _write_suite(tmp_path / "suites", expected_output="4")
    monkeypatch.setattr(
        cli_mod,
        "OllamaClient",
        _make_client_factory(response="nope"),
    )

    exit_code = await cli_mod._execute_run(_make_config(tmp_path))
    assert exit_code != 0
    # Requirement 10.3 reserves exit 2 for preflight errors; a metric
    # failure must map to a non-preflight non-zero value (the CLI's
    # current implementation returns 1).
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Scenario 3 — Ollama unreachable → exit 2
# ---------------------------------------------------------------------------


async def test_exit_code_2_when_ollama_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 1.3 + 10.3: ``ollama_unreachable`` maps to exit code 2.

    The scripted client raises :class:`httpx.ConnectError` from
    ``version``. The scheduler's preflight catches it, emits
    ``run-failed`` with ``error_code=ollama_unreachable``, and the
    CLI's exit-code resolver (in :func:`_execute_run`) maps that
    preflight code to ``2`` via the ``_PREFLIGHT_ERROR_CODES``
    frozenset.
    """
    _write_suite(tmp_path / "suites", expected_output="4")

    def _connect_error_factory(*_args: Any, **_kwargs: Any) -> _ScriptedOllamaClient:
        return _ScriptedOllamaClient(
            version_raises=httpx.ConnectError("connection refused"),
        )

    monkeypatch.setattr(cli_mod, "OllamaClient", _connect_error_factory)

    exit_code = await cli_mod._execute_run(_make_config(tmp_path))
    assert exit_code == 2
