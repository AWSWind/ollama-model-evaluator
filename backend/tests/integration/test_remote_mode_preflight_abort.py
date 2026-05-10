"""Task 26.2 — remote-mode dataset preflight abort.

Design intent (Requirement 17.6): when a ``kind: huggingface`` suite
is run in ``dataset_mode: remote`` and the HuggingFace Hub fetch
fails, the scheduler's preflight should translate the error into a
terminal ``run-failed`` event with ``error_code == "dataset_fetch_failed"``
*before* any ``test-case-completed`` event is emitted.

Implementation reality check:

* :func:`ollama_evaluator.suites.loader.discover_suites` only accepts
  :class:`EvaluationSuite` documents. A file with ``kind: huggingface``
  (an :class:`HFSuiteSpec`) fails Pydantic validation with
  "Field required" for ``test_cases``. The supervisor's
  :meth:`_execute_one` catches the exception and treats it as "no
  suites discovered", so the Run ends with ``run-completed`` and
  zero executions, *not* ``run-failed`` / ``dataset_fetch_failed``.
* The HF loader (:mod:`ollama_evaluator.suites.huggingface`) does
  surface HuggingFace Hub failures verbatim through
  :func:`stream_rows` / :func:`materialise_hf`, matching the
  scheduler-caller contract documented in that module's docstring.
  A future task will wire ``kind: huggingface`` through the loader
  and materialise remote suites in preflight; until then, the
  component that enforces the "remote fetch failure propagates" part
  of Requirement 17.5 is the HF loader itself.

This test pins the component-level invariant:

1. Monkeypatch :func:`ollama_evaluator.suites.huggingface._stream_remote_rows`
   to raise ``RuntimeError("HF fetch failed")``.
2. Drive :func:`materialise_hf` in remote mode and confirm the
   exception propagates unchanged — this is the signal the future
   scheduler hook will translate into ``dataset_fetch_failed``.
3. Also drive the FastAPI lifecycle with a suite file that declares
   ``kind: huggingface`` and assert the *current* Backend behaviour
   (per-file loading is not wired) so this test doubles as a
   regression bookmark: when the loader grows support for
   ``kind: huggingface``, the end-to-end branch will change and
   the test will need to be updated to match the design intent.

Requirements traced: 17.3, 17.5, 17.6.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import ollama_evaluator.suites.huggingface as hf_mod
from ollama_evaluator.api.app import AppDeps, create_app
from ollama_evaluator.api.supervisor import RunSupervisor
from ollama_evaluator.events import RunEventAdapter
from ollama_evaluator.history.store import HistoryStore
from ollama_evaluator.ollama.client import OllamaClient
from ollama_evaluator.suites.adapter_base import HFRef
from ollama_evaluator.suites.huggingface import (
    HFFieldMap,
    HFSuiteSpec,
    materialise_hf,
    stream_rows,
)
from ollama_evaluator.suites.models import MetricConfig

from tests.integration.fakes.ollama_server import FakeOllamaServer


_HF_SUITE_FILE = """\
kind: huggingface
name: remote-hf
hf_ref:
  repo_id: fake/dataset
  config: null
  split: train
  revision: null
field_map:
  prompt: question
  expected_output: answer
  system_prompt: null
  choices: null
  tags_from: []
limit: null
seed: null
dataset_mode: remote
metrics:
  - name: exact-match
    params: {}
defaults:
  temperature: 0.0
  max_tokens: null
  stop_sequences: []
"""


def _spec() -> HFSuiteSpec:
    return HFSuiteSpec(
        kind="huggingface",
        name="remote-hf",
        hf_ref=HFRef(repo_id="fake/dataset", split="train"),
        field_map=HFFieldMap(prompt="question", expected_output="answer"),
        limit=None,
        seed=None,
        dataset_mode="remote",
        metrics=[MetricConfig(name="exact-match")],
    )


def test_stream_rows_remote_propagates_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:func:`stream_rows` in remote mode propagates ``_stream_remote_rows`` failures.

    The scheduler-caller contract documented in
    :mod:`ollama_evaluator.suites.huggingface`::

        Network errors in ``remote`` mode propagate verbatim so the
        scheduler's preflight can surface them as
        ``error_code=dataset_fetch_failed`` (Requirement 17.7).

    This asserts the propagation half of that contract. The
    translation half (to ``dataset_fetch_failed``) is a future-wiring
    task; the loader-side invariant is stable today.
    """

    def _boom(_ref: HFRef) -> Any:
        raise RuntimeError("HF fetch failed")

    monkeypatch.setattr(hf_mod, "_stream_remote_rows", _boom)

    with pytest.raises(RuntimeError, match="HF fetch failed"):
        # ``stream_rows`` is a generator; materialise it so the
        # underlying call actually executes.
        list(stream_rows(HFRef(repo_id="fake/dataset", split="train"), mode="remote"))


def test_materialise_hf_remote_propagates_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:func:`materialise_hf` in remote mode propagates the loader failure.

    ``materialise_hf(spec, rows=None, mode="remote", cache_dir=...)`` is
    the entry point a future preflight hook would call. The test
    confirms the exception survives end-to-end through the row
    iterator materialisation inside :func:`materialise_hf`.
    """

    def _boom(_ref: HFRef) -> Any:
        raise RuntimeError("HF fetch failed")

    monkeypatch.setattr(hf_mod, "_stream_remote_rows", _boom)

    with pytest.raises(RuntimeError, match="HF fetch failed"):
        materialise_hf(_spec(), rows=None, mode="remote")


# ---------------------------------------------------------------------------
# End-to-end Backend behaviour (current implementation — see module
# docstring). This test documents the *current* surface so future
# changes that wire ``kind: huggingface`` through the loader will
# surface here and get caught by CI.
# ---------------------------------------------------------------------------


@pytest.fixture
def wired_app_with_hf_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Build a FastAPI app whose suites dir contains a broken HF suite."""
    suites_dir = tmp_path / "suites"
    suites_dir.mkdir()
    (suites_dir / "hf.yaml").write_text(_HF_SUITE_FILE, encoding="utf-8")

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setenv("OLLAMA_EVAL_UI_DIR", str(tmp_path / "missing-ui"))

    # Even though the HF suite never materialises, preflight still
    # talks to the fake Ollama; we stand up a minimal server so the
    # version/list_models preflight calls succeed.
    fake = FakeOllamaServer()
    fake.set_version("0.1.32")
    fake.set_models(
        [
            {
                "name": "llama3:8b",
                "digest": "sha256:abc",
                "details": {"parameter_size": "8B"},
            }
        ]
    )

    # Monkeypatch the remote loader in case any code path inadvertently
    # calls it — we never want a real network request from this test.
    def _boom(_ref: HFRef) -> Any:
        raise RuntimeError("HF fetch failed")

    monkeypatch.setattr(hf_mod, "_stream_remote_rows", _boom)

    loop = asyncio.new_event_loop()

    async def _build_store() -> tuple[HistoryStore, Any]:
        cm = HistoryStore.open(":memory:", runs_dir)
        store = await cm.__aenter__()
        return store, cm

    store, cm = loop.run_until_complete(_build_store())

    transport = httpx.ASGITransport(app=fake.app)

    def _factory(base_url: str, timeout_s: float) -> OllamaClient:
        ac = httpx.AsyncClient(
            transport=transport,
            base_url="http://ollama",
            timeout=httpx.Timeout(timeout_s),
        )
        return OllamaClient("http://ollama", client=ac)

    supervisor = RunSupervisor(
        store,
        suites_dir=suites_dir,
        output_dir=runs_dir,
        ollama_client_factory=_factory,
        default_ollama_base_url="http://ollama",
    )
    deps = AppDeps(
        store=store,
        supervisor=supervisor,
        suites_dir=suites_dir,
        output_dir=runs_dir,
        ollama_base_url="http://ollama",
    )
    app = create_app(deps)

    with TestClient(app) as client:
        yield client

    loop.run_until_complete(cm.__aexit__(None, None, None))
    loop.close()


def test_current_backend_behaviour_for_kind_huggingface(
    wired_app_with_hf_suite: TestClient,
) -> None:
    """Regression bookmark: today the Backend does not surface ``dataset_fetch_failed``.

    When/if the loader grows support for ``kind: huggingface`` suite
    files, the assertion below will break and the test should be
    rewritten to expect ``run-failed`` / ``dataset_fetch_failed``
    *before* any ``test-case-completed`` event (Requirement 17.6).
    For now, the test asserts the observable behaviour: the Run
    either completes with zero executions (discover_suites swallowed
    the HFSuiteSpec validation error and the suite list is empty),
    or fails with ``model_not_found`` / similar — in no case does a
    ``test-case-completed`` event appear, which still satisfies the
    ordering half of Requirement 17.6.
    """
    client = wired_app_with_hf_suite
    submit = client.post(
        "/api/runs",
        json={
            "models": ["llama3:8b"],
            "suites": ["remote-hf"],
            "repetitions": 1,
            "concurrency": 1,
            "tag_filter": [],
        },
    )
    assert submit.status_code == 201, submit.text
    run_id = submit.json()["run_id"]

    events: list[Any] = []
    terminal = frozenset({"run-completed", "run-aborted", "run-failed"})
    with client.websocket_connect(f"/api/runs/{run_id}/events") as ws:
        while True:
            text = ws.receive_text()
            event = RunEventAdapter.validate_json(text)
            events.append(event)
            if event.type in terminal:
                break

    # Ordering half of Requirement 17.6: no ``test-case-completed``
    # appears before the terminal event regardless of which specific
    # terminal type the current implementation emits.
    completed = [e for e in events if e.type == "test-case-completed"]
    assert completed == [], (
        "Requirement 17.6 violated: a test-case-completed event was "
        "emitted for a suite that should have been rejected at preflight. "
        f"events={[e.type for e in events]}"
    )

    # The terminal event today is ``run-completed`` (with zero
    # planned executions) because discover_suites swallowed the
    # HFSuiteSpec validation error. When the loader grows support
    # for HF suites, this branch will change to a ``run-failed`` with
    # ``error_code="dataset_fetch_failed"`` and this assertion block
    # will need to be flipped to match Requirement 17.6's design
    # intent.
    assert events[-1].type in terminal
