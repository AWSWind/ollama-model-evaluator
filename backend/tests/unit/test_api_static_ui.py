"""Unit tests for the UI static-mount wiring in :mod:`api.app`.

Requirements traced:

* 10.7 — ``serve`` subcommand mounts the UI bundle.
* 15.1 — UI is delivered as a web application that runs in the browser.

The tests exercise two branches of :func:`create_app` in isolation:

1. With ``OLLAMA_EVAL_UI_DIR`` pointed at a synthetic ``tmp_path/dist``
   containing an ``index.html``, ``GET /`` returns that HTML so the UI
   is reachable alongside the API.
2. With no UI bundle discoverable and no env override, the app still
   boots and ``GET /`` returns 404 (FastAPI's default for an unknown
   route) so the existing ``pytest`` suite that never builds the UI
   continues to pass.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from ollama_evaluator.api.app import AppDeps, create_app
from ollama_evaluator.api.supervisor import RunSupervisor
from ollama_evaluator.history.store import HistoryStore


@pytest.fixture
def deps_factory(tmp_path: Path):
    """Factory producing an :class:`AppDeps` backed by an in-memory store.

    Returned as a factory rather than a pre-built ``deps`` so each test
    can decorate the surrounding environment (e.g. set / unset
    ``OLLAMA_EVAL_UI_DIR``) before the app is instantiated.
    """
    loop = asyncio.new_event_loop()

    async def _setup() -> tuple[HistoryStore, object]:
        cm = HistoryStore.open(":memory:", tmp_path / "runs")
        store = await cm.__aenter__()
        return store, cm

    store, cm = loop.run_until_complete(_setup())
    (tmp_path / "suites").mkdir(parents=True, exist_ok=True)
    sup = RunSupervisor(
        store,
        suites_dir=tmp_path / "suites",
        output_dir=tmp_path / "runs",
    )

    def build() -> AppDeps:
        return AppDeps(
            store=store,
            supervisor=sup,
            suites_dir=tmp_path / "suites",
            output_dir=tmp_path / "runs",
        )

    yield build

    loop.run_until_complete(cm.__aexit__(None, None, None))  # type: ignore[attr-defined]
    loop.close()


def test_serves_ui_index_when_dist_env_points_at_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deps_factory,
) -> None:
    """With ``OLLAMA_EVAL_UI_DIR`` set and valid, ``GET /`` returns HTML."""
    dist = tmp_path / "dist"
    dist.mkdir()
    html = "<!doctype html><html><body>Ollama UI</body></html>"
    (dist / "index.html").write_text(html, encoding="utf-8")

    monkeypatch.setenv("OLLAMA_EVAL_UI_DIR", str(dist))

    app = create_app(deps_factory())
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        # Content-type should be HTML; body includes the marker.
        assert "text/html" in response.headers.get("content-type", "")
        assert "Ollama UI" in response.text

        # Deep-link fallback: any non-API path returns the same index
        # so the React Router tree can claim the route.
        spa = client.get("/runs/some-id")
        assert spa.status_code == 200
        assert "Ollama UI" in spa.text

        # API routes still behave normally (not swallowed by the
        # fallback). ``/api/health`` returns its JSON body.
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.headers["content-type"].startswith("application/json")


def test_no_ui_dist_still_boots_and_root_is_404(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deps_factory,
) -> None:
    """Without a UI bundle, the app boots and ``GET /`` returns 404."""
    # Ensure no env override leaks from outer scope.
    monkeypatch.delenv("OLLAMA_EVAL_UI_DIR", raising=False)
    # Set an explicit override to a non-existent directory to guarantee
    # the "no bundle" branch regardless of any sibling ``ui/dist`` that
    # might exist in the developer workspace.
    monkeypatch.setenv("OLLAMA_EVAL_UI_DIR", str(tmp_path / "missing"))

    app = create_app(deps_factory())
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 404
        # API continues to work.
        health = client.get("/api/health")
        assert health.status_code == 200
