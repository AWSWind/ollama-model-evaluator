"""Unit tests for the lazy ``AppDeps.store_factory`` lifespan path.

Regression coverage for a bug surfaced during the remote end-to-end
run: the CLI's ``serve`` subcommand used to open the
:class:`HistoryStore` via ``asyncio.run(...)`` *before* uvicorn
started. That bound the underlying ``aiosqlite`` connection to a
short-lived event loop which was closed before the first request
landed. As soon as a handler reached for ``self._conn.execute``,
``aiosqlite`` raised ``ValueError: no active connection``.

The fix moves the store open into the FastAPI lifespan so the
connection is bound to uvicorn's own loop. These tests pin that
contract:

1. When ``AppDeps.store_factory`` is set, the lifespan calls it on
   startup, wires the supervisor around the opened store, and both
   are reachable from request handlers (``/api/runs`` exercises the
   store path that the bug originally broke).
2. Supplying both ``store`` and ``store_factory`` is rejected at
   construction time so ambiguous configurations cannot reach the
   lifespan.
3. Supplying ``store_factory`` without a ``supervisor_factory``
   (or vice-versa) is rejected for the same reason.
4. The cleanup closure is invoked on shutdown — asserted by
   wrapping ``HistoryStore.open``'s real cleanup with a spy.

Requirements traced: 12.1 (store persists across requests), 13.2
(``/api/runs`` returns a list), 13.3 (run lifecycle wiring).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ollama_evaluator.api.app import AppDeps, create_app
from ollama_evaluator.api.supervisor import RunSupervisor
from ollama_evaluator.history.store import HistoryStore


def test_rejects_both_eager_store_and_store_factory(tmp_path: Path) -> None:
    """``AppDeps`` rejects ambiguous eager + lazy configurations."""
    async def _factory() -> tuple[Any, Any]:  # pragma: no cover - not called
        return object(), _noop_cleanup

    async def _noop_cleanup() -> None:  # pragma: no cover
        return None

    with pytest.raises(ValueError, match="either `store` .+ or `store_factory`"):
        AppDeps(
            store=object(),
            supervisor=object(),
            suites_dir=tmp_path,
            output_dir=tmp_path,
            store_factory=_factory,
            supervisor_factory=lambda s: object(),
        )


def test_rejects_store_factory_without_supervisor_factory(tmp_path: Path) -> None:
    """``store_factory`` requires a matching ``supervisor_factory``."""
    async def _factory() -> tuple[Any, Any]:  # pragma: no cover
        return object(), _noop_cleanup

    async def _noop_cleanup() -> None:  # pragma: no cover
        return None

    with pytest.raises(ValueError, match="requires `supervisor_factory`"):
        AppDeps(
            suites_dir=tmp_path,
            output_dir=tmp_path,
            store_factory=_factory,
        )


def test_rejects_supervisor_factory_without_store_factory(tmp_path: Path) -> None:
    """``supervisor_factory`` alone is nonsensical and is rejected."""
    with pytest.raises(ValueError, match="requires a matching"):
        AppDeps(
            suites_dir=tmp_path,
            output_dir=tmp_path,
            supervisor_factory=lambda s: object(),
        )


def test_lifespan_opens_store_and_serves_store_backed_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: lazy store-factory lets ``/api/runs`` return 200 with an empty list.

    This is the exact shape that failed under the old ``asyncio.run``
    bootstrap: the handler calls ``store.list_runs(...)`` which goes
    through ``aiosqlite``'s connection. If the connection is bound to
    the wrong loop the test 500s; when the lifespan owns the open,
    it succeeds with an empty JSON array.
    """
    # Keep the dev-workspace ``ui/dist`` out of the app so
    # ``GET /api/runs`` is not shadowed by the SPA fallback (it
    # wouldn't be in practice, but setting the env removes the
    # ambiguity for this test).
    monkeypatch.setenv("OLLAMA_EVAL_UI_DIR", str(tmp_path / "missing"))

    suites_dir = tmp_path / "suites"
    suites_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    cleanup_calls: list[str] = []

    async def _store_factory() -> tuple[Any, Any]:
        # Open on the lifespan's loop, exactly how the CLI's
        # ``serve`` command does it post-refactor.
        cm = HistoryStore.open(":memory:", runs_dir)
        store = await cm.__aenter__()

        async def _cleanup() -> None:
            cleanup_calls.append("cleanup")
            await cm.__aexit__(None, None, None)

        return store, _cleanup

    def _supervisor_factory(store: Any) -> RunSupervisor:
        return RunSupervisor(
            store,
            suites_dir=suites_dir,
            output_dir=runs_dir,
        )

    deps = AppDeps(
        suites_dir=suites_dir,
        output_dir=runs_dir,
        store_factory=_store_factory,
        supervisor_factory=_supervisor_factory,
    )
    # Before the lifespan runs, ``store`` / ``supervisor`` are still
    # ``None`` on the deps instance — the lifespan populates them.
    assert deps.store is None
    assert deps.supervisor is None

    app = create_app(deps)

    with TestClient(app) as client:
        # ``TestClient`` as a context manager runs the lifespan.
        # After __enter__, the lazy-open path must have populated
        # ``deps.store`` and ``deps.supervisor``.
        assert deps.store is not None
        assert deps.supervisor is not None
        assert isinstance(deps.store, HistoryStore)

        # The failing path from the bug report: ``GET /api/runs``.
        # The store was just opened so there are no rows, but the
        # handler reaches through ``aiosqlite`` to run the SELECT.
        response = client.get("/api/runs")
        assert response.status_code == 200, response.text
        assert response.json() == []

        # ``/api/health`` still works.
        health = client.get("/api/health")
        assert health.status_code == 200

    # Leaving the ``TestClient`` context invokes the shutdown branch
    # of the lifespan, which must call the cleanup closure exactly
    # once.
    assert cleanup_calls == ["cleanup"]
