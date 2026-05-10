"""FastAPI application factory for the Backend HTTP API.

Callers build an app with :func:`create_app` passing an :class:`AppDeps`
bag of shared dependencies (store, Ollama client factory, suites dir,
output dir, supervisor). The factory wires the REST router
(:mod:`api.rest`), the WebSocket endpoint (:mod:`api.events_ws`), the
global validation-error exception handler (Requirement 13.6), the
startup/shutdown hooks for the :class:`RunSupervisor`, and — when a
built UI bundle is present — a static file mount so the Backend can
serve the UI alongside the REST API (Requirements 10.7, 15.1).

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§REST API, §WebSocket event stream.

Store lifecycle
---------------
``AppDeps`` carries two mutually-exclusive ways to plug a
:class:`HistoryStore` into the app:

* **Eager** — set ``AppDeps.store`` to an already-open store. Used
  exclusively by tests and in-process embeddings where the caller
  owns the connection. The lifespan does not open or close it.

* **Lazy** — set ``AppDeps.store_factory`` to an async callable
  returning an ``(store, cleanup)`` tuple. The lifespan opens the
  store on *its own* event loop (the one uvicorn serves on), wires
  the supervisor around it, and calls ``cleanup`` at shutdown. This
  is the path the CLI ``serve`` subcommand uses so that ``aiosqlite``
  never binds to a short-lived loop and then tries to execute on the
  uvicorn loop (Issue: "ValueError: no active connection" on the
  first store query after startup).

The two paths are mutually exclusive — ``create_app`` accepts either
but not both. Supplying both raises at construction time because the
lifespan has no meaningful way to pick between an already-open store
and a factory that would produce a second one.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncContextManager, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ..ollama.client import OllamaClient
from .errors import ErrorCode, ErrorEnvelope
from .rest import build_router

log = logging.getLogger(__name__)


StoreCleanup = Callable[[], Awaitable[None]]
"""Async no-arg callable that releases resources acquired by a store factory."""

StoreFactory = Callable[[], Awaitable[tuple[Any, StoreCleanup]]]
"""Builds a fresh :class:`HistoryStore` on the caller's event loop.

Returns ``(store, cleanup)``: ``store`` is the open
:class:`HistoryStore` instance, ``cleanup`` is an async callable the
lifespan invokes at shutdown to release the underlying connection.
"""

SupervisorFactory = Callable[[Any], Any]
"""Builds a supervisor around the store the lifespan just opened.

Receives the opened :class:`HistoryStore` and returns the matching
:class:`RunSupervisor`. Used in the lazy-open path so the supervisor
can keep a reference to the long-lived store.
"""


@dataclass
class AppDeps:
    """Container for every dependency the FastAPI app needs at request time.

    A plain dataclass rather than a Pydantic model so callers can pass
    complex objects (:class:`HistoryStore` instances, supervisor
    references, client factories) without Pydantic trying to validate
    them. The app stashes this on ``app.state.deps`` and request
    handlers fetch it via :func:`api.rest._req_state`.

    Two wiring patterns are supported (see module docstring for the
    full rationale):

    * **Eager** (tests, in-process embeddings). Set ``store`` and
      ``supervisor`` to already-initialised instances. The lifespan
      does not open or close the store; it only calls
      ``supervisor.start``/``supervisor.stop`` if the supervisor
      exposes those methods.

    * **Lazy** (``cli serve``). Leave ``store`` / ``supervisor`` as
      ``None`` and provide ``store_factory`` + ``supervisor_factory``
      instead. The lifespan calls ``store_factory()`` on uvicorn's
      own event loop, hands the opened store to ``supervisor_factory``
      to build a supervisor, starts that supervisor, and closes
      everything on shutdown. This path keeps the ``aiosqlite``
      connection bound to the serving loop.
    """

    store: Any = None
    supervisor: Any = None
    suites_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("./runs"))
    ollama_client_factory: Callable[[str, float], Any] = lambda url, t: OllamaClient(  # noqa: E731
        base_url=url, timeout_s=t
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_s: float = 120.0

    # Lazy-open hooks. When both are provided, ``store`` and
    # ``supervisor`` must be ``None`` — the lifespan takes ownership.
    store_factory: StoreFactory | None = None
    supervisor_factory: SupervisorFactory | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous eager+lazy configurations at construction time."""
        eager_store = self.store is not None
        lazy_store = self.store_factory is not None
        if eager_store and lazy_store:
            raise ValueError(
                "AppDeps: pass either `store` (eager) or "
                "`store_factory` (lazy), not both"
            )
        if lazy_store and self.supervisor_factory is None:
            raise ValueError(
                "AppDeps: `store_factory` requires `supervisor_factory` "
                "so the lifespan can build the supervisor around the "
                "opened store"
            )
        if (
            lazy_store is False
            and self.supervisor_factory is not None
        ):
            raise ValueError(
                "AppDeps: `supervisor_factory` requires a matching "
                "`store_factory`"
            )


def _dotted_loc(loc: tuple[Any, ...]) -> str | None:
    """Return the dotted-path representation of a Pydantic ``loc`` tuple.

    Pydantic validation errors locate the offending field with a
    ``loc`` tuple like ``("body", "models", 0)``. For the response's
    ``field`` we strip the leading ``"body"`` / ``"query"`` /
    ``"path"`` marker FastAPI adds (Requirement 13.6 wants the field
    path within the request body) and join the rest with ``.``.
    """
    if not loc:
        return None
    parts = list(loc)
    if parts and parts[0] in ("body", "query", "path", "header", "cookie"):
        parts = parts[1:]
    if not parts:
        return None
    return ".".join(str(p) for p in parts)


async def _validation_handler(
    request: Request, exc: RequestValidationError | ValidationError
) -> JSONResponse:
    """Global handler for Pydantic / FastAPI validation errors.

    Returns ``400 {error_code: "validation_failed", field, message}``
    with ``field`` set to the dotted path of the first error in
    document order (Requirement 13.6, Property 30).
    """
    del request  # unused
    errors = exc.errors() if hasattr(exc, "errors") else []
    if errors:
        first = errors[0]
        field = _dotted_loc(tuple(first.get("loc") or ()))
        message = str(first.get("msg") or "Validation failed")
    else:  # pragma: no cover - Pydantic always produces at least one error.
        field = None
        message = "Validation failed"
    envelope = ErrorEnvelope(
        error_code=ErrorCode.validation_failed,
        message=message,
        field=field,
    )
    return JSONResponse(status_code=400, content=envelope.model_dump(mode="json"))


async def _http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Unwrap :class:`HTTPException` whose ``detail`` is an envelope dict.

    :func:`ollama_evaluator.api.errors.http_error` packs an
    :class:`ErrorEnvelope` into ``detail`` so the envelope matches the
    wire contract (Requirement 13.5). FastAPI's default handler would
    otherwise wrap the dict inside ``{"detail": {...}}``; this handler
    surfaces the envelope at the top level.
    """
    del request  # unused
    detail = exc.detail
    if isinstance(detail, dict) and "error_code" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    # Fall back to the default shape for exceptions raised by callers
    # that did not use :func:`http_error`.
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


def _resolve_ui_dir() -> Path | None:
    """Resolve the directory containing the built UI, or return ``None``.

    Resolution order:

    1. ``OLLAMA_EVAL_UI_DIR`` environment variable (absolute or
       relative). Wins so tests and deployments can point at a
       synthetic or packaged bundle.
    2. Walk up from this module file looking for ``<ancestor>/ui/dist``.
       This works in the developer workspace where the Backend and UI
       sit side-by-side in the repo root.

    Returns ``None`` when neither source yields an existing directory
    containing ``index.html``; the caller skips the static mount in that
    case so the API still boots without a UI build (Requirement 15.1).
    """
    env = os.environ.get("OLLAMA_EVAL_UI_DIR")
    if env:
        candidate = Path(env).resolve()
        if (candidate / "index.html").is_file():
            return candidate
        return None

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "ui" / "dist"
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _mount_ui(app: FastAPI, ui_dir: Path) -> None:
    """Serve ``ui_dir`` at ``/`` with SPA deep-link fallback.

    Static files are mounted at ``/assets`` (Vite's default output
    subdirectory) so ``/api`` routes take precedence. A catch-all route
    maps every other non-API, non-docs path to ``index.html`` so
    client-side router routes like ``/runs/abc``, ``/history``, and
    ``/compare`` deep-link correctly. Paths under ``/api``,
    ``/openapi.json``, and ``/docs`` are left to FastAPI.
    """
    index_path = ui_dir / "index.html"

    # Serve the build's ``assets/`` subtree verbatim (JS/CSS bundles).
    assets_dir = ui_dir / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="ui-assets",
        )

    # Expose individual top-level files (favicon.ico, vite.svg, ...)
    # that are not covered by the ``/assets`` mount. ``StaticFiles``
    # cannot be mounted at ``/`` because that would shadow every API
    # route; instead we register a best-effort catch-all below.

    @app.get("/", include_in_schema=False)
    async def _serve_index() -> FileResponse:
        return FileResponse(str(index_path))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _ui_fallback(full_path: str) -> FileResponse:
        # Reserve API, docs, and OpenAPI namespaces for FastAPI.
        if (
            full_path.startswith("api/")
            or full_path == "openapi.json"
            or full_path.startswith("docs")
            or full_path.startswith("redoc")
        ):
            raise HTTPException(status_code=404, detail="Not Found")
        # If the requested path exists inside the UI bundle, serve it
        # verbatim (e.g. ``/vite.svg``). Otherwise fall back to
        # ``index.html`` so React Router can claim the route.
        candidate = ui_dir / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(index_path))


def create_app(deps: AppDeps) -> FastAPI:
    """Build and return a wired-up :class:`FastAPI` instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        """Manage the store + supervisor for the lifetime of the app.

        Two code paths:

        * Eager (``deps.store`` already set): just start/stop the
          supervisor. The store is owned by the caller.
        * Lazy (``deps.store_factory`` set): open the store on
          *this* event loop (uvicorn's), construct the supervisor
          against it, start the supervisor, and clean both up at
          shutdown. This keeps the ``aiosqlite`` connection bound to
          the loop that actually services requests.
        """
        cleanup: StoreCleanup | None = None

        if deps.store is None and deps.store_factory is not None:
            # Lazy path. Both `store` and `supervisor` attributes on
            # ``deps`` are mutated here so request handlers fetching
            # ``request.app.state.deps`` see the opened instances.
            factory = deps.store_factory
            store, cleanup = await factory()
            deps.store = store
            assert deps.supervisor_factory is not None
            deps.supervisor = deps.supervisor_factory(store)

        supervisor = deps.supervisor
        if supervisor is not None and hasattr(supervisor, "start"):
            await supervisor.start()
        try:
            yield
        finally:
            if supervisor is not None and hasattr(supervisor, "stop"):
                try:
                    await supervisor.stop()
                except Exception:  # noqa: BLE001 — teardown must not raise
                    log.exception("supervisor.stop() raised during shutdown")
            if cleanup is not None:
                try:
                    await cleanup()
                except Exception:  # noqa: BLE001 — teardown must not raise
                    log.exception("store cleanup raised during shutdown")

    app = FastAPI(
        title="Ollama Model Evaluator",
        version="0.1.0",
        description="Local evaluation harness for Ollama-hosted LLMs.",
        lifespan=lifespan,
    )
    app.state.deps = deps

    # Global validation-error handlers (Requirement 13.6).
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(ValidationError, _validation_handler)
    # Unwrap HTTPException.detail when it is already an envelope dict.
    app.add_exception_handler(HTTPException, _http_exception_handler)

    app.include_router(build_router())

    # WebSocket endpoint is registered via a helper so the rest.py
    # router stays focused on REST and the events endpoint can share
    # the same ``AppDeps`` lookup.
    from .events_ws import register_events_ws

    register_events_ws(app)

    # Optional UI bundle. The mount is a no-op when ``ui/dist`` does
    # not exist so ``pytest`` can exercise the REST surface without
    # first building the frontend (Requirement 15.1).
    ui_dir = _resolve_ui_dir()
    if ui_dir is not None:
        _mount_ui(app, ui_dir)

    return app


__all__ = [
    "AppDeps",
    "create_app",
    "StoreCleanup",
    "StoreFactory",
    "SupervisorFactory",
]
