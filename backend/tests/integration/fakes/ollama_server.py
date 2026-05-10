"""In-process fake of the Ollama_Server HTTP API.

``FakeOllamaServer`` is a Starlette ASGI application that responds to
the four endpoints :class:`~ollama_evaluator.ollama.client.OllamaClient`
calls:

* ``GET  /api/version``
* ``GET  /api/tags``
* ``POST /api/pull``   (streaming NDJSON)
* ``POST /api/generate`` (streaming NDJSON)

The fake is deliberately scriptable from the outside so each test can
set up exactly the behaviour it needs (a specific version string, a
set of models, a scripted sequence of generate chunks, or a
non-2xx response). It never reaches the network.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Test data hygiene. Task 4.3 requires this fake to be reusable by
integration tests, hence its home under ``tests/integration/fakes/``.

Wire into a test via :class:`httpx.ASGITransport` so
:class:`httpx.AsyncClient` dispatches in-process::

    server = FakeOllamaServer()
    server.set_models([...])
    transport = httpx.ASGITransport(app=server.app)
    httpx_client = httpx.AsyncClient(transport=transport, base_url="http://ollama")
    client = OllamaClient("http://ollama", client=httpx_client)

The ``base_url`` is arbitrary (the transport captures every request
before it hits the network); using a memorable host string keeps the
test logs readable.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


class FakeOllamaServer:
    """Scriptable ASGI fake of the Ollama_Server.

    The fake exposes an ``.app`` attribute holding a Starlette
    application wired to the four Ollama endpoints the Backend uses.
    Tests configure the fake by calling setter methods *before*
    dispatching the request under test; each setter replaces any
    previous value for that endpoint.

    Two fields need scripted streams (``generate`` and ``pull``);
    for those the fake accepts a list of pre-serialised JSON *lines*
    so a test can control both the number and the content of emitted
    chunks (including malformed lines for negative-path tests).

    Attributes:
        app: The Starlette ASGI application; pass to
            :class:`httpx.ASGITransport`.
        received_generate_bodies: In-order list of request bodies
            received on ``POST /api/generate``. Tests can assert
            against this to verify the Backend assembled the correct
            payload (system prompt handling, options serialisation,
            etc.).
        received_pull_bodies: In-order list of request bodies received
            on ``POST /api/pull``.
    """

    def __init__(self) -> None:
        self._version = "0.1.32"
        self._models: list[dict[str, Any]] = []
        # Default: a single partial chunk followed by a done chunk with
        # populated timing/token counts. Tests override via
        # :meth:`set_generate_chunks` and :meth:`set_generate_status`.
        self._generate_lines: list[str] = []
        self._generate_status: int = 200
        self._generate_body_on_error: str = ""
        self._pull_lines: list[str] = []
        self._pull_status: int = 200
        self._tags_status: int = 200
        self._version_status: int = 200

        self.received_generate_bodies: list[dict[str, Any]] = []
        self.received_pull_bodies: list[dict[str, Any]] = []

        self.app = Starlette(
            routes=[
                Route("/api/version", self._handle_version, methods=["GET"]),
                Route("/api/tags", self._handle_tags, methods=["GET"]),
                Route("/api/generate", self._handle_generate, methods=["POST"]),
                Route("/api/pull", self._handle_pull, methods=["POST"]),
            ]
        )

    # ------------------------------------------------------------------
    # Scripting API
    # ------------------------------------------------------------------

    def set_version(self, version: str) -> None:
        """Override the string returned from ``GET /api/version``."""
        self._version = version

    def set_version_status(self, status: int) -> None:
        """Override the HTTP status returned from ``GET /api/version``."""
        self._version_status = status

    def set_models(self, models: Iterable[dict[str, Any]]) -> None:
        """Set the raw entries returned under ``"models"`` on ``GET /api/tags``.

        Pass the full Ollama-style entries including ``"details"``;
        the :class:`OllamaClient` flattens them via ``_parse_tags_entry``
        so the fake must emit the server's native shape.
        """
        self._models = list(models)

    def set_tags_status(self, status: int) -> None:
        """Override the HTTP status returned from ``GET /api/tags``."""
        self._tags_status = status

    def set_generate_chunks(self, chunks: Iterable[dict[str, Any]]) -> None:
        """Set the sequence of chunks streamed from ``POST /api/generate``.

        Each dict is serialised to one line of NDJSON in the response.
        Use :meth:`set_generate_raw_lines` for tests that need to emit
        malformed JSON (for example a property test on validation
        error handling).
        """
        self._generate_lines = [json.dumps(chunk) for chunk in chunks]

    def set_generate_raw_lines(self, lines: Iterable[str]) -> None:
        """Set the raw NDJSON lines streamed from ``POST /api/generate``.

        Lines are emitted verbatim. Use this when a test needs to emit
        a non-JSON line to exercise the validation path.
        """
        self._generate_lines = list(lines)

    def set_generate_error(self, status: int, body: str = "") -> None:
        """Make ``POST /api/generate`` respond with a non-2xx status.

        When ``status >= 400``, the server answers with the given
        ``body`` instead of streaming chunks. Used to test
        :class:`OllamaHTTPError` translation.
        """
        self._generate_status = status
        self._generate_body_on_error = body

    def set_pull_chunks(self, chunks: Iterable[dict[str, Any]]) -> None:
        """Set the sequence of chunks streamed from ``POST /api/pull``."""
        self._pull_lines = [json.dumps(chunk) for chunk in chunks]

    def set_pull_status(self, status: int) -> None:
        """Override the HTTP status returned from ``POST /api/pull``."""
        self._pull_status = status

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_version(self, request: Request) -> Response:
        del request  # unused
        if self._version_status >= 400:
            return Response(
                content="version endpoint error",
                status_code=self._version_status,
            )
        return JSONResponse({"version": self._version})

    async def _handle_tags(self, request: Request) -> Response:
        del request  # unused
        if self._tags_status >= 400:
            return Response(
                content="tags endpoint error",
                status_code=self._tags_status,
            )
        return JSONResponse({"models": self._models})

    async def _handle_generate(self, request: Request) -> Response:
        body = await request.json()
        self.received_generate_bodies.append(body)
        if self._generate_status >= 400:
            return Response(
                content=self._generate_body_on_error,
                status_code=self._generate_status,
            )
        return StreamingResponse(
            _line_stream(self._generate_lines),
            media_type="application/x-ndjson",
        )

    async def _handle_pull(self, request: Request) -> Response:
        body = await request.json()
        self.received_pull_bodies.append(body)
        if self._pull_status >= 400:
            return Response(
                content="pull endpoint error",
                status_code=self._pull_status,
            )
        return StreamingResponse(
            _line_stream(self._pull_lines),
            media_type="application/x-ndjson",
        )


async def _line_stream(lines: Iterable[str]) -> AsyncIterator[bytes]:
    """Yield each pre-serialised line with a trailing newline as bytes.

    Ollama's streaming format is NDJSON, so each JSON object is followed
    by ``\\n``. We materialise the list first so the iterator is
    decoupled from any mutation on the fake between request start and
    stream completion.
    """
    for line in list(lines):
        # Each chunk terminated by newline so ``httpx.aiter_lines`` can
        # split them.
        yield (line + "\n").encode("utf-8")


__all__ = ["FakeOllamaServer"]
