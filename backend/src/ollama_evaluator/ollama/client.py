"""Async HTTP client for the Ollama_Server.

Thin wrapper around :class:`httpx.AsyncClient` that exposes the four
Ollama REST endpoints the Backend uses during a Run:

* ``GET  /api/version`` — preflight reachability check (Requirements
  1.1, 1.2).
* ``GET  /api/tags`` — list locally-available models (Requirement 2.1).
* ``POST /api/pull`` — streaming model pull for the
  ``pull_missing_models`` option (Requirement 2.4).
* ``POST /api/generate`` — streaming generation for scoring (Requirements
  1.4, 5.3, 5.4, 6.1).

The client is deliberately minimal: it speaks HTTP, parses NDJSON, and
raises well-typed errors. It intentionally does **not**:

* Measure time-to-first-token. The runner (Task 12.1) owns the
  wall-clock because it already owns the event loop and the test-case
  lifecycle. Exposing raw async iterators from :meth:`generate` lets
  the runner time the first yield without the client having to return
  a side-band timing object — simpler surface, same result.
* Retry on failure. The retry policy lives in the scheduler (Task 12.2)
  because it depends on ``RunConfig.retry_max_attempts`` and needs to
  interact with the run's ``asyncio.Semaphore`` for concurrency. The
  client raises the original httpx exceptions so the policy can
  pattern-match on them unchanged (see module docstring of
  :mod:`ollama_evaluator.ollama.errors`).
* Interpret streaming semantics beyond "parse each line as one
  chunk". The runner decides what a final chunk means, what counts as
  "first token", and how to compute ``tokens_per_second``.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Components and Interfaces > Ollama Client.

Key design decisions:

* **Caller-supplied ``httpx.AsyncClient``.** If the caller passes a
  client, we use it verbatim and never close it — that pattern is how
  FastAPI tests and the :class:`FakeOllamaServer` ASGI fixture inject
  an in-process transport. When we create the client ourselves,
  :meth:`aclose` closes it so the owner-side lifecycle is obvious at
  the call site.

* **``httpx.Timeout(total=timeout_s)``.** The design pins this to
  Requirement 1.5 — a single request-level timeout that covers
  connect, write, pool, and read. Streaming endpoints still honour the
  total: an Ollama generation that runs past the timeout raises
  :class:`httpx.ReadTimeout` mid-stream, which the retry policy treats
  as a terminal ``timeout`` status on the affected Test_Case.

* **NDJSON parsing via ``aiter_lines()``.** Ollama's streaming format
  is one JSON object per line. Using ``httpx`` line-aware iteration
  lets us convert chunks incrementally without buffering the whole
  body — the runner needs the first chunk for TTFT (Requirement 6.1)
  before the last one arrives.

* **Pydantic validation at the line level.** Every chunk is validated
  through :meth:`GenerateChunk.model_validate_json` /
  :meth:`PullProgress.model_validate_json`. A malformed line raises
  :class:`pydantic.ValidationError` which propagates unchanged — the
  caller decides whether to mark the Test_Case ``error`` or attempt
  another request.

* **Error translation only on HTTP status.** Network errors
  (``httpx.ConnectError``, ``httpx.ReadError``) and timeouts
  (``httpx.TimeoutException``) propagate unchanged because the retry
  policy keys off the concrete httpx type. Non-2xx responses become
  :class:`OllamaHTTPError` so the scheduler has a single type to
  classify by status code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import httpx

from .errors import OllamaHTTPError
from .types import (
    GenerateChunk,
    GenerateOptions,
    OllamaModelInfo,
    PullProgress,
    _parse_tags_entry,
)


class OllamaClient:
    """Async HTTP client for the Ollama_Server.

    The client is an ``async`` context manager: entering the context
    returns ``self`` and exiting calls :meth:`aclose`. Callers that
    want explicit lifecycle can skip the context manager and call
    :meth:`aclose` directly.

    Args:
        base_url: Base URL of the Ollama_Server, for example
            ``"http://localhost:11434"`` (Requirement 1.1). Used as the
            ``base_url`` of the underlying ``httpx.AsyncClient`` so
            methods can pass endpoint-relative paths.
        timeout_s: Per-request timeout in seconds (Requirement 1.4,
            default 120.0). Applied as ``httpx.Timeout(total=...)`` so
            the same budget covers connect, write, read, and pool
            acquisition. For streaming endpoints the total covers the
            entire stream — an overly-long generation raises
            :class:`httpx.ReadTimeout` part-way through.
        client: Optional pre-built ``httpx.AsyncClient``. When
            provided, the caller owns the client's lifecycle and
            :meth:`aclose` is a no-op. This is the injection point for
            :class:`FakeOllamaServer` in tests, and for any caller
            that wants to share a client across components (for
            example the scheduler pooling connections).
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout_s = timeout_s
        if client is None:
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(timeout_s),
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if we own it.

        When the caller supplied their own client via the ``client``
        argument to :meth:`__init__`, this method is a no-op — lifecycle
        stays with the caller.
        """
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Simple (non-streaming) endpoints
    # ------------------------------------------------------------------

    async def version(self) -> str:
        """Return the Ollama_Server version string.

        Calls ``GET /api/version`` and returns the value of the
        response's ``"version"`` field. Used by the preflight step
        (Task 12.4) to satisfy Requirement 1.2's connectivity check.

        Raises:
            OllamaHTTPError: If the server responds with a non-2xx
                status. The 4xx/5xx body is preserved for debugging.
            httpx.ConnectError: If the server is unreachable. The
                caller (preflight) wraps this as
                :class:`OllamaConnectionError` when reporting
                ``ollama_unreachable`` to the user.
        """
        url = "/api/version"
        response = await self._client.get(url)
        self._raise_for_status(response, url)
        data: dict[str, Any] = response.json()
        return str(data["version"])

    async def list_models(self) -> list[OllamaModelInfo]:
        """List every model available on the Ollama_Server.

        Calls ``GET /api/tags`` and flattens each entry through
        :func:`_parse_tags_entry` so the returned records carry
        ``parameter_size`` and ``quantization_level`` alongside ``name``,
        ``digest``, ``size``, and ``modified_at`` (Requirement 2.5).

        The order returned by Ollama is preserved. An empty models list
        returns an empty list.

        Raises:
            OllamaHTTPError: On non-2xx responses.
            pydantic.ValidationError: If a ``/api/tags`` entry is
                structurally invalid (for example missing both
                ``"name"`` and ``"model"``).
        """
        url = "/api/tags"
        response = await self._client.get(url)
        self._raise_for_status(response, url)
        payload: dict[str, Any] = response.json()
        entries: list[dict[str, Any]] = payload.get("models") or []
        return [_parse_tags_entry(entry) for entry in entries]

    # ------------------------------------------------------------------
    # Streaming endpoints
    # ------------------------------------------------------------------

    async def pull_model(self, name: str) -> AsyncIterator[PullProgress]:
        """Pull a model by name, yielding progress chunks as they arrive.

        Calls ``POST /api/pull`` with ``{"name": name, "stream": true}``
        and converts each NDJSON line into :class:`PullProgress`. Used
        by the ``pull_missing_models`` preflight step (Requirement 2.4)
        to drive a progress indicator; callers typically iterate the
        stream to exhaustion and check the final ``status == "success"``
        chunk.

        The Ollama ``/api/pull`` endpoint does not enforce a small
        number of chunks — large models can emit hundreds of progress
        messages per layer. The caller's ``asyncio`` task remains
        responsive because each chunk is produced as soon as a full
        line arrives.

        Args:
            name: Ollama model tag, for example ``"llama3:8b"``.

        Yields:
            :class:`PullProgress` records in the order Ollama emits them.

        Raises:
            OllamaHTTPError: On non-2xx responses (including 404 for
                non-existent repository references).
            pydantic.ValidationError: On malformed NDJSON lines.
        """
        url = "/api/pull"
        body: dict[str, Any] = {"name": name, "stream": True}
        async for line in self._stream_lines("POST", url, json=body):
            yield PullProgress.model_validate_json(line)

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """Stream a generation, yielding :class:`GenerateChunk` per NDJSON line.

        Calls ``POST /api/generate`` with ``stream=true`` so the caller
        receives partial response chunks as they arrive. This is the
        primitive the runner uses to measure time-to-first-token
        (Requirement 6.1): the runner times ``await anext(stream)`` and
        records the delta against the request dispatch timestamp.

        The request body is ``{"model", "prompt", "stream", "options"}``
        with ``"system"`` included *only* when the caller passes a
        non-``None`` value — the Ollama_Server applies model-default
        system prompts when the field is absent but *not* when it is
        explicitly ``null``, so omission matters. ``options`` is
        serialised with ``exclude_none=True`` so unset generation
        parameters take the Ollama_Server default (Requirements 5.3,
        5.4 via ``GenerationDefaults`` in the caller).

        Args:
            model: Ollama model tag to target.
            prompt: User prompt for the generation.
            system: Optional system prompt. Omitted from the request
                body entirely when ``None``.
            options: Optional :class:`GenerateOptions`. When ``None``
                an empty ``options`` object is sent so the
                Ollama_Server applies its defaults.

        Yields:
            :class:`GenerateChunk` records in the order Ollama emits
            them. The final chunk has ``done=True`` and (if Ollama
            populated them) non-``None`` timing and token-count fields
            (Requirement 6.3). Partial chunks leave those fields as
            ``None`` (Requirement 6.5).

        Raises:
            OllamaHTTPError: On non-2xx responses.
            httpx.TimeoutException: If the total request time exceeds
                the configured ``timeout_s``. The retry policy in
                Task 12.2 records this as ``timeout`` on the Test_Case
                without retrying (Requirement 1.5).
            pydantic.ValidationError: On malformed NDJSON lines.
        """
        url = "/api/generate"
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options.model_dump(exclude_none=True) if options else {},
        }
        # Only include "system" when the caller explicitly set one —
        # absent vs. null has different semantics on the server side.
        if system is not None:
            body["system"] = system

        async for line in self._stream_lines("POST", url, json=body):
            yield GenerateChunk.model_validate_json(line)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _stream_lines(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Dispatch a streaming request and yield one non-empty line at a time.

        Uses ``httpx.AsyncClient.stream`` so the response body is not
        buffered and ``aiter_lines()`` can surface chunks as the
        Ollama_Server emits them. Empty keep-alive lines (which Ollama
        does not send today but the HTTP spec allows) are filtered out
        so downstream validators never see an empty string.

        Error handling:

        * Non-2xx response → :class:`OllamaHTTPError` with the decoded
          body. We call ``aread()`` before raising so the scheduler can
          log Ollama's error payload verbatim.
        * Network / timeout errors → propagated unchanged so the retry
          policy in Task 12.2 can pattern-match on the httpx type.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, etc.).
            url: Path relative to ``base_url`` or absolute URL.
            json: Optional request body to JSON-encode.

        Yields:
            Each non-empty line from the response, as an ``str``.
        """
        async with self._client.stream(method, url, json=json) as response:
            if response.status_code >= 400:
                # Materialise the body so the exception carries the
                # server's error payload. ``aread`` is mandatory before
                # inspecting ``.text`` on a streaming response.
                body_bytes = await response.aread()
                body = body_bytes.decode("utf-8", errors="replace")
                raise OllamaHTTPError(response.status_code, url, body)

            async for line in response.aiter_lines():
                # Ollama's NDJSON never emits blank lines today, but the
                # HTTP spec does permit them as keep-alives. Skip them
                # so Pydantic does not see empty input.
                if not line:
                    continue
                yield line

    def _raise_for_status(self, response: httpx.Response, url: str) -> None:
        """Raise :class:`OllamaHTTPError` for non-2xx responses on simple endpoints.

        Used by :meth:`version` and :meth:`list_models` — the streaming
        endpoints do their own status check before entering the line
        loop because they must call ``aread`` before reading ``.text``.
        """
        if response.status_code >= 400:
            raise OllamaHTTPError(response.status_code, url, response.text)


__all__ = ["OllamaClient"]
