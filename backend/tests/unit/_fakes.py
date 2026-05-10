"""Reusable fakes for unit tests.

Currently hosts :class:`FakeOllamaClient`, a lightweight in-memory
substitute for :class:`ollama_evaluator.ollama.client.OllamaClient`
used by the runner unit tests (``test_runner_scheduler.py``) and the
runner property tests (``tests/property/test_property_*.py``).

The existing ASGI-level fake lives at
``tests/integration/fakes/ollama_server.py`` and sits behind an
``httpx.AsyncClient`` transport. That fake is appropriate for
integration tests of the HTTP boundary but is overkill for unit tests
of the scheduler: the scheduler only needs an object that quacks like
:class:`OllamaClient` (``version``, ``list_models``, ``generate``,
``pull_model``). :class:`FakeOllamaClient` skips the HTTP layer
entirely so the tests stay pure asyncio, deterministic, and fast.

Scripting API is deliberately narrow so it's obvious what a test is
doing at a glance:

* :meth:`set_version` — fix the string returned from :meth:`version`.
* :meth:`set_models` — fix the list returned from :meth:`list_models`.
* :meth:`set_generate_chunks` — fix the chunks streamed from
  :meth:`generate`. Accepts either a fixed list (applied to every
  call) or a callable that maps ``(model, prompt, system, options)``
  to a list — the callable form lets property tests branch on the
  inputs to simulate failures or varying token counts.
* :meth:`set_generate_failure_plan` — install a
  ``name → list[exception_or_None]`` plan that raises the next
  exception in the sequence for each call keyed by a plan-label.
  Used by the retry property test.
* :meth:`set_generate_raise` — unconditionally raise the given
  exception on every call. Handy for failure-isolation tests.

Two auxiliary hooks are exposed for assertions the tests want to make:

* :attr:`concurrency_observed` — the maximum number of concurrently
  in-flight :meth:`generate` calls seen since construction (Property 8).
* :attr:`dispatch_log` — ordered list of ``(model, prompt, repetition
  marker)`` tuples the test harness can consult to build bijection
  proofs (Property 32-flavoured unit checks).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from ollama_evaluator.ollama.errors import OllamaHTTPError
from ollama_evaluator.ollama.types import (
    GenerateChunk,
    GenerateOptions,
    OllamaModelInfo,
    PullProgress,
)

# A "chunk factory" is either a fixed list of GenerateChunk objects or
# a callable that takes the Generate arguments and returns such a
# list. Using a Union keeps the simple case simple while still allowing
# property tests to simulate per-call behaviour.
ChunkFactory = (
    Iterable[GenerateChunk]
    | Callable[[str, str, str | None, GenerateOptions | None], Iterable[GenerateChunk]]
)


class FakeOllamaClient:
    """In-memory stand-in for :class:`OllamaClient`.

    All methods match the signatures on the real client. Async methods
    stay async to keep the call-graph the same shape as production.

    Construction is synchronous and cheap. Every scripting field has
    a sensible default so an unconfigured instance still returns
    well-typed results for the non-failure paths.
    """

    def __init__(self) -> None:
        self._version: str = "0.1.32"
        self._models: list[OllamaModelInfo] = []
        self._chunk_factory: ChunkFactory = [
            _default_final_chunk("fake"),
        ]
        # Per-call failure plan: first pop drives the next call
        # matching the plan's key. ``None`` means "succeed this call";
        # anything else is raised.
        self._failure_plan: list[BaseException | None] | None = None
        self._generate_raise: BaseException | None = None
        self._pulled_models: list[str] = []
        self._pull_raise: BaseException | None = None

        # Instrumentation --------------------------------------------------
        self._in_flight = 0
        self._in_flight_lock = asyncio.Lock()
        self.concurrency_observed = 0
        self.dispatch_log: list[tuple[str, str]] = []
        self.version_calls = 0
        self.list_models_calls = 0

    # ------------------------------------------------------------------
    # Scripting API
    # ------------------------------------------------------------------

    def set_version(self, version: str) -> None:
        self._version = version

    def set_version_raise(self, exc: BaseException) -> None:
        """Make :meth:`version` raise ``exc`` on the next call."""
        self._version_raise = exc

    _version_raise: BaseException | None = None

    def set_models(self, models: Iterable[OllamaModelInfo]) -> None:
        self._models = list(models)

    def set_generate_chunks(self, chunks: ChunkFactory) -> None:
        """Install either a fixed chunk list or a per-call factory callable."""
        self._chunk_factory = chunks

    def set_generate_raise(self, exc: BaseException | None) -> None:
        """Unconditionally raise ``exc`` from every :meth:`generate` call.

        ``None`` clears the override (subsequent calls again return
        the configured chunk factory).
        """
        self._generate_raise = exc

    def set_generate_failure_plan(
        self, plan: list[BaseException | None]
    ) -> None:
        """Install a sequence of per-call outcomes for :meth:`generate`.

        Each call pops the next element: ``None`` → success, anything
        else → raise. When the plan is exhausted, subsequent calls
        fall through to the standard chunk factory (unchanged).

        Used by the retry property test to reproduce "n failures
        followed by either success or exhaustion" precisely.
        """
        self._failure_plan = list(plan)

    def set_pull_raise(self, exc: BaseException | None) -> None:
        """Unconditionally raise ``exc`` from every :meth:`pull_model` call."""
        self._pull_raise = exc

    # ------------------------------------------------------------------
    # OllamaClient-compatible surface
    # ------------------------------------------------------------------

    async def version(self) -> str:
        self.version_calls += 1
        if self._version_raise is not None:
            exc = self._version_raise
            self._version_raise = None
            raise exc
        return self._version

    async def list_models(self) -> list[OllamaModelInfo]:
        self.list_models_calls += 1
        return list(self._models)

    async def aclose(self) -> None:
        """No-op aclose so callers that use the context-manager pattern work."""
        return None

    async def pull_model(self, name: str) -> AsyncIterator[PullProgress]:
        if self._pull_raise is not None:
            raise self._pull_raise
        self._pulled_models.append(name)
        # Emit a minimal success stream so callers that iterate the
        # async iterator see the Ollama "success" status.
        yield PullProgress(status="pulling manifest")
        yield PullProgress(status="success")

    @property
    def pulled_models(self) -> list[str]:
        """Models that were successfully pulled through :meth:`pull_model`."""
        return list(self._pulled_models)

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """Stream back :class:`GenerateChunk`s according to the installed script.

        Order of precedence:

        1. Unconditional raise (``set_generate_raise``).
        2. Per-call failure plan (``set_generate_failure_plan``).
        3. Chunk factory (``set_generate_chunks``) — list or callable.
        """
        # Concurrency instrumentation: bump in-flight before producing
        # the first chunk, decrement after the caller exhausts the
        # stream. The ``try/finally`` ensures the decrement happens
        # even if the caller breaks mid-stream.
        async with self._in_flight_lock:
            self._in_flight += 1
            if self._in_flight > self.concurrency_observed:
                self.concurrency_observed = self._in_flight
        self.dispatch_log.append((model, prompt))

        # Resolve errors *before* producing the async iterator so the
        # caller's ``anext()`` sees the raise immediately, matching the
        # real client which fails before the first chunk.
        try:
            if self._generate_raise is not None:
                raise self._generate_raise
            if self._failure_plan is not None and self._failure_plan:
                outcome = self._failure_plan.pop(0)
                if outcome is not None:
                    raise outcome
            chunks = self._resolve_chunks(model, prompt, system, options)
            for chunk in chunks:
                # Yield cooperatively so the asyncio semaphore can see
                # the in-flight state transition.
                await asyncio.sleep(0)
                yield chunk
        finally:
            async with self._in_flight_lock:
                self._in_flight -= 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_chunks(
        self,
        model: str,
        prompt: str,
        system: str | None,
        options: GenerateOptions | None,
    ) -> list[GenerateChunk]:
        factory = self._chunk_factory
        if callable(factory):
            return list(factory(model, prompt, system, options))
        return list(factory)


def _default_final_chunk(model: str) -> GenerateChunk:
    """Build a done-chunk with non-``None`` timing + token counts."""
    return GenerateChunk(
        model=model,
        created_at=datetime.now(tz=timezone.utc),
        response="",
        done=True,
        total_duration=1_000_000_000,  # 1 s in nanoseconds
        load_duration=500_000_000,
        prompt_eval_count=3,
        prompt_eval_duration=300_000_000,
        eval_count=5,
        eval_duration=700_000_000,
    )


def make_chunks(
    text: str,
    *,
    model: str = "fake",
    total_duration_ns: int = 1_000_000_000,
    prompt_eval_count: int | None = 3,
    eval_count: int | None = 5,
    include_final: bool = True,
) -> list[GenerateChunk]:
    """Helper that builds a realistic partial + final chunk sequence.

    The partial chunk carries the full response text so the runner's
    TTFT measurement (first non-empty chunk) fires immediately; the
    final chunk carries the timing + token counts that populate
    :class:`PerformanceMetrics`.

    Pass ``include_final=False`` to exercise the "server never marks
    the stream done" edge case the scheduler needs to tolerate.
    """
    now = datetime.now(tz=timezone.utc)
    partial = GenerateChunk(
        model=model,
        created_at=now,
        response=text,
        done=False,
    )
    if not include_final:
        return [partial]
    final = GenerateChunk(
        model=model,
        created_at=now,
        response="",
        done=True,
        total_duration=total_duration_ns,
        load_duration=0,
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration=0,
        eval_count=eval_count,
        eval_duration=total_duration_ns,
    )
    return [partial, final]


def make_http_error(status: int = 503, body: str = "") -> OllamaHTTPError:
    """Build an :class:`OllamaHTTPError` for use in failure plans."""
    return OllamaHTTPError(status=status, url="/api/generate", body=body)


__all__ = [
    "ChunkFactory",
    "FakeOllamaClient",
    "make_chunks",
    "make_http_error",
]
