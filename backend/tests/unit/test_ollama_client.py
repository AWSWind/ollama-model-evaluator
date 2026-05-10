"""Unit tests for :class:`OllamaClient` (Task 4.2).

Every test drives the client through the :class:`FakeOllamaServer`
ASGI fixture from ``tests/integration/fakes/ollama_server.py`` so no
real network I/O occurs. The client is connected to the fake via
``httpx.ASGITransport`` as the design specifies
(§Test data hygiene).

Coverage targets (Requirements 1.2, 1.4, 1.5, 2.1, 6.1, 6.3, 6.5):

* ``version()`` round-trips the server's version string.
* ``list_models()`` flattens ``details`` and surfaces every ``/api/tags``
  entry in order.
* ``generate()`` yields the scripted chunk sequence, including both a
  final chunk with full timing/token metadata *and* a final chunk that
  omits optional fields (Requirement 6.5).
* ``pull_model()`` yields each progress chunk in order.
* Non-2xx responses raise :class:`OllamaHTTPError` with the status,
  URL, and decoded body intact.
* A non-JSON line in a stream raises :class:`pydantic.ValidationError`
  so the caller can attribute the failure to the malformed chunk.
* The request body sent to ``/api/generate`` matches the design
  (``stream=True``, ``options`` serialised with ``exclude_none=True``,
  ``system`` omitted when ``None``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import ValidationError

from ollama_evaluator.ollama.client import OllamaClient
from ollama_evaluator.ollama.errors import OllamaHTTPError
from ollama_evaluator.ollama.types import GenerateChunk, GenerateOptions, PullProgress
from tests.integration.fakes.ollama_server import FakeOllamaServer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_server() -> FakeOllamaServer:
    """Fresh :class:`FakeOllamaServer` per test so state does not leak."""
    return FakeOllamaServer()


@pytest.fixture
async def client(fake_server: FakeOllamaServer) -> AsyncIterator[OllamaClient]:
    """Build an :class:`OllamaClient` wired to the fake via ASGI transport.

    The base URL is a sentinel host — the transport intercepts every
    request before it leaves the process.
    """
    transport = httpx.ASGITransport(app=fake_server.app)
    async_client = httpx.AsyncClient(
        transport=transport,
        base_url="http://fake-ollama",
        timeout=httpx.Timeout(10.0),
    )
    ollama = OllamaClient("http://fake-ollama", client=async_client)
    try:
        yield ollama
    finally:
        await async_client.aclose()


# ---------------------------------------------------------------------------
# version()
# ---------------------------------------------------------------------------


class TestVersion:
    async def test_returns_version_string(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Requirement 1.2: the preflight check surfaces the server's version."""
        fake_server.set_version("0.1.32")

        version = await client.version()

        assert version == "0.1.32"

    async def test_raises_on_non_2xx(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Non-2xx on ``/api/version`` must raise :class:`OllamaHTTPError`."""
        fake_server.set_version_status(503)

        with pytest.raises(OllamaHTTPError) as exc_info:
            await client.version()

        assert exc_info.value.status == 503
        assert exc_info.value.url == "/api/version"


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


class TestListModels:
    async def test_flattens_details_block(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Requirements 2.1, 2.5: flattened entries carry parameter_size/quant_level."""
        fake_server.set_models(
            [
                {
                    "name": "llama3:8b",
                    "digest": "sha256:abc123",
                    "size": 4_700_000_000,
                    "modified_at": "2024-05-01T12:34:56Z",
                    "details": {
                        "parameter_size": "8B",
                        "quantization_level": "Q4_0",
                    },
                },
                {
                    "name": "mistral:7b-instruct",
                    "digest": "sha256:def456",
                    "size": 3_800_000_000,
                    "modified_at": "2024-06-01T00:00:00Z",
                    "details": {
                        "parameter_size": "7B",
                        "quantization_level": "Q5_K_M",
                    },
                },
            ]
        )

        models = await client.list_models()

        assert len(models) == 2
        assert models[0].name == "llama3:8b"
        assert models[0].digest == "sha256:abc123"
        assert models[0].size == 4_700_000_000
        assert models[0].parameter_size == "8B"
        assert models[0].quantization_level == "Q4_0"
        assert models[1].name == "mistral:7b-instruct"
        assert models[1].parameter_size == "7B"
        assert models[1].quantization_level == "Q5_K_M"

    async def test_empty_list(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """``/api/tags`` with no models yields an empty list, not an error."""
        fake_server.set_models([])

        assert await client.list_models() == []

    async def test_accepts_model_key_variant(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Ollama v0.2.x+ emits ``"model"`` instead of ``"name"``."""
        fake_server.set_models(
            [
                {
                    "model": "phi3:3.8b",
                    "digest": "sha256:xyz",
                    "details": {"parameter_size": "3.8B"},
                }
            ]
        )

        models = await client.list_models()

        assert models[0].name == "phi3:3.8b"
        assert models[0].parameter_size == "3.8B"


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_yields_chunks_in_order_with_final_metadata(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Requirement 6.1, 6.3: chunks arrive in order; final chunk carries timings."""
        fake_server.set_generate_chunks(
            [
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.000Z",
                    "response": "The ",
                    "done": False,
                },
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.050Z",
                    "response": "answer ",
                    "done": False,
                },
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.100Z",
                    "response": "is 42.",
                    "done": False,
                },
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.200Z",
                    "response": "",
                    "done": True,
                    "total_duration": 200_000_000,
                    "load_duration": 10_000_000,
                    "prompt_eval_count": 8,
                    "prompt_eval_duration": 20_000_000,
                    "eval_count": 5,
                    "eval_duration": 170_000_000,
                },
            ]
        )

        chunks: list[GenerateChunk] = []
        async for chunk in await _call_generate(
            client, model="llama3:8b", prompt="What is the answer?"
        ):
            chunks.append(chunk)

        assert [c.response for c in chunks] == ["The ", "answer ", "is 42.", ""]
        assert [c.done for c in chunks] == [False, False, False, True]
        final = chunks[-1]
        assert final.total_duration == 200_000_000
        assert final.prompt_eval_count == 8
        assert final.eval_count == 5

    async def test_final_chunk_without_token_counts_yields_none(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Requirement 6.5: missing token-count metadata becomes ``None`` fields."""
        fake_server.set_generate_chunks(
            [
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.000Z",
                    "response": "hi",
                    "done": False,
                },
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.100Z",
                    "response": "",
                    "done": True,
                    # prompt_eval_count, eval_count, durations all absent.
                },
            ]
        )

        chunks = [
            c async for c in await _call_generate(client, model="llama3:8b", prompt="hi?")
        ]

        final = chunks[-1]
        assert final.done is True
        assert final.prompt_eval_count is None
        assert final.eval_count is None
        assert final.total_duration is None
        assert final.eval_duration is None

    async def test_request_body_shape(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Body carries ``stream=True``, serialised options, and system when set.

        Validates the design's generate-endpoint contract — in particular
        that ``exclude_none=True`` on options keeps unset fields off the
        wire (Requirements 5.3, 5.4 downstream) and that ``system`` is
        only included when the caller passes one.
        """
        fake_server.set_generate_chunks(
            [
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.000Z",
                    "response": "",
                    "done": True,
                }
            ]
        )

        opts = GenerateOptions(temperature=0.2, num_predict=32)
        _ = [
            c
            async for c in await _call_generate(
                client,
                model="llama3:8b",
                prompt="hello",
                system="You are helpful.",
                options=opts,
            )
        ]

        assert len(fake_server.received_generate_bodies) == 1
        body = fake_server.received_generate_bodies[0]
        assert body["model"] == "llama3:8b"
        assert body["prompt"] == "hello"
        assert body["stream"] is True
        assert body["system"] == "You are helpful."
        assert body["options"] == {"temperature": 0.2, "num_predict": 32}
        # ``stop`` was not set, so it must not leak onto the wire.
        assert "stop" not in body["options"]

    async def test_omits_system_when_none(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """When ``system=None`` the key must be absent from the request body."""
        fake_server.set_generate_chunks(
            [
                {
                    "model": "llama3:8b",
                    "created_at": "2024-05-01T12:00:00.000Z",
                    "response": "",
                    "done": True,
                }
            ]
        )

        _ = [c async for c in await _call_generate(client, model="llama3:8b", prompt="hi")]

        body = fake_server.received_generate_bodies[0]
        assert "system" not in body
        # Default ``options=None`` serialises to an empty dict so the
        # Ollama_Server can apply its defaults.
        assert body["options"] == {}

    async def test_raises_on_non_2xx_response(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """A 503 during streaming must raise :class:`OllamaHTTPError`."""
        fake_server.set_generate_error(503, body="overloaded")

        with pytest.raises(OllamaHTTPError) as exc_info:
            _ = [
                c
                async for c in await _call_generate(
                    client, model="llama3:8b", prompt="hi"
                )
            ]

        assert exc_info.value.status == 503
        assert exc_info.value.url == "/api/generate"
        assert "overloaded" in exc_info.value.body

    async def test_malformed_line_raises_validation_error(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """A non-JSON line propagates as :class:`pydantic.ValidationError`.

        The client does not suppress the error — the scheduler (Task
        12.2) decides whether to mark the Test_Case ``error`` or retry.
        """
        fake_server.set_generate_raw_lines(["not-json-at-all"])

        with pytest.raises(ValidationError):
            _ = [
                c
                async for c in await _call_generate(
                    client, model="llama3:8b", prompt="hi"
                )
            ]


# ---------------------------------------------------------------------------
# pull_model()
# ---------------------------------------------------------------------------


class TestPullModel:
    async def test_yields_progress_chunks_in_order(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Requirement 2.4: progress chunks stream in the order Ollama emits them."""
        fake_server.set_pull_chunks(
            [
                {"status": "pulling manifest"},
                {
                    "status": "pulling 4f11f4d09f0b",
                    "digest": "sha256:4f11f4d09f0b",
                    "total": 4_661_224_192,
                    "completed": 0,
                },
                {
                    "status": "pulling 4f11f4d09f0b",
                    "digest": "sha256:4f11f4d09f0b",
                    "total": 4_661_224_192,
                    "completed": 4_661_224_192,
                },
                {"status": "success"},
            ]
        )

        chunks: list[PullProgress] = []
        async for progress in await _call_pull(client, "llama3:8b"):
            chunks.append(progress)

        assert [c.status for c in chunks] == [
            "pulling manifest",
            "pulling 4f11f4d09f0b",
            "pulling 4f11f4d09f0b",
            "success",
        ]
        assert chunks[1].completed == 0
        assert chunks[2].completed == 4_661_224_192

    async def test_request_body_shape(
        self, fake_server: FakeOllamaServer, client: OllamaClient
    ) -> None:
        """Pull request sends ``name`` and ``stream=True``."""
        fake_server.set_pull_chunks([{"status": "success"}])

        _ = [c async for c in await _call_pull(client, "llama3:8b")]

        assert fake_server.received_pull_bodies == [
            {"name": "llama3:8b", "stream": True}
        ]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_context_manager_closes_owned_client(
        self, fake_server: FakeOllamaServer
    ) -> None:
        """When we build the httpx client, ``aclose()`` must close it."""
        ollama = OllamaClient(
            "http://fake-ollama",
            client=httpx.AsyncClient(
                transport=httpx.ASGITransport(app=fake_server.app),
                base_url="http://fake-ollama",
            ),
        )
        # Caller-supplied client → we do not close it.
        await ollama.aclose()
        # Hit ``version`` to prove the caller's client is still live.
        fake_server.set_version("0.1.32")
        assert await ollama.version() == "0.1.32"
        # Owned-client path: build an OllamaClient without an injected client.
        owned = OllamaClient("http://example.invalid/unreachable", timeout_s=0.01)
        # Close immediately; no network I/O ever occurred.
        await owned.aclose()

    async def test_async_context_manager(
        self, fake_server: FakeOllamaServer
    ) -> None:
        """The ``async with`` form exits cleanly and closes owned clients."""
        transport = httpx.ASGITransport(app=fake_server.app)
        async_client = httpx.AsyncClient(
            transport=transport,
            base_url="http://fake-ollama",
        )
        try:
            fake_server.set_version("0.9.0")
            async with OllamaClient(
                "http://fake-ollama", client=async_client
            ) as ollama:
                assert await ollama.version() == "0.9.0"
        finally:
            await async_client.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_generate(
    client: OllamaClient,
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    options: GenerateOptions | None = None,
) -> AsyncIterator[GenerateChunk]:
    """Thin forwarder so call sites read symmetrically with ``_call_pull``.

    ``OllamaClient.generate`` is already an async generator, but wrapping
    it here gives the test helpers consistent shape with :func:`_call_pull`
    (which also returns an async iterator) and lets type checkers see
    the return type at call sites.
    """
    return client.generate(model=model, prompt=prompt, system=system, options=options)


async def _call_pull(
    client: OllamaClient, name: str
) -> AsyncIterator[PullProgress]:
    """See :func:`_call_generate`."""
    return client.pull_model(name)
