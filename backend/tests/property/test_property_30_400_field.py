"""Feature: ollama-model-evaluator, Property 30: 400 field identification.

Invalid ``POST /api/runs`` request bodies must return
``400 {error_code: "validation_failed", field, message}`` where
``field`` is the dotted path of the first failing field in document
order.

Validates: Requirement 13.6.

Approach: Hypothesis generates :class:`RunConfig`-shaped request bodies
that violate exactly one invariant (missing field, empty list, out-of-
range numeric). The expected ``field`` is computed directly from the
manipulated key; the assertion then checks that the API envelope
identifies the same key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.api.app import AppDeps, create_app
from ollama_evaluator.api.supervisor import RunSupervisor
from ollama_evaluator.history.store import HistoryStore


@pytest.fixture
def client(tmp_path: Path):
    async def _setup():
        cm = HistoryStore.open(":memory:", tmp_path / "runs")
        store = await cm.__aenter__()
        return store, cm

    loop = asyncio.new_event_loop()
    try:
        store, cm = loop.run_until_complete(_setup())
        (tmp_path / "suites").mkdir(parents=True, exist_ok=True)
        sup = RunSupervisor(
            store,
            suites_dir=tmp_path / "suites",
            output_dir=tmp_path / "runs",
        )
        deps = AppDeps(
            store=store,
            supervisor=sup,
            suites_dir=tmp_path / "suites",
            output_dir=tmp_path / "runs",
        )
        app = create_app(deps)
        with TestClient(app) as c:
            yield c
        loop.run_until_complete(cm.__aexit__(None, None, None))
    finally:
        loop.close()


def _valid_body() -> dict:
    """Return a minimally-valid :class:`RunConfig` request body."""
    return {
        "models": ["m1"],
        "suites": ["s1"],
        "repetitions": 1,
        "concurrency": 1,
        "retry_max_attempts": 2,
        "ollama_timeout_s": 60.0,
    }


# Each strategy returns a ``(body, expected_field_prefix)`` pair.
# ``expected_field_prefix`` is the dotted path fragment that the
# envelope's ``field`` must start with (it may carry list indices).
_invalid_body_strategies = st.one_of(
    # Missing ``models``.
    st.just(({"suites": ["s1"]}, "models")),
    # Missing ``suites``.
    st.just(({"models": ["m1"]}, "suites")),
    # Empty ``models`` list — rejected by validator on the field.
    st.just(({**_valid_body(), "models": []}, "models")),
    # Empty ``suites`` list.
    st.just(({**_valid_body(), "suites": []}, "suites")),
    # ``repetitions`` out of range (<1).
    st.integers(max_value=0).map(
        lambda v: ({**_valid_body(), "repetitions": v}, "repetitions")
    ),
    # ``concurrency`` out of range (<1).
    st.integers(max_value=0).map(
        lambda v: ({**_valid_body(), "concurrency": v}, "concurrency")
    ),
    # Negative ``retry_max_attempts``.
    st.integers(max_value=-1).map(
        lambda v: ({**_valid_body(), "retry_max_attempts": v}, "retry_max_attempts")
    ),
    # Zero or negative ``ollama_timeout_s``.
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False).map(
        lambda v: ({**_valid_body(), "ollama_timeout_s": v}, "ollama_timeout_s")
    ),
    # Unknown top-level field — rejected by ``extra="forbid"``.
    st.just(({**_valid_body(), "bogus_field": 42}, "bogus_field")),
    # Wrong type for ``models`` (list of int → string-coercion fails per-item).
    st.just(({**_valid_body(), "models": [123]}, "models")),
)


@given(case=_invalid_body_strategies)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_invalid_body_returns_validation_failed_envelope(
    client, case: tuple[dict, str]
) -> None:
    """**Validates: Requirement 13.6**

    Every invalid request body returns 400 with the standard envelope
    and ``field`` identifying the first failing field.
    """
    body, expected_prefix = case
    response = client.post("/api/runs", json=body)
    assert response.status_code == 400, response.text
    payload = response.json()
    assert set(payload.keys()) == {"error_code", "message", "field"}, payload
    assert payload["error_code"] == "validation_failed", payload
    assert isinstance(payload["message"], str) and payload["message"], payload
    field = payload["field"]
    assert field is not None, payload
    # The first path segment must match the expected top-level key.
    head = field.split(".", 1)[0]
    assert head == expected_prefix, (
        f"expected field to start with {expected_prefix!r}, got {field!r}"
    )
