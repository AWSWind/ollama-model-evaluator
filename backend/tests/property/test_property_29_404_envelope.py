"""Feature: ollama-model-evaluator, Property 29: 404 error envelope.

Every REST endpoint that references a missing ``run_id`` or
``suite_name`` (and, by extension, ``model_name``) returns
``404 {error_code, message, field}`` where
``error_code`` is one of ``run_not_found``, ``suite_not_found``,
``model_not_found``.

Validates: Requirement 13.5.

Approach: build a FastAPI :class:`TestClient` backed by an in-memory
:class:`HistoryStore` and an empty suites directory. Hypothesis
generates random ``run_id``/``suite_name`` values that cannot exist
and asserts the envelope shape and ``error_code`` on each endpoint.
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


_VALID_CODES = frozenset({"run_not_found", "suite_not_found", "model_not_found"})


@pytest.fixture
def client(tmp_path: Path):
    """Return a :class:`TestClient` wired to an isolated backend instance."""

    async def _setup():
        cm = HistoryStore.open(":memory:", tmp_path / "runs")
        store = await cm.__aenter__()
        return store, cm

    loop = asyncio.new_event_loop()
    try:
        store, cm = loop.run_until_complete(_setup())
        sup = RunSupervisor(
            store,
            suites_dir=tmp_path / "suites",
            output_dir=tmp_path / "runs",
        )
        # Ensure suites_dir exists so discover_suites does not error.
        (tmp_path / "suites").mkdir(parents=True, exist_ok=True)

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


# ASCII-only identifier strategy that is guaranteed not to clash with any
# real run_id / suite_name in the store (the store is empty).
_identifiers = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=20,
)


def _assert_envelope_404(response, expected_code: str) -> None:
    assert response.status_code == 404, (
        f"expected 404, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert set(body.keys()) == {"error_code", "message", "field"}, body
    assert body["error_code"] in _VALID_CODES, body
    assert body["error_code"] == expected_code, body
    assert isinstance(body["message"], str) and body["message"], body


@given(run_id=_identifiers)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_missing_run_id_returns_run_not_found(client, run_id: str) -> None:
    """**Validates: Requirement 13.5**

    ``GET /api/runs/{id}``, ``GET /api/runs/{id}/report.md``,
    ``DELETE /api/runs/{id}``, ``POST /api/runs/{id}/cancel`` all
    return 404 ``run_not_found`` for unknown ids.
    """
    for method, url in [
        ("GET", f"/api/runs/{run_id}"),
        ("GET", f"/api/runs/{run_id}/report.md"),
        ("DELETE", f"/api/runs/{run_id}"),
        ("POST", f"/api/runs/{run_id}/cancel"),
    ]:
        response = client.request(method, url)
        _assert_envelope_404(response, "run_not_found")


@given(suite_name=_identifiers)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_missing_suite_returns_suite_not_found(client, suite_name: str) -> None:
    """**Validates: Requirement 13.5**

    ``GET /api/suites/{name}`` returns 404 ``suite_not_found`` when the
    suite is not present in ``suites_dir``.
    """
    response = client.get(f"/api/suites/{suite_name}")
    _assert_envelope_404(response, "suite_not_found")


@given(run_id=_identifiers)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_compare_missing_run_returns_run_not_found(client, run_id: str) -> None:
    """**Validates: Requirement 13.5**

    ``GET /api/compare?a=&b=`` returns 404 ``run_not_found`` when
    either Run id is not in the store.
    """
    response = client.get("/api/compare", params={"a": run_id, "b": run_id})
    _assert_envelope_404(response, "run_not_found")
