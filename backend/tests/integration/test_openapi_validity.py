"""Task 26.5 — OpenAPI document validity.

Requirement 13.7: the Backend exposes a machine-readable OpenAPI
schema the UI's generated TypeScript client depends on. The schema
must parse as a valid OpenAPI 3.x document and stay in lockstep with
the committed ``shared/openapi.yaml`` artifact.

What this test covers:

* ``GET /openapi.json`` responds 200 with a JSON body.
* The body has the ``openapi`` field set to a ``"3.x"`` version
  string (FastAPI emits 3.1 by default in 0.110+; either 3.0 or 3.1
  satisfies the requirement).
* Every expected top-level path is present in ``paths``:
  ``/api/health``, ``/api/models``, ``/api/suites``, ``/api/runs``,
  ``/api/compare``.
* The committed ``shared/openapi.yaml`` parses as valid YAML and
  contains the same top-level path set as the live schema. Full
  byte-for-byte equivalence is already pinned by
  ``tests/unit/test_shared_schemas_regen.py``; this test adds a
  weaker structural equivalence check so a failure here points at
  path drift specifically rather than any whitespace change.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from ruamel.yaml import YAML

from ollama_evaluator.api.app import AppDeps, create_app
from ollama_evaluator.api.supervisor import RunSupervisor
from ollama_evaluator.history.store import HistoryStore


_EXPECTED_PATHS = frozenset(
    {
        "/api/health",
        "/api/models",
        "/api/suites",
        "/api/runs",
        "/api/compare",
    }
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_OPENAPI = _REPO_ROOT / "shared" / "openapi.yaml"


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Minimal FastAPI :class:`TestClient` with an in-memory store."""
    suites_dir = tmp_path / "suites"
    suites_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setenv("OLLAMA_EVAL_UI_DIR", str(tmp_path / "missing-ui"))

    loop = asyncio.new_event_loop()

    async def _open() -> tuple[HistoryStore, Any]:
        cm = HistoryStore.open(":memory:", runs_dir)
        store = await cm.__aenter__()
        return store, cm

    store, cm = loop.run_until_complete(_open())
    supervisor = RunSupervisor(
        store,
        suites_dir=suites_dir,
        output_dir=runs_dir,
    )
    deps = AppDeps(
        store=store,
        supervisor=supervisor,
        suites_dir=suites_dir,
        output_dir=runs_dir,
    )
    app = create_app(deps)
    with TestClient(app) as client:
        yield client

    loop.run_until_complete(cm.__aexit__(None, None, None))
    loop.close()


def test_openapi_json_endpoint_returns_3x_document(app_client: TestClient) -> None:
    """``GET /openapi.json`` returns a valid OpenAPI 3.x JSON document."""
    response = app_client.get("/openapi.json")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")

    doc = response.json()
    assert isinstance(doc, dict)

    # ``openapi`` field — FastAPI 0.110+ defaults to 3.1, older
    # versions to 3.0. Both satisfy Requirement 13.7.
    version = doc.get("openapi")
    assert isinstance(version, str), f"openapi field missing or non-string: {version!r}"
    assert version.startswith("3."), f"expected OpenAPI 3.x, got {version!r}"

    # Paths contract — every canonical endpoint the UI depends on.
    paths = doc.get("paths")
    assert isinstance(paths, dict), "paths missing from OpenAPI doc"
    missing = _EXPECTED_PATHS - set(paths.keys())
    assert not missing, f"missing expected paths in OpenAPI doc: {sorted(missing)!r}"


def test_committed_shared_openapi_is_valid_yaml() -> None:
    """``shared/openapi.yaml`` parses as valid YAML (Requirement 13.7)."""
    assert _SHARED_OPENAPI.exists(), (
        f"shared/openapi.yaml missing at {_SHARED_OPENAPI}; "
        "run `python backend/scripts/regen_schemas.py`"
    )
    text = _SHARED_OPENAPI.read_text(encoding="utf-8")
    parsed = YAML(typ="safe").load(text)
    assert isinstance(parsed, dict), (
        f"shared/openapi.yaml did not parse to a mapping; got {type(parsed).__name__}"
    )
    # Sanity: a parsed OpenAPI 3.x doc has an ``openapi`` key starting with "3.".
    version = parsed.get("openapi")
    assert isinstance(version, str) and version.startswith("3."), (
        f"shared/openapi.yaml has unexpected openapi version {version!r}"
    )


def test_live_and_committed_openapi_have_same_top_level_paths(
    app_client: TestClient,
) -> None:
    """Live ``/openapi.json`` and committed YAML agree on the path set.

    Byte-for-byte equivalence is already enforced by
    ``test_shared_schemas_regen.py``; this test adds a narrower
    structural check so a failure here specifically indicates a
    drift in the *routes*, not in some unrelated schema field.
    """
    live = app_client.get("/openapi.json").json()
    committed = YAML(typ="safe").load(_SHARED_OPENAPI.read_text(encoding="utf-8"))

    live_paths = set(live.get("paths", {}).keys())
    committed_paths = set(committed.get("paths", {}).keys())

    # Both must contain every expected endpoint.
    assert _EXPECTED_PATHS <= live_paths, (
        f"live OpenAPI missing {sorted(_EXPECTED_PATHS - live_paths)!r}"
    )
    assert _EXPECTED_PATHS <= committed_paths, (
        f"committed openapi.yaml missing {sorted(_EXPECTED_PATHS - committed_paths)!r}"
    )

    # And they must agree on the full path set. If ``regen_schemas.py``
    # has been run since the last change this is guaranteed; failure
    # points at drift and the user fixing it is a single command.
    assert live_paths == committed_paths, (
        "shared/openapi.yaml path set drifted from live FastAPI app. "
        "Run `python backend/scripts/regen_schemas.py` to refresh. "
        f"only-in-live={sorted(live_paths - committed_paths)!r}, "
        f"only-in-committed={sorted(committed_paths - live_paths)!r}"
    )
