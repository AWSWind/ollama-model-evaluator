"""Pytest fixtures shared by integration tests.

Every test collected under ``tests/integration/`` is automatically tagged with
the ``integration`` marker so it can be selected or skipped via
``pytest -m integration``. The marker itself is registered in ``pyproject.toml``.

Concrete fixtures (``FakeOllamaServer``, ``FakeHFHub``, FastAPI ``TestClient``)
are added in later tasks.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Automatically apply the ``integration`` marker to tests in this directory."""
    del config  # unused
    for item in items:
        if "tests/integration" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.integration)
