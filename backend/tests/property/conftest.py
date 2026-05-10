"""Pytest fixtures shared by property-based tests.

Every test collected under ``tests/property/`` is automatically tagged with the
``property`` marker so it can be selected or skipped via ``pytest -m property``.
The marker itself is registered in ``pyproject.toml``.

Hypothesis strategies live in ``tests/property/generators.py`` (added in later
tasks).
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Automatically apply the ``property`` marker to tests in this directory."""
    del config  # unused
    for item in items:
        if "tests/property" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.property)
