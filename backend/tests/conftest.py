"""Shared pytest configuration for the backend test suite.

Test-tier-specific fixtures live in `tests/unit/`, `tests/property/`, and
`tests/integration/` alongside their own `conftest.py` files. This top-level
module exists so pytest treats the `tests/` tree as a package and so future
shared fixtures have a natural home.
"""

from __future__ import annotations
