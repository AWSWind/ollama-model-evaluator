"""CI guard: committed ``shared/*`` artifacts must match regenerated output.

When a developer changes a Pydantic model or adds a new REST endpoint,
``shared/openapi.yaml`` and the two ``.schema.json`` files need to be
regenerated via ``python backend/scripts/regen_schemas.py``. This test
runs the same regeneration against a scratch directory and diffs it
byte-for-byte against the committed copy, so the CI guard message
points users at the exact command to run.

Requirement 13.7: the shared schemas are a wire contract with the UI;
drift between the committed copy and the live Pydantic models would
cause the generated TypeScript client to be out of sync with the
Backend.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_DIR = _REPO_ROOT / "shared"
_ARTIFACTS = (
    "openapi.yaml",
    "evaluation-suite.schema.json",
    "run-report.schema.json",
)


@pytest.fixture(scope="module")
def regenerated_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Regenerate every shared artifact into a fresh scratch directory."""
    # Import lazily so the scripts directory is not required to be on
    # sys.path at test collection time.
    import sys

    script_dir = _REPO_ROOT / "backend" / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import regen_schemas  # type: ignore[import-not-found]

    out = tmp_path_factory.mktemp("shared_regen")
    regen_schemas.regenerate(out)
    return out


@pytest.mark.parametrize("name", _ARTIFACTS)
def test_committed_schema_matches_regenerated(name: str, regenerated_dir: Path) -> None:
    """Committed ``shared/<name>`` must match a freshly-generated copy.

    If this test fails, run::

        python backend/scripts/regen_schemas.py

    to refresh the committed schemas.
    """
    committed = _SHARED_DIR / name
    regenerated = regenerated_dir / name
    assert committed.exists(), f"missing committed {name!s}"
    assert regenerated.exists(), f"regenerator did not produce {name!s}"

    committed_text = committed.read_text(encoding="utf-8")
    regenerated_text = regenerated.read_text(encoding="utf-8")

    if committed_text != regenerated_text:
        # Use ``filecmp`` for a clean failure message when sizes differ.
        same = filecmp.cmp(committed, regenerated, shallow=False)
        pytest.fail(
            f"shared/{name} is stale. Run "
            "'python backend/scripts/regen_schemas.py' to refresh. "
            f"(byte-equal={same}, committed_bytes={len(committed_text)}, "
            f"regenerated_bytes={len(regenerated_text)})"
        )
