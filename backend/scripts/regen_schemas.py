"""Regenerate ``shared/openapi.yaml`` and ``shared/*.schema.json`` artifacts.

Reads the Pydantic models in :mod:`ollama_evaluator` and the live
FastAPI application built by :func:`ollama_evaluator.api.app.create_app`
to produce three canonical schema artifacts committed under
``shared/``:

* ``shared/openapi.yaml`` — dumped via :meth:`FastAPI.openapi` and
  serialised with :func:`yaml.safe_dump`.
* ``shared/evaluation-suite.schema.json`` — from
  :meth:`EvaluationSuite.model_json_schema`.
* ``shared/run-report.schema.json`` — from
  :meth:`RunReport.model_json_schema`.

All three writes are atomic (write-temp-then-rename) so CI checks that
diff the committed copy against a fresh regeneration cannot observe a
half-written file.

Invocation::

    python backend/scripts/regen_schemas.py [shared_dir]

If ``shared_dir`` is omitted the script writes to
``<repo-root>/shared``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow `python scripts/regen_schemas.py` from inside ``backend/``.
_HERE = Path(__file__).resolve()
_BACKEND_DIR = _HERE.parent.parent  # backend/
_REPO_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR / "src"))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via write-temp-then-rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _build_stub_app() -> Any:
    """Construct a minimal :class:`FastAPI` to dump the OpenAPI schema.

    The real app requires a populated :class:`HistoryStore` and a
    :class:`RunSupervisor`; neither is needed to materialise the
    OpenAPI schema since FastAPI introspects the router statically.
    A lightweight stub dependency bag keeps us from touching the
    filesystem or Ollama while still exercising the same code path.
    """
    from ollama_evaluator.api.app import AppDeps, create_app

    class _StubSupervisor:
        async def start(self) -> None:  # pragma: no cover - unused in schema dump
            return None

        async def stop(self) -> None:  # pragma: no cover - unused in schema dump
            return None

        def get_state(self, run_id: str) -> None:  # pragma: no cover
            return None

        def get_bus(self, run_id: str) -> None:  # pragma: no cover
            return None

        def cancel(self, run_id: str) -> bool:  # pragma: no cover
            return False

        async def submit(self, config: Any) -> str:  # pragma: no cover
            return ""

    deps = AppDeps(
        store=object(),
        supervisor=_StubSupervisor(),
        suites_dir=Path("suites"),
        output_dir=Path("runs"),
    )
    return create_app(deps)


def _canonical_openapi_yaml(app: Any) -> str:
    """Dump ``app.openapi()`` to a deterministic YAML string."""
    import yaml  # imported lazily so `import regen_schemas` stays cheap

    schema = app.openapi()
    return yaml.safe_dump(schema, sort_keys=True, allow_unicode=False)


def _canonical_json_schema(model_cls: Any) -> str:
    """Dump ``model_cls.model_json_schema()`` as a pretty JSON string."""
    schema = model_cls.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def regenerate(shared_dir: Path) -> dict[str, Path]:
    """Regenerate every shared artifact under ``shared_dir`` atomically.

    Returns a mapping of artifact name to output path for logging.
    """
    from ollama_evaluator.models import RunReport
    from ollama_evaluator.suites.models import EvaluationSuite

    app = _build_stub_app()

    openapi_path = shared_dir / "openapi.yaml"
    suite_path = shared_dir / "evaluation-suite.schema.json"
    report_path = shared_dir / "run-report.schema.json"

    _atomic_write_text(openapi_path, _canonical_openapi_yaml(app))
    _atomic_write_text(suite_path, _canonical_json_schema(EvaluationSuite))
    _atomic_write_text(report_path, _canonical_json_schema(RunReport))

    return {
        "openapi.yaml": openapi_path,
        "evaluation-suite.schema.json": suite_path,
        "run-report.schema.json": report_path,
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    shared_dir = Path(argv[0]) if argv else _REPO_ROOT / "shared"
    written = regenerate(shared_dir)
    for name, path in written.items():
        print(f"wrote {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
