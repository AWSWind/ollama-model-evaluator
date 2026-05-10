"""Command-line interface for the Ollama Model Evaluator.

Entry point is the ``ollama-evaluator`` console script (see
``pyproject.toml`` ``project.scripts``). Built with Typer; each
subcommand mirrors a user story from Requirement 10:

* ``list-models``   — Requirement 10.1: enumerate Ollama models.
* ``run``           — Requirement 10.2-10.3: execute a Run.
* ``compare``       — Requirement 10.4: compare two Run_Reports.
* ``validate-suite``— Requirement 10.5: validate a suite file offline.
* ``serve``         — Requirement 10.6, 13.1: launch the HTTP API.
* ``convert``       — Requirement 17.8: materialise adapter suites.

Exit codes (Requirement 10.3, Property 20):

* ``0`` — Run completed and every ``TestCaseResult.status == 'pass'``.
* ``1`` — Run completed but at least one test case did not pass, or a
  catch-all error happened during ``run``/``compare``/``validate-suite``
  execution.
* ``2`` — Preflight error (``ollama_unreachable``, ``model_not_found``,
  ``dataset_fetch_failed``, ``field_map_invalid``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer

from . import __version__ as _BACKEND_VERSION
from .compare import NoCommonDimensionsError, compare
from .config import ConfigFile, RunConfig
from .history.store import HistoryStore
from .ollama.client import OllamaClient
from .runner.reports import write_artifacts
from .runner.run_state import RunEventBus, RunState
from .runner.scheduler import RunScheduler, install_signal_handlers
from .suites.loader import SuiteValidationError, discover_suites, load_suite
from .suites.models import EvaluationSuite, GenerationDefaults

# Preflight error_codes from design.md §Error Handling / §Error-taxonomy.
_PREFLIGHT_ERROR_CODES = frozenset(
    {
        "ollama_unreachable",
        "model_not_found",
        "dataset_fetch_failed",
        "field_map_invalid",
    }
)


app = typer.Typer(
    add_completion=False,
    help="Local evaluation harness for Ollama-hosted LLMs.",
    no_args_is_help=True,
)


class _GlobalOpts:
    """Mutable container for global CLI options.

    Typer does not natively support "global" options shared across
    subcommands; we emulate them by stashing the main-callback's
    parameters on a module-level instance that subcommands read.
    """

    def __init__(self) -> None:
        self.config: Path | None = None
        self.output_dir: Path | None = None
        self.log_level: str = "info"
        self.dataset_mode: str | None = None
        self.hf_cache_dir: Path | None = None


_GLOBAL = _GlobalOpts()


_LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARN,
    "error": logging.ERROR,
}


@app.callback()
def main_callback(
    config: Path | None = typer.Option(None, "--config", help="Path to ConfigFile (YAML/JSON)."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Override ConfigFile.output_dir."),
    log_level: str = typer.Option("info", "--log-level", help="debug|info|warn|error."),
    dataset_mode: str | None = typer.Option(None, "--dataset-mode", help="local|remote."),
    hf_cache_dir: Path | None = typer.Option(None, "--hf-cache-dir", help="HuggingFace cache directory."),
) -> None:
    """Parse global flags and configure logging."""
    _GLOBAL.config = config
    _GLOBAL.output_dir = output_dir
    _GLOBAL.log_level = log_level
    _GLOBAL.dataset_mode = dataset_mode
    _GLOBAL.hf_cache_dir = hf_cache_dir
    level = _LOG_LEVELS.get(log_level.lower(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_config_file() -> ConfigFile:
    """Load the user's ConfigFile or raise a typer :class:`Exit`."""
    path = _GLOBAL.config
    if path is None:
        typer.secho("--config is required for this command", fg="red", err=True)
        raise typer.Exit(code=2)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.secho(f"Cannot read config file {path}: {exc}", fg="red", err=True)
        raise typer.Exit(code=2)
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            # Use ruamel via suites.loader to avoid a second YAML import.
            from ruamel.yaml import YAML

            raw: Any = YAML(typ="safe").load(text)
        else:
            raw = json.loads(text)
        config = ConfigFile.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — config parsing is broad by design.
        typer.secho(f"Invalid config file: {exc}", fg="red", err=True)
        raise typer.Exit(code=2)
    # Apply global overrides.
    if _GLOBAL.output_dir is not None:
        config = config.model_copy(update={"output_dir": _GLOBAL.output_dir})
    if _GLOBAL.dataset_mode is not None:
        config = config.model_copy(update={"dataset_mode": _GLOBAL.dataset_mode})
    if _GLOBAL.hf_cache_dir is not None:
        config = config.model_copy(update={"hf_cache_dir": _GLOBAL.hf_cache_dir})
    return config


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------


@app.command("list-models")
def list_models_cmd(
    ollama_base_url: str = typer.Option(
        "http://localhost:11434",
        "--ollama-base-url",
        help="Base URL of the Ollama server.",
    ),
) -> None:
    """List models available on the local Ollama server."""

    async def _run() -> int:
        client = OllamaClient(base_url=ollama_base_url)
        try:
            models = await client.list_models()
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"ollama_unreachable: {exc}", fg="red", err=True)
            return 2
        finally:
            await client.aclose()
        for m in models:
            typer.echo(f"{m.name}\t{m.digest or 'n/a'}\t{m.parameter_size or 'n/a'}")
        return 0

    raise typer.Exit(code=asyncio.run(_run()))


# ---------------------------------------------------------------------------
# validate-suite
# ---------------------------------------------------------------------------


@app.command("validate-suite")
def validate_suite_cmd(
    path: Path = typer.Argument(..., help="Path to an Evaluation_Suite YAML/JSON file."),
) -> None:
    """Validate a single Evaluation_Suite file offline."""
    try:
        suite = load_suite(path)
    except SuiteValidationError as exc:
        typer.secho(f"suite_invalid: {exc.message}", fg="red", err=True)
        if exc.missing_field:
            typer.secho(f"  field: {exc.missing_field}", fg="red", err=True)
        if exc.line:
            typer.secho(f"  line: {exc.line}", fg="red", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {suite.name} ({len(suite.test_cases)} test cases)")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd() -> None:
    """Execute a Run using the supplied ConfigFile.

    Exit codes:

    * ``0`` — every :class:`TestCaseResult.status == 'pass'``.
    * ``1`` — at least one test case did not pass.
    * ``2`` — preflight error.
    """
    config = _load_config_file()
    exit_code = asyncio.run(_execute_run(config))
    raise typer.Exit(code=exit_code)


async def _execute_run(config: ConfigFile) -> int:
    """Run the scheduler and return an exit code."""
    # Discover suites.
    try:
        all_suites = discover_suites(config.suites_dir)
    except SuiteValidationError as exc:
        typer.secho(f"suite_invalid: {exc.message}", fg="red", err=True)
        return 2

    by_name = {s.name: s for s in all_suites}
    selected_suites: list[EvaluationSuite] = [
        by_name[n] for n in config.run.suites if n in by_name
    ]
    if len(selected_suites) != len(config.run.suites):
        missing = [n for n in config.run.suites if n not in by_name]
        typer.secho(
            f"suite_not_found: {', '.join(missing)}", fg="red", err=True
        )
        return 2

    defaults = selected_suites[0].defaults if selected_suites else GenerationDefaults()

    # Set up the store so the Run is persisted (so `compare` can later see it).
    db_path = config.output_dir / "history.db"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    async with HistoryStore.open(db_path, config.output_dir) as store:
        run_id = await store.create_run(config)
        state = RunState(run_id=run_id, status="pending")
        bus = RunEventBus(state)

        restore = install_signal_handlers(state)
        client = OllamaClient(
            base_url=config.ollama_base_url,
            timeout_s=config.run.ollama_timeout_s,
        )
        try:
            scheduler = RunScheduler(
                run_state=state,
                bus=bus,
                ollama_client=client,
                run_config=config.run,
                suites=selected_suites,
                generation_defaults=defaults,
                config_file=config,
                store=store,
            )
            report = await scheduler.execute()
        finally:
            try:
                restore()
            except Exception:  # noqa: BLE001
                pass
            await client.aclose()

        # Write Markdown + JSON artefacts. The store.write_report call
        # already wrote report.json; write_artifacts additionally emits
        # report.md.
        await write_artifacts(run_id, report, config.output_dir)

    # Decide exit code.
    for event in state.events:
        if event.type == "run-failed":
            code = getattr(event, "error_code", "")
            if code in _PREFLIGHT_ERROR_CODES:
                typer.secho(f"{code}: {event.message}", fg="red", err=True)
                return 2
            typer.secho(f"run failed: {event.message}", fg="red", err=True)
            return 1

    # Success path: exit 0 iff every TestCaseResult.status == 'pass'.
    results = state.events  # type: ignore[assignment]
    del results
    all_passed = all(r.status == "pass" for r in report.results)
    summary = (
        f"run {run_id}: {len(report.results)} executions, "
        f"passed={sum(1 for r in report.results if r.status == 'pass')}, "
        f"failed={sum(1 for r in report.results if r.status == 'fail')}, "
        f"error={sum(1 for r in report.results if r.status == 'error')}, "
        f"timeout={sum(1 for r in report.results if r.status == 'timeout')}"
    )
    typer.echo(summary)
    return 0 if all_passed and report.results else 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
def compare_cmd(
    run_a: str = typer.Argument(..., metavar="RUN_A", help="Run id of the base run."),
    run_b: str = typer.Argument(..., metavar="RUN_B", help="Run id of the comparison run."),
) -> None:
    """Compare two persisted Run_Reports."""
    config = _load_config_file()

    async def _run() -> int:
        db_path = config.output_dir / "history.db"
        async with HistoryStore.open(db_path, config.output_dir) as store:
            report_a = await store.get_run(run_a)
            report_b = await store.get_run(run_b)
            if report_a is None:
                typer.secho(f"run_not_found: {run_a}", fg="red", err=True)
                return 1
            if report_b is None:
                typer.secho(f"run_not_found: {run_b}", fg="red", err=True)
                return 1
            try:
                result = compare(report_a, report_b)
            except NoCommonDimensionsError as exc:
                typer.secho(f"no_common_dimensions: {exc}", fg="red", err=True)
                return 1
            typer.echo(result.model_dump_json(indent=2))
            return 0

    raise typer.Exit(code=asyncio.run(_run()))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
) -> None:
    """Start the FastAPI + Uvicorn server until SIGINT/SIGTERM.

    Uses :class:`~ollama_evaluator.api.app.AppDeps`' lazy wiring so
    that the :class:`HistoryStore` (and its underlying ``aiosqlite``
    connection) is opened on uvicorn's own event loop rather than on
    a short-lived loop spun up for bootstrap. Before this rework the
    store was opened via ``asyncio.run(...)`` before the server
    started, which bound the connection to a loop that no longer
    existed by the time the first request arrived — requests hitting
    store endpoints failed with ``ValueError: no active connection``
    from inside ``aiosqlite``.
    """
    import uvicorn

    from .api.app import AppDeps, create_app
    from .api.supervisor import RunSupervisor

    config = _load_config_file()

    db_path = config.output_dir / "history.db"
    output_dir = config.output_dir
    suites_dir = config.suites_dir
    # Ensure the output dir exists before uvicorn spins up so the
    # first ``create_run`` does not have to race a missing directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _store_factory() -> tuple[Any, Any]:
        """Open the ``HistoryStore`` on the serving loop.

        Returns the opened store plus a cleanup closure that
        releases the :class:`HistoryStore.open` async-context-manager
        cleanly at shutdown.
        """
        cm = HistoryStore.open(db_path, output_dir)
        store = await cm.__aenter__()

        async def _cleanup() -> None:
            # ``__aexit__`` is idempotent — ``HistoryStore.open``
            # closes its aiosqlite connection inside the ``finally``
            # branch of its own async generator body regardless of
            # the exception triple we pass.
            await cm.__aexit__(None, None, None)

        return store, _cleanup

    def _supervisor_factory(store: Any) -> RunSupervisor:
        """Build the supervisor bound to the store opened above."""
        return RunSupervisor(
            store,
            suites_dir=suites_dir,
            output_dir=output_dir,
            default_ollama_base_url=config.ollama_base_url,
        )

    deps = AppDeps(
        suites_dir=suites_dir,
        output_dir=output_dir,
        ollama_base_url=config.ollama_base_url,
        ollama_timeout_s=config.run.ollama_timeout_s,
        store_factory=_store_factory,
        supervisor_factory=_supervisor_factory,
    )
    app_obj = create_app(deps)
    uvicorn.run(app_obj, host=host, port=port, log_level=_GLOBAL.log_level)


# ---------------------------------------------------------------------------
# convert (Task 19.2)
# ---------------------------------------------------------------------------


convert_app = typer.Typer(help="Materialise public-benchmark datasets into Evaluation_Suite YAML.")
app.add_typer(convert_app, name="convert")


def _run_convert(
    adapter_name: str,
    output: Path,
    source: Path | None,
    limit: int | None,
    seed: int | None,
    **extra: Any,
) -> None:
    """Shared implementation for every ``convert <adapter>`` subcommand."""
    from .suites.adapter_base import AdapterOptions
    from .suites.adapters import get_adapter
    from .suites.writer import dump_suite

    adapter = get_adapter(adapter_name)
    opts = AdapterOptions(limit=limit, seed=seed, **extra)

    try:
        suite = adapter.materialise(
            mode="local",
            opts=opts,
            cache_dir=source,
        )
    except Exception as exc:  # noqa: BLE001 — user-facing error boundary.
        typer.secho(f"dataset_fetch_failed: {exc}", fg="red", err=True)
        raise typer.Exit(code=2)

    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{suite.name}.yaml"
    output_path.write_text(dump_suite(suite, "yaml"), encoding="utf-8")
    typer.echo(f"wrote {output_path} ({len(suite.test_cases)} test cases)")


@convert_app.command("mmlu")
def convert_mmlu(
    source: Path | None = typer.Option(None, "--source", help="Local adapter cache dir."),
    output: Path = typer.Option(..., "--output", help="Directory to write YAML suite."),
    subjects: str | None = typer.Option(None, "--subjects", help="Optional MMLU subject filter."),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Materialise MMLU suites from a local cache of source rows."""
    _run_convert(
        "mmlu",
        output=output,
        source=source,
        limit=limit,
        seed=seed,
        subject=subjects,
    )


@convert_app.command("hellaswag")
def convert_hellaswag(
    source: Path | None = typer.Option(None, "--source", help="Local adapter cache dir."),
    output: Path = typer.Option(..., "--output"),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Materialise HellaSwag."""
    _run_convert("hellaswag", output=output, source=source, limit=limit, seed=seed)


@convert_app.command("truthfulqa")
def convert_truthfulqa(
    source: Path | None = typer.Option(None, "--source", help="Local adapter cache dir."),
    output: Path = typer.Option(..., "--output"),
    form: str = typer.Option("mc1", "--form", help="mc1|mc2 (only mc1 in v1)."),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Materialise TruthfulQA."""
    _run_convert("truthfulqa", output=output, source=source, limit=limit, seed=seed, form=form)


@convert_app.command("gsm8k")
def convert_gsm8k(
    source: Path | None = typer.Option(None, "--source", help="Local adapter cache dir."),
    output: Path = typer.Option(..., "--output"),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Materialise GSM8K."""
    _run_convert("gsm8k", output=output, source=source, limit=limit, seed=seed)


@convert_app.command("humaneval")
def convert_humaneval(
    source: Path | None = typer.Option(None, "--source", help="Local adapter cache dir."),
    output: Path = typer.Option(..., "--output"),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Materialise HumanEval (response-capture metric only in v1)."""
    _run_convert("humaneval", output=output, source=source, limit=limit, seed=seed)


@convert_app.command("hf")
def convert_hf(
    hf_ref: str = typer.Option(..., "--hf-ref", help="repo_id[:config][:split]"),
    field_map: Path = typer.Option(..., "--field-map", help="Path to HFFieldMap YAML/JSON."),
    output: Path = typer.Option(..., "--output"),
    name: str = typer.Option("custom-hf", "--name", help="Suite name."),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Materialise a custom HuggingFace dataset using an explicit field map."""
    import json as _json

    from ruamel.yaml import YAML

    from .suites.adapter_base import HFRef
    from .suites.huggingface import HFFieldMap, HFSuiteSpec, materialise_hf, stream_rows
    from .suites.models import MetricConfig
    from .suites.writer import dump_suite

    # Parse the field map file (YAML or JSON).
    text = field_map.read_text(encoding="utf-8")
    if field_map.suffix.lower() in (".yaml", ".yml"):
        raw = YAML(typ="safe").load(text)
    else:
        raw = _json.loads(text)
    try:
        fm = HFFieldMap.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"field_map_invalid: {exc}", fg="red", err=True)
        raise typer.Exit(code=2)

    spec = HFSuiteSpec(
        kind="huggingface",
        name=name,
        hf_ref=HFRef.parse(hf_ref),
        field_map=fm,
        limit=limit,
        seed=seed,
        metrics=[MetricConfig(name="exact-match")],
    )
    try:
        rows = list(stream_rows(spec.hf_ref, mode="remote", cache_dir=_GLOBAL.hf_cache_dir))
        suite = materialise_hf(spec, rows)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"dataset_fetch_failed: {exc}", fg="red", err=True)
        raise typer.Exit(code=2)

    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{suite.name}.yaml"
    output_path.write_text(dump_suite(suite, "yaml"), encoding="utf-8")
    typer.echo(f"wrote {output_path} ({len(suite.test_cases)} test cases)")


__all__ = ["app"]


def main() -> None:  # pragma: no cover - thin alias.
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
