"""REST endpoints for the Ollama Model Evaluator Backend.

All endpoints return JSON except ``GET /api/runs/{id}/report.md``
which returns ``text/markdown``. Non-2xx responses always carry an
:class:`~ollama_evaluator.api.errors.ErrorEnvelope` body produced by
:func:`ollama_evaluator.api.errors.http_error` or the global
validation handler in :mod:`ollama_evaluator.api.app`.

Endpoints:

* ``GET  /api/health``                  — liveness probe.
* ``GET  /api/models``                  — list models known to Ollama.
* ``GET  /api/suites``                  — list discovered suite names.
* ``GET  /api/suites/summaries``        — list [{name, test_case_count, description}].
* ``GET  /api/suites/{name}``           — full :class:`EvaluationSuite`.
* ``POST /api/runs``                    — submit a :class:`RunConfig`.
* ``GET  /api/runs``                    — list run reports matching filters.
* ``GET  /api/runs/{id}``               — single run report.
* ``GET  /api/runs/{id}/report.md``     — markdown rendering.
* ``DELETE /api/runs/{id}``             — remove a run.
* ``POST /api/runs/{id}/cancel``        — cooperative cancel.
* ``GET  /api/compare``                 — comparison between two runs.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§REST API. Requirements 2.1, 9.1, 12.4, 12.5, 13.1-13.6, 16.1, 16.3,
16.5.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from ..compare import ComparisonReport, NoCommonDimensionsError, compare
from ..config import RunConfig
from ..history.store import RunListFilter
from ..models import RunReport
from ..runner.reports import render_markdown
from ..suites.loader import SuiteValidationError, discover_suites
from ..suites.models import EvaluationSuite
from .errors import ErrorCode, http_error

log = logging.getLogger(__name__)


class SubmitRunResponse(BaseModel):
    """Body of the 201 response from ``POST /api/runs`` (Requirement 13.3)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Identifier of the newly queued Run.")
    status: str = Field(..., description="Initial status; always ``pending``.")


class SuiteSummary(BaseModel):
    """Lightweight metadata for a single discovered Evaluation_Suite.

    Surfaced by ``GET /api/suites/summaries`` so clients (notably the
    UI's New Run page) can render per-suite case counts and a time
    estimate without issuing N full-suite fetches. Returning only the
    counts (not the full ``test_cases`` list) is a ~100× payload
    reduction for suites like ``mmlu`` (180 KB → ~100 bytes).

    The field set is intentionally minimal; callers that need any
    additional metadata can still issue the targeted
    ``GET /api/suites/{name}`` fetch.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique suite name (same identifier as ``GET /api/suites``).")
    test_case_count: int = Field(
        ...,
        ge=0,
        description="Number of Test_Cases contained in the suite.",
    )
    description: str | None = Field(
        default=None,
        description="Optional free-form description copied from the suite file.",
    )


class CancelRunResponse(BaseModel):
    """Body of the 200 response from ``POST /api/runs/{id}/cancel``."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str = Field(..., description="The new status after the cancel flag is set.")


def _req_state(request: Request) -> "AppDeps":
    """Return the :class:`AppDeps` bag attached to the app state.

    Imported locally to avoid a circular import at module load time:
    :mod:`api.app` imports this module (for the router) before it
    defines :class:`AppDeps`.
    """
    from .app import AppDeps  # noqa: F401

    return request.app.state.deps


def build_router() -> APIRouter:
    """Return an :class:`APIRouter` wired with every REST endpoint."""

    router = APIRouter()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @router.get("/api/health")
    async def health() -> dict[str, str]:
        """Liveness probe used by the UI to detect Backend availability."""
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    @router.get("/api/models")
    async def list_models(request: Request) -> list[dict]:
        """List models reported by the configured Ollama_Server.

        Translates network/HTTP failures to ``502 ollama_unreachable``
        so UI clients can branch on a well-known code.
        """
        deps = _req_state(request)
        client = deps.ollama_client_factory(deps.ollama_base_url, deps.ollama_timeout_s)
        try:
            models = await client.list_models()
        except Exception as exc:  # noqa: BLE001 — per-request isolation.
            log.warning("ollama list_models failed: %s", exc)
            raise http_error(
                ErrorCode.ollama_unreachable,
                f"Ollama server unreachable: {exc}",
                status_code=502,
            )
        finally:
            if hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001 — best-effort.
                    pass
        return [m.model_dump(mode="json") for m in models]

    # ------------------------------------------------------------------
    # Suites
    # ------------------------------------------------------------------

    @router.get("/api/suites")
    async def list_suites(request: Request) -> list[str]:
        """Return sorted suite names discovered under ``suites_dir``."""
        deps = _req_state(request)
        try:
            suites = discover_suites(deps.suites_dir)
        except SuiteValidationError as exc:
            raise http_error(
                ErrorCode.suite_invalid,
                exc.message,
                status_code=400,
                field=exc.missing_field,
            )
        return [s.name for s in suites]

    # NOTE: Registered **before** ``GET /api/suites/{name}`` so FastAPI's
    # router matches the literal ``summaries`` segment rather than treating
    # it as the ``{name}`` path parameter. The two endpoints are siblings,
    # not a sub-resource — this endpoint exists solely to let clients
    # compute per-suite case counts cheaply.
    @router.get(
        "/api/suites/summaries",
        response_model=list[SuiteSummary],
    )
    async def list_suite_summaries(request: Request) -> list[SuiteSummary]:
        """Return lightweight metadata (name, case count, description) for every suite.

        Used by the UI's New Run page to annotate the suites multi-select
        with case counts and rough-ETA estimates without pulling every
        Test_Case. Results are emitted in the same order as
        ``GET /api/suites``.
        """
        deps = _req_state(request)
        try:
            suites = discover_suites(deps.suites_dir)
        except SuiteValidationError as exc:
            raise http_error(
                ErrorCode.suite_invalid,
                exc.message,
                status_code=400,
                field=exc.missing_field,
            )
        return [
            SuiteSummary(
                name=s.name,
                test_case_count=len(s.test_cases),
                description=s.description,
            )
            for s in suites
        ]

    @router.get("/api/suites/{name}", response_model=EvaluationSuite)
    async def get_suite(name: str, request: Request) -> EvaluationSuite:
        """Return the full :class:`EvaluationSuite` matching ``name``."""
        deps = _req_state(request)
        try:
            suites = discover_suites(deps.suites_dir)
        except SuiteValidationError as exc:
            raise http_error(
                ErrorCode.suite_invalid,
                exc.message,
                status_code=400,
                field=exc.missing_field,
            )
        for s in suites:
            if s.name == name:
                return s
        raise http_error(
            ErrorCode.suite_not_found,
            f"No suite named {name!r}",
            status_code=404,
        )

    # ------------------------------------------------------------------
    # Runs — submit, list, get, cancel, delete, markdown report
    # ------------------------------------------------------------------

    @router.post(
        "/api/runs",
        status_code=status.HTTP_201_CREATED,
        response_model=SubmitRunResponse,
    )
    async def submit_run(config: RunConfig, request: Request) -> SubmitRunResponse:
        """Persist a new Run as ``pending`` and hand it to the supervisor."""
        deps = _req_state(request)
        run_id = await deps.supervisor.submit(config)
        return SubmitRunResponse(run_id=run_id, status="pending")

    @router.get("/api/runs")
    async def list_runs(
        request: Request,
        model: str | None = Query(default=None),
        suite: str | None = Query(default=None),
        run_status: str | None = Query(default=None, alias="status"),
        since: datetime | None = Query(default=None),
        until: datetime | None = Query(default=None),
    ) -> list[dict]:
        """Return stored runs matching the supplied filters."""
        deps = _req_state(request)
        filter = RunListFilter(
            model=model,
            suite=suite,
            status=run_status,
            since=since,
            until=until,
        )
        reports = await deps.store.list_runs(filter)
        return [r.model_dump(mode="json") for r in reports]

    @router.get("/api/runs/{run_id}", response_model=RunReport)
    async def get_run(run_id: str, request: Request) -> RunReport:
        """Return the full :class:`RunReport` for ``run_id``."""
        deps = _req_state(request)
        report = await deps.store.get_run(run_id)
        if report is None:
            raise http_error(
                ErrorCode.run_not_found,
                f"No run with id {run_id!r}",
                status_code=404,
            )
        return report

    @router.get("/api/runs/{run_id}/report.md")
    async def get_run_markdown(run_id: str, request: Request) -> Response:
        """Return the Run_Report rendered as ``text/markdown``."""
        deps = _req_state(request)
        report = await deps.store.get_run(run_id)
        if report is None:
            raise http_error(
                ErrorCode.run_not_found,
                f"No run with id {run_id!r}",
                status_code=404,
            )
        text = render_markdown(report)
        return Response(content=text, media_type="text/markdown; charset=utf-8")

    @router.delete(
        "/api/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT
    )
    async def delete_run(run_id: str, request: Request) -> Response:
        """Delete a Run and its persisted artifacts (Requirement 12.5)."""
        deps = _req_state(request)
        existing = await deps.store.get_run(run_id)
        if existing is None:
            raise http_error(
                ErrorCode.run_not_found,
                f"No run with id {run_id!r}",
                status_code=404,
            )
        await deps.store.delete_run(run_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post(
        "/api/runs/{run_id}/cancel",
        response_model=CancelRunResponse,
    )
    async def cancel_run(run_id: str, request: Request) -> CancelRunResponse:
        """Request cooperative cancellation of a live Run."""
        deps = _req_state(request)
        flipped = deps.supervisor.cancel(run_id)
        if not flipped:
            # No live state — check whether the Run ever existed.
            report = await deps.store.get_run(run_id)
            if report is None:
                raise http_error(
                    ErrorCode.run_not_found,
                    f"No run with id {run_id!r}",
                    status_code=404,
                )
            # Run already reached a terminal state; echo it back.
            return CancelRunResponse(run_id=run_id, status=report.status)
        state = deps.supervisor.get_state(run_id)
        new_status = state.status if state is not None else "aborted"
        return CancelRunResponse(run_id=run_id, status=new_status)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    @router.get("/api/compare", response_model=ComparisonReport)
    async def compare_runs(
        request: Request,
        a: str = Query(..., description="Run id of the base run."),
        b: str = Query(..., description="Run id of the comparison run."),
    ) -> ComparisonReport:
        """Return the comparison between two persisted Run_Reports."""
        deps = _req_state(request)
        report_a = await deps.store.get_run(a)
        if report_a is None:
            raise http_error(
                ErrorCode.run_not_found,
                f"No run with id {a!r}",
                status_code=404,
            )
        report_b = await deps.store.get_run(b)
        if report_b is None:
            raise http_error(
                ErrorCode.run_not_found,
                f"No run with id {b!r}",
                status_code=404,
            )
        try:
            return compare(report_a, report_b)
        except NoCommonDimensionsError as exc:
            raise http_error(
                ErrorCode.no_common_dimensions,
                str(exc),
                status_code=400,
            )

    return router


__all__ = [
    "CancelRunResponse",
    "SubmitRunResponse",
    "build_router",
]
