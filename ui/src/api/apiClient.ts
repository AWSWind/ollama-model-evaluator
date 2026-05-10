/**
 * Thin typed wrapper around ``fetch`` for the Backend REST API.
 *
 * Endpoints and request/response shapes come from the generated
 * {@link ./types.ts} module, which is produced by ``openapi-typescript``
 * from ``shared/openapi.yaml``. This file keeps the binding between
 * those generated types and the UI surface area: one function per REST
 * endpoint, each returning the concrete schema referenced by that
 * endpoint's response.
 *
 * Base URL resolution
 * -------------------
 * In the browser the UI is always served by the backend that also owns
 * the REST API (``cli serve`` mounts ``ui/dist/`` at ``/``), so we want
 * every fetch to go to the *same origin the page came from*. Hardcoding
 * ``http://localhost:8765`` used to break the moment a user opened the
 * UI against a non-local host (for example ``http://192.168.1.224:8765``
 * over the LAN) — the browser would make cross-origin requests back to
 * ``localhost`` on the viewer's own machine, which of course has no
 * backend running.
 *
 * To avoid that, {@link getBaseUrl} returns an empty string when we
 * detect a browser environment (``typeof window !== 'undefined'``).
 * ``fetch("" + "/api/health")`` is then a same-origin request and the
 * browser automatically points it at whatever host:port served the
 * page.
 *
 * For non-browser contexts (Vitest under jsdom, Node-based tests, the
 * generated client used from scripts) we retain an explicit default of
 * ``http://localhost:8765`` so existing tests keep working, and
 * {@link setBaseUrl} can still override the value for tests that run
 * against an ephemeral port.
 */

import type { components } from "./types";
import type { ErrorCode, ErrorEnvelope } from "./errorTypes";

export type RunConfig = components["schemas"]["RunConfig"];
export type RunReport = components["schemas"]["RunReport"];
export type ModelInfo = components["schemas"]["ModelInfo"];
export type EvaluationSuite = components["schemas"]["EvaluationSuite"];
export type SuiteSummary = components["schemas"]["SuiteSummary"];
export type ComparisonReport = components["schemas"]["ComparisonReport"];
export type SubmitRunResponse = components["schemas"]["SubmitRunResponse"];
export type CancelRunResponse = components["schemas"]["CancelRunResponse"];
export type { ErrorEnvelope, ErrorCode };

/**
 * Filter supported by ``GET /api/runs``.
 */
export interface RunListFilter {
  model?: string;
  suite?: string;
  status?: string;
  since?: string;
  until?: string;
}

/**
 * Sentinel meaning "use relative URLs so the browser targets the same
 * origin that served the page". Only meaningful at runtime inside a
 * browser; Node/Vitest code paths receive a concrete URL instead.
 */
const SAME_ORIGIN = "";

/**
 * Default base URL used when nothing else has overridden it.
 *
 * * In a browser (``window`` defined): the empty string so every
 *   fetch becomes a same-origin relative call.
 * * Outside a browser: the historic ``http://localhost:8765`` default
 *   so Node-level tests can still target a local dev server.
 */
const DEFAULT_BASE_URL =
  typeof window !== "undefined" ? SAME_ORIGIN : "http://localhost:8765";

let baseUrl: string = DEFAULT_BASE_URL;

/**
 * Override the base URL for every subsequent request.
 *
 * Two important callers:
 *
 * 1. Tests that spin up a FastAPI ``TestClient`` on an ephemeral port
 *    and need the client to target that URL.
 * 2. Non-browser scripts (``openapi-typescript``-generated clients,
 *    one-off node tools) that want to drive a specific backend.
 */
export function setBaseUrl(url: string): void {
  baseUrl = url.replace(/\/$/, "");
}

/**
 * Return the current base URL.
 *
 * The empty string means "same-origin" when we are in a browser —
 * ``getBaseUrl() + "/api/health"`` still produces a valid fetch target
 * (``/api/health``), and the browser resolves it against
 * ``window.location.origin``.
 *
 * Callers that must construct a full URL (for example the WebSocket
 * client that has to replace ``http://`` with ``ws://``) can use
 * {@link getOriginUrl} below, which returns ``window.location.origin``
 * when the sentinel is active.
 */
export function getBaseUrl(): string {
  return baseUrl;
}

/**
 * Return a concrete ``scheme://host:port`` origin suitable for building
 * WebSocket URLs or for logging.
 *
 * When {@link getBaseUrl} returns the same-origin sentinel (empty
 * string) this falls back to ``window.location.origin`` so callers can
 * still produce an absolute URL. Outside the browser the sentinel is
 * never active, so this simply returns ``baseUrl``.
 */
export function getOriginUrl(): string {
  if (baseUrl !== SAME_ORIGIN) {
    return baseUrl;
  }
  if (typeof window !== "undefined" && window.location) {
    return window.location.origin;
  }
  // Fallback for exotic environments where neither ``setBaseUrl`` nor a
  // browser ``window`` is available. Matches the historical default so
  // we do not silently break a consumer that was relying on it.
  return "http://localhost:8765";
}

/**
 * Thrown for any non-2xx response. Carries the full decoded
 * :class:`ErrorEnvelope` body when the Backend emitted one, plus the
 * HTTP status code. Callers can discriminate on ``envelope.error_code``
 * (e.g. ``"validation_failed"``, ``"no_common_dimensions"``) to render
 * specific messages.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly envelope: ErrorEnvelope | null;

  constructor(status: number, message: string, envelope: ErrorEnvelope | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.envelope = envelope;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    headers: {
      "content-type": "application/json",
      accept: "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    let envelope: ErrorEnvelope | null = null;
    let message = `HTTP ${response.status}`;
    try {
      const text = await response.text();
      if (text) {
        const body = JSON.parse(text) as ErrorEnvelope;
        envelope = body;
        message = body.message ?? message;
      }
    } catch {
      // Non-JSON body; fall through with the default message.
    }
    throw new ApiError(response.status, message, envelope);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return (await response.text()) as unknown as T;
}

export async function listModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/api/models");
}

export async function listSuites(): Promise<string[]> {
  return request<string[]>("/api/suites");
}

/**
 * Fetch lightweight per-suite metadata in a single round-trip.
 *
 * Used by the New Run route to annotate each suite option with its
 * case count (and thus an ETA) without downloading every Test_Case.
 * Equivalent to ``GET /api/suites/summaries``.
 */
export async function listSuiteSummaries(): Promise<SuiteSummary[]> {
  return request<SuiteSummary[]>("/api/suites/summaries");
}

export async function getSuite(name: string): Promise<EvaluationSuite> {
  return request<EvaluationSuite>(`/api/suites/${encodeURIComponent(name)}`);
}

export async function submitRun(config: RunConfig): Promise<SubmitRunResponse> {
  return request<SubmitRunResponse>("/api/runs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

function buildQueryString(params: Record<string, string | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") {
      search.append(key, value);
    }
  }
  const str = search.toString();
  return str ? `?${str}` : "";
}

export async function listRuns(filter: RunListFilter = {}): Promise<RunReport[]> {
  const query = buildQueryString({
    model: filter.model,
    suite: filter.suite,
    status: filter.status,
    since: filter.since,
    until: filter.until,
  });
  return request<RunReport[]>(`/api/runs${query}`);
}

export async function getRun(id: string): Promise<RunReport> {
  return request<RunReport>(`/api/runs/${encodeURIComponent(id)}`);
}

export async function getRunMarkdown(id: string): Promise<string> {
  const response = await fetch(
    `${baseUrl}/api/runs/${encodeURIComponent(id)}/report.md`,
    { headers: { accept: "text/markdown" } },
  );
  if (!response.ok) {
    throw new ApiError(response.status, `HTTP ${response.status}`, null);
  }
  return response.text();
}

export async function deleteRun(id: string): Promise<void> {
  await request<void>(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function cancelRun(id: string): Promise<CancelRunResponse> {
  return request<CancelRunResponse>(
    `/api/runs/${encodeURIComponent(id)}/cancel`,
    { method: "POST" },
  );
}

export async function compareRuns(a: string, b: string): Promise<ComparisonReport> {
  const query = buildQueryString({ a, b });
  return request<ComparisonReport>(`/api/compare${query}`);
}
