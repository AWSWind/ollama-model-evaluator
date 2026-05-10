/**
 * Pure reducer that folds Run_Events into UI-renderable state.
 *
 * The reducer is the single source of truth for the Run-detail view's
 * progress panel, counters, and per-execution table. It is deliberately
 * kept pure (no I/O, no timers) so that:
 *
 * * the React component simply picks ``useReducer(runEventReducer,
 *   initialRunState)`` and routes incoming frames to ``dispatch``;
 * * replay under test is trivial — fold left twice and compare;
 * * Property 37 (Task 23.4) can drive arbitrary event sequences through
 *   the reducer and assert replay determinism without any React setup.
 *
 * Requirements traced:
 *
 * * 15.6 — consume every event arriving on the WebSocket.
 * * 15.7 — display status, percent-complete, counters and a live table.
 * * 16.3 — derived state for the completed-run view.
 * * 16.6 — terminal event error message is retained for display.
 */

import type { components } from "../api/types";

type TestCaseResult = components["schemas"]["TestCaseResult"];
type MetricResult = components["schemas"]["MetricResult"];
type RunSummary = {
  planned_executions: number;
  completed_executions: number;
  passed: number;
  failed: number;
  errored: number;
  timed_out: number;
};

/**
 * Closed discriminated union of Run_Event variants the UI understands.
 *
 * We mirror the Pydantic discriminator in
 * ``backend/src/ollama_evaluator/events.py`` locally because the
 * generated ``components["schemas"]`` surface does not include the
 * Run_Event types (they traverse the WebSocket, not the REST contract
 * OpenAPI describes). Keeping the union here lets the reducer enjoy
 * exhaustive ``switch`` coverage in TypeScript.
 */
export type RunEventUnion =
  | {
      type: "run-started";
      run_id: string;
      seq: number;
      ts: string;
      planned_executions: number;
    }
  | {
      type: "run-progress";
      run_id: string;
      seq: number;
      ts: string;
      completed: number;
      in_progress: number;
      pending: number;
    }
  | {
      type: "test-case-completed";
      run_id: string;
      seq: number;
      ts: string;
      result: TestCaseResult;
    }
  | {
      type: "run-completed";
      run_id: string;
      seq: number;
      ts: string;
      summary: RunSummary;
    }
  | {
      type: "run-aborted";
      run_id: string;
      seq: number;
      ts: string;
      reason: string;
    }
  | {
      type: "run-failed";
      run_id: string;
      seq: number;
      ts: string;
      error_code: string;
      message: string;
    };

/**
 * One row of the live-updating execution table displayed under the
 * progress panel (Requirement 15.7).
 *
 * The shape is a flat projection of :class:`TestCaseResult` because the
 * table renders one cell per field; nesting ``performance`` inside the
 * row would just force the React code to flatten again at render time.
 */
export interface RunEventRow {
  model: string;
  suite: string;
  test_case_id: string;
  repetition: number;
  status: string;
  metrics: Array<{ name: string; score: number; passed: boolean }>;
  ttft_ms: number | null;
  total_ms: number;
  tokens_per_second: number | null;
}

/**
 * Folded UI state driven by :func:`runEventReducer`.
 *
 * Every field has a sensible "no data yet" value so the initial render,
 * before any event arrives, displays a blank dashboard without
 * conditional rendering at every read site.
 */
export interface RunEventState {
  /**
   * Observed Run status. ``null`` until the ``run-started`` event
   * arrives so the component can distinguish "still connecting" from
   * "running".
   */
  status:
    | "pending"
    | "running"
    | "completed"
    | "aborted"
    | "failed"
    | null;
  /**
   * Total executions planned for the Run, taken from the
   * ``run-started`` event. ``null`` when not yet known; also drives the
   * progress-bar percentage computation.
   */
  planned_executions: number | null;
  /**
   * Per-status counters derived from :class:`TestCaseResult.status`
   * values carried on ``test-case-completed`` events. Replaced wholesale
   * by the ``run-completed`` summary when that event arrives so the
   * final rendering matches the report's ground truth.
   */
  counters: {
    passed: number;
    failed: number;
    error: number;
    timeout: number;
  };
  /**
   * Per-model counters, built incrementally from
   * ``test-case-completed`` events. Lets the Run-detail view show each
   * model's pass/fail rate live rather than only at terminal time —
   * important for multi-model runs where aggregate counters mask
   * individual model behaviour.
   *
   * The map is keyed by :class:`TestCaseResult.model` and each bucket
   * has the same shape as ``counters``. Not replaced by the
   * ``run-completed`` summary (the summary is aggregate-only) — so the
   * incremental view remains available on the terminal state too.
   */
  per_model: Record<
    string,
    { passed: number; failed: number; error: number; timeout: number }
  >;
  completed: number;
  in_progress: number;
  pending: number;
  /**
   * ``(completed / planned) * 100`` when both known; ``0`` otherwise.
   */
  percent_complete: number;
  /** One row per ``test-case-completed`` event in arrival order. */
  rows: RunEventRow[];
  /**
   * Error message attached to ``run-aborted``/``run-failed``. ``null``
   * for all non-terminal and happy-path terminal states.
   */
  terminal_error: string | null;
}

export const initialRunState: RunEventState = {
  status: null,
  planned_executions: null,
  counters: { passed: 0, failed: 0, error: 0, timeout: 0 },
  per_model: {},
  completed: 0,
  in_progress: 0,
  pending: 0,
  percent_complete: 0,
  rows: [],
  terminal_error: null,
};

function percent(completed: number, planned: number | null): number {
  if (planned == null || planned <= 0) {
    return 0;
  }
  return (completed / planned) * 100;
}

function counterKeyFor(
  status: TestCaseResult["status"],
): keyof RunEventState["counters"] {
  switch (status) {
    case "pass":
      return "passed";
    case "fail":
      return "failed";
    case "error":
      return "error";
    case "timeout":
      return "timeout";
  }
}

function rowFromResult(result: TestCaseResult): RunEventRow {
  const metrics = result.metrics.map((m: MetricResult) => ({
    name: m.name,
    score: m.score,
    passed: m.passed,
  }));
  return {
    model: result.model,
    suite: result.suite,
    test_case_id: result.test_case_id,
    repetition: result.repetition,
    status: result.status,
    metrics,
    ttft_ms: result.performance.ttft_ms ?? null,
    total_ms: result.performance.total_ms,
    tokens_per_second: result.performance.tokens_per_second ?? null,
  };
}

/**
 * Pure fold of a :class:`RunEventUnion` into :class:`RunEventState`.
 *
 * Every branch returns a fresh object (no mutation) so the reducer is
 * safe to use with ``useReducer`` without accidental reference-equality
 * bailouts in downstream memoisation.
 */
export function runEventReducer(
  state: RunEventState,
  event: RunEventUnion,
): RunEventState {
  switch (event.type) {
    case "run-started": {
      return {
        ...state,
        status: "running",
        planned_executions: event.planned_executions,
        percent_complete: percent(state.completed, event.planned_executions),
      };
    }
    case "run-progress": {
      return {
        ...state,
        completed: event.completed,
        in_progress: event.in_progress,
        pending: event.pending,
        percent_complete: percent(event.completed, state.planned_executions),
      };
    }
    case "test-case-completed": {
      const row = rowFromResult(event.result);
      const key = counterKeyFor(event.result.status);
      const existingModelBucket =
        state.per_model[event.result.model] ?? {
          passed: 0,
          failed: 0,
          error: 0,
          timeout: 0,
        };
      return {
        ...state,
        rows: [...state.rows, row],
        counters: { ...state.counters, [key]: state.counters[key] + 1 },
        per_model: {
          ...state.per_model,
          [event.result.model]: {
            ...existingModelBucket,
            [key]: existingModelBucket[key] + 1,
          },
        },
      };
    }
    case "run-completed": {
      return {
        ...state,
        status: "completed",
        counters: {
          passed: event.summary.passed,
          failed: event.summary.failed,
          error: event.summary.errored,
          timeout: event.summary.timed_out,
        },
        completed: event.summary.completed_executions,
        planned_executions:
          state.planned_executions ?? event.summary.planned_executions,
        percent_complete: percent(
          event.summary.completed_executions,
          state.planned_executions ?? event.summary.planned_executions,
        ),
      };
    }
    case "run-aborted": {
      return {
        ...state,
        status: "aborted",
        terminal_error: event.reason,
      };
    }
    case "run-failed": {
      return {
        ...state,
        status: "failed",
        terminal_error: event.message,
      };
    }
  }
}

/**
 * Convenience helper used by component code and the property test:
 * fold a full event sequence onto ``initialRunState``.
 */
export function foldRunEvents(
  events: readonly RunEventUnion[],
  start: RunEventState = initialRunState,
): RunEventState {
  return events.reduce<RunEventState>(
    (acc, event) => runEventReducer(acc, event),
    start,
  );
}

