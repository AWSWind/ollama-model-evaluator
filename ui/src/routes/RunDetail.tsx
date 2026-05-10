import { useEffect, useReducer, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import {
  cancelRun,
  getBaseUrl,
  getRun,
  type RunReport,
} from "../api/apiClient";
import { RunEventStream, type RunEventStreamStatus } from "../stream/runEvents";
import {
  initialRunState,
  runEventReducer,
  type RunEventState,
  type RunEventUnion,
} from "../stream/runEventState";

/**
 * Hook that wires :class:`RunEventStream` into the pure reducer.
 *
 * Opens a stream on mount, folds every arriving event into
 * :data:`initialRunState` via :func:`runEventReducer`, and exposes the
 * latest folded state plus the current stream status. The stream is
 * closed on unmount or when ``runId`` changes.
 *
 * Kept local to ``RunDetail.tsx`` because no other view owns a live
 * Run connection in v1; promoting it to a separate module would require
 * speculative plumbing without a second consumer.
 */
function useRunEvents(runId: string): {
  state: RunEventState;
  status: RunEventStreamStatus;
} {
  const [state, dispatch] = useReducer(runEventReducer, initialRunState);
  const [status, setStatus] = useState<RunEventStreamStatus>("connecting");

  useEffect(() => {
    const stream = new RunEventStream(
      runId,
      (event) => {
        // ``event`` is whatever JSON the Backend sent; we trust the
        // discriminator matches :type:`RunEventUnion` and let runtime
        // errors surface in the console rather than silently dropping
        // frames. The reducer is exhaustive on the declared tags.
        if (
          typeof event === "object" &&
          event !== null &&
          "type" in (event as Record<string, unknown>)
        ) {
          dispatch(event as RunEventUnion);
        }
      },
      (s) => {
        setStatus(s);
      },
    );
    return () => {
      stream.close();
    };
  }, [runId]);

  return { state, status };
}

function isTerminal(state: RunEventState): boolean {
  return (
    state.status === "completed" ||
    state.status === "aborted" ||
    state.status === "failed"
  );
}

/**
 * Live Run-detail view (Requirements 15.5–15.10, 16.3, 16.5, 16.6).
 *
 * While the Run is live, the view renders the event-stream-driven
 * progress panel, counters, disconnected/polling indicators, and a
 * live-updating execution table. Once the Run is terminal the view
 * additionally fetches the full :class:`RunReport` via ``getRun`` and
 * renders per-test-case details plus download links for the JSON and
 * Markdown reports.
 */
export function RunDetail(): JSX.Element {
  const { runId: rawRunId } = useParams<{ runId: string }>();
  // Use a stable non-empty fallback so hook call order does not depend
  // on runtime ``undefined``; the empty-id branch below short-circuits
  // rendering before any network call is made.
  const runId = rawRunId ?? "";

  const { state, status } = useRunEvents(runId);
  const terminal = isTerminal(state);

  const reportQuery = useQuery({
    queryKey: ["run-report", runId],
    queryFn: () => getRun(runId),
    // Only fetch the full report once the Run is terminal; while the
    // Run is live the WebSocket feed is authoritative and a REST fetch
    // would race with the stream.
    enabled: terminal && runId !== "",
  });

  const [cancelBusy, setCancelBusy] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  async function handleCancel(): Promise<void> {
    if (!window.confirm(`Cancel Run ${runId}?`)) {
      return;
    }
    setCancelBusy(true);
    setCancelError(null);
    try {
      await cancelRun(runId);
    } catch (err) {
      setCancelError((err as Error).message || "Cancel failed");
    } finally {
      setCancelBusy(false);
    }
  }

  const displayStatus = state.status ?? "connecting";
  const percent = Math.round(state.percent_complete);
  const canCancel = state.status === "running" || state.status === "pending";

  const reportJsonHref = `${getBaseUrl()}/api/runs/${encodeURIComponent(runId)}`;
  const reportMdHref = `${getBaseUrl()}/api/runs/${encodeURIComponent(runId)}/report.md`;

  if (!runId) {
    return <p>Missing run id in URL.</p>;
  }

  return (
    <section aria-labelledby="run-detail-heading">
      <h2 id="run-detail-heading">Run {runId}</h2>

      <div data-testid="run-status-badge" role="status">
        Status: <strong>{displayStatus}</strong>
      </div>

      {status === "disconnected" ? (
        <p data-testid="stream-disconnected" role="status">
          Disconnected — reconnecting…
        </p>
      ) : null}
      {status === "polling" ? (
        <p data-testid="stream-polling" role="status">
          Polling for updates…
        </p>
      ) : null}

      <div data-testid="run-progress-bar" aria-label="Progress">
        <div
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          {percent}% complete
          {state.planned_executions != null
            ? ` (${state.completed}/${state.planned_executions})`
            : null}
        </div>
      </div>

      <dl data-testid="run-counters">
        <dt>Passed</dt>
        <dd data-testid="counter-passed">{state.counters.passed}</dd>
        <dt>Failed</dt>
        <dd data-testid="counter-failed">{state.counters.failed}</dd>
        <dt>Error</dt>
        <dd data-testid="counter-error">{state.counters.error}</dd>
        <dt>Timeout</dt>
        <dd data-testid="counter-timeout">{state.counters.timeout}</dd>
      </dl>

      {Object.keys(state.per_model).length > 0 ? (
        <table data-testid="live-per-model">
          <caption>Live per-model counts</caption>
          <thead>
            <tr>
              <th>Model</th>
              <th>Passed</th>
              <th>Failed</th>
              <th>Error</th>
              <th>Timeout</th>
              <th>Pass rate</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(state.per_model)
              .sort()
              .map((model) => {
                const b = state.per_model[model];
                const scored = b.passed + b.failed;
                const rate =
                  scored > 0
                    ? `${((b.passed / scored) * 100).toFixed(1)}%`
                    : "n/a";
                return (
                  <tr
                    key={model}
                    data-testid={`live-per-model-${model}`}
                  >
                    <td>{model}</td>
                    <td>{b.passed}</td>
                    <td>{b.failed}</td>
                    <td>{b.error}</td>
                    <td>{b.timeout}</td>
                    <td>{rate}</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      ) : null}

      {canCancel ? (
        <button
          type="button"
          onClick={handleCancel}
          disabled={cancelBusy}
          data-testid="cancel-button"
        >
          {cancelBusy ? "Cancelling…" : "Cancel Run"}
        </button>
      ) : null}
      {cancelError ? (
        <p role="alert" data-testid="cancel-error">
          {cancelError}
        </p>
      ) : null}

      {state.terminal_error ? (
        <p
          role="alert"
          data-testid="terminal-error"
          style={{ color: "crimson" }}
        >
          {state.terminal_error}
        </p>
      ) : null}

      <h3>Executions</h3>
      <table data-testid="executions-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Suite</th>
            <th>Test Case</th>
            <th>Repetition</th>
            <th>Status</th>
            <th>TTFT (ms)</th>
            <th>Total (ms)</th>
            <th>Tokens/s</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {state.rows.map((row, idx) => (
            <tr
              key={`${row.model}-${row.suite}-${row.test_case_id}-${row.repetition}-${idx}`}
              data-testid="execution-row"
            >
              <td>{row.model}</td>
              <td>{row.suite}</td>
              <td>{row.test_case_id}</td>
              <td>{row.repetition}</td>
              <td>{row.status}</td>
              <td>{row.ttft_ms ?? "—"}</td>
              <td>{row.total_ms}</td>
              <td>{row.tokens_per_second ?? "—"}</td>
              <td>
                {row.metrics.map((m) => (
                  <span key={m.name} style={{ marginRight: "0.5rem" }}>
                    {m.name}: {m.score}
                    {m.passed ? " ✓" : " ✗"}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {terminal ? (
        <section data-testid="terminal-report">
          <h3>Report</h3>
          <p>
            Download:{" "}
            <a href={reportJsonHref} data-testid="report-json-link">
              report.json
            </a>{" "}
            |{" "}
            <a href={reportMdHref} data-testid="report-md-link">
              report.md
            </a>
          </p>
          {reportQuery.isLoading ? <p>Loading report…</p> : null}
          {reportQuery.isError ? (
            <p role="alert">Failed to load report.</p>
          ) : null}
          {reportQuery.data ? (
            <TerminalReport report={reportQuery.data} />
          ) : null}
        </section>
      ) : null}
    </section>
  );
}

/** Per-test-case report details shown on a terminal Run. */
function TerminalReport({ report }: { report: RunReport }): JSX.Element {
  // --- per-model + per-model-per-suite aggregates --------------------
  // Built on the client from the full results list so we do not depend
  // on the Backend pre-computing a multi-model breakdown. Using only
  // ``pass`` + ``fail`` in the denominator keeps pass-rate honest even
  // when a small number of cases errored out or timed out.
  type Bucket = { passed: number; failed: number; errored: number; timedOut: number };
  const newBucket = (): Bucket => ({ passed: 0, failed: 0, errored: 0, timedOut: 0 });

  const modelOrder: string[] = [];
  const suiteOrder: string[] = [];
  const byModel: Record<string, Bucket> = {};
  const byModelSuite: Record<string, Record<string, Bucket>> = {};

  for (const r of report.results) {
    if (!(r.model in byModel)) {
      byModel[r.model] = newBucket();
      byModelSuite[r.model] = {};
      modelOrder.push(r.model);
    }
    if (!suiteOrder.includes(r.suite)) suiteOrder.push(r.suite);
    if (!(r.suite in byModelSuite[r.model])) {
      byModelSuite[r.model][r.suite] = newBucket();
    }

    const add = (b: Bucket): void => {
      if (r.status === "pass") b.passed += 1;
      else if (r.status === "fail") b.failed += 1;
      else if (r.status === "error") b.errored += 1;
      else if (r.status === "timeout") b.timedOut += 1;
    };
    add(byModel[r.model]);
    add(byModelSuite[r.model][r.suite]);
  }

  const fmtRate = (b: Bucket): string => {
    const total = b.passed + b.failed;
    if (total <= 0) return "n/a";
    return `${((b.passed / total) * 100).toFixed(1)}%`;
  };

  return (
    <div data-testid="run-report-details">
      <p>
        Started at: {report.started_at}
        {report.ended_at ? ` · Ended at: ${report.ended_at}` : null}
      </p>

      <h4>Summary by model</h4>
      <table data-testid="summary-by-model">
        <thead>
          <tr>
            <th>Model</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Errored</th>
            <th>Timed out</th>
            <th>Pass rate</th>
          </tr>
        </thead>
        <tbody>
          {modelOrder.map((model) => {
            const b = byModel[model];
            return (
              <tr key={model} data-testid={`summary-model-${model}`}>
                <td>{model}</td>
                <td>{b.passed}</td>
                <td>{b.failed}</td>
                <td>{b.errored}</td>
                <td>{b.timedOut}</td>
                <td>{fmtRate(b)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {modelOrder.length > 0 && suiteOrder.length > 1 ? (
        <>
          <h4>Model × Suite breakdown</h4>
          <table data-testid="summary-model-suite">
            <thead>
              <tr>
                <th>Model</th>
                <th>Suite</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Errored</th>
                <th>Timed out</th>
                <th>Pass rate</th>
              </tr>
            </thead>
            <tbody>
              {modelOrder.flatMap((model) =>
                suiteOrder.flatMap((suite) => {
                  const b = byModelSuite[model]?.[suite];
                  if (!b) return [];
                  return [
                    <tr
                      key={`${model}::${suite}`}
                      data-testid={`summary-cell-${model}-${suite}`}
                    >
                      <td>{model}</td>
                      <td>{suite}</td>
                      <td>{b.passed}</td>
                      <td>{b.failed}</td>
                      <td>{b.errored}</td>
                      <td>{b.timedOut}</td>
                      <td>{fmtRate(b)}</td>
                    </tr>,
                  ];
                }),
              )}
            </tbody>
          </table>
        </>
      ) : null}

      <h4>Per-test-case results</h4>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Suite</th>
            <th>Test Case</th>
            <th>Repetition</th>
            <th>Status</th>
            <th>Response</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {report.results.map((r, idx) => (
            <tr
              key={`${r.model}-${r.suite}-${r.test_case_id}-${r.repetition}-${idx}`}
              data-testid="report-row"
            >
              <td>{r.model}</td>
              <td>{r.suite}</td>
              <td>{r.test_case_id}</td>
              <td>{r.repetition}</td>
              <td>{r.status}</td>
              <td>
                <pre style={{ whiteSpace: "pre-wrap" }}>
                  {r.response ?? r.error_message ?? ""}
                </pre>
              </td>
              <td>
                {r.metrics.map((m) => (
                  <span key={m.name} style={{ marginRight: "0.5rem" }}>
                    {m.name}: {m.score}
                    {m.passed ? " ✓" : " ✗"}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {report.error_summary.length > 0 ? (
        <>
          <h4>Errors</h4>
          <ul>
            {report.error_summary.map((e, idx) => (
              <li key={`${e.test_case_id}-${idx}`}>
                {e.model} / {e.suite} / {e.test_case_id} #{e.repetition}:{" "}
                {e.error_message}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

