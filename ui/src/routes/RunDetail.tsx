import { useEffect, useReducer, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Download, XCircle } from "lucide-react";

import {
  cancelRun,
  getBaseUrl,
  getRun,
  type RunReport,
} from "../api/apiClient";
import {
  RunEventStream,
  type RunEventStreamStatus,
} from "../stream/runEvents";
import {
  initialRunState,
  runEventReducer,
  type RunEventState,
  type RunEventUnion,
} from "../stream/runEventState";
import {
  Button,
  Card,
  CardHeader,
  CardHint,
  CardTitle,
  Pill,
  Progress,
  Table,
  Tbody,
  Thead,
  cn,
} from "../ui";

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
        if (
          typeof event === "object" &&
          event !== null &&
          "type" in (event as Record<string, unknown>)
        ) {
          dispatch(event as RunEventUnion);
        }
      },
      (s) => setStatus(s),
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

function tonePill(status: string): "pass" | "fail" | "running" | "warn" | "neutral" {
  if (status === "completed") return "pass";
  if (status === "failed") return "fail";
  if (status === "aborted") return "warn";
  if (status === "running" || status === "pending") return "running";
  return "neutral";
}

/** Live Run-detail view (Requirements 15.5–15.10, 16.3, 16.5, 16.6). */
export function RunDetail(): JSX.Element {
  const { runId: rawRunId } = useParams<{ runId: string }>();
  const runId = rawRunId ?? "";

  const { state, status } = useRunEvents(runId);
  const terminal = isTerminal(state);

  const reportQuery = useQuery({
    queryKey: ["run-report", runId],
    queryFn: () => getRun(runId),
    enabled: terminal && runId !== "",
  });

  const [cancelBusy, setCancelBusy] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  async function handleCancel(): Promise<void> {
    if (!window.confirm(`Cancel Run ${runId}?`)) return;
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
    return <p className="text-fg-muted">Missing run id in URL.</p>;
  }

  return (
    <section aria-labelledby="run-detail-heading">
      <div className="flex items-center gap-3 mb-1">
        <h1
          id="run-detail-heading"
          className="text-[1.4rem] font-semibold tracking-tight m-0"
        >
          Run
        </h1>
        <code className="text-sm font-mono px-2 py-0.5 bg-bg-alt rounded border border-border text-fg-muted">
          {runId}
        </code>
      </div>
      <div className="flex items-center gap-2 mb-6 text-sm">
        <span className="text-fg-muted">Status:</span>
        <Pill tone={tonePill(displayStatus)} dot={displayStatus === "running"}>
          <span data-testid="run-status-badge">{displayStatus}</span>
        </Pill>
        {status === "disconnected" ? (
          <span
            data-testid="stream-disconnected"
            role="status"
            className="text-xs text-warn"
          >
            Disconnected — reconnecting…
          </span>
        ) : null}
        {status === "polling" ? (
          <span
            data-testid="stream-polling"
            role="status"
            className="text-xs text-warn"
          >
            Polling for updates…
          </span>
        ) : null}
      </div>

      <Card>
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-sm font-semibold">Progress</p>
            <p className="text-xs text-fg-muted num mt-0.5">
              {percent}% complete
              {state.planned_executions != null
                ? ` · ${state.completed} / ${state.planned_executions}`
                : null}
            </p>
          </div>
          {canCancel ? (
            <Button
              variant="secondary"
              size="sm"
              onClick={handleCancel}
              disabled={cancelBusy}
              data-testid="cancel-button"
            >
              <XCircle className="h-4 w-4" aria-hidden="true" />
              {cancelBusy ? "Cancelling…" : "Cancel run"}
            </Button>
          ) : null}
        </div>

        <Progress
          value={percent}
          aria-label="Run progress"
          className="mb-4"
        />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Kpi label="Passed" value={state.counters.passed} tone="pass" />
          <Kpi label="Failed" value={state.counters.failed} tone="fail" />
          <Kpi label="Errored" value={state.counters.error} />
          <Kpi label="Timeout" value={state.counters.timeout} />
        </div>

        <dl data-testid="run-counters" className="sr-only">
          <dt>Passed</dt>
          <dd data-testid="counter-passed">{state.counters.passed}</dd>
          <dt>Failed</dt>
          <dd data-testid="counter-failed">{state.counters.failed}</dd>
          <dt>Error</dt>
          <dd data-testid="counter-error">{state.counters.error}</dd>
          <dt>Timeout</dt>
          <dd data-testid="counter-timeout">{state.counters.timeout}</dd>
        </dl>

        {cancelError ? (
          <p
            role="alert"
            data-testid="cancel-error"
            className="mt-3 text-xs text-fail"
          >
            {cancelError}
          </p>
        ) : null}

        {state.terminal_error ? (
          <p
            role="alert"
            data-testid="terminal-error"
            className="mt-3 text-sm text-fail"
          >
            {state.terminal_error}
          </p>
        ) : null}
      </Card>

      {Object.keys(state.per_model).length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Live per-model counts</CardTitle>
            <CardHint>Updates as each test case completes.</CardHint>
          </CardHeader>
          <Table data-testid="live-per-model">
            <Thead>
              <tr>
                <th>Model</th>
                <th className="num">Passed</th>
                <th className="num">Failed</th>
                <th className="num">Error</th>
                <th className="num">Timeout</th>
                <th className="num">Pass rate</th>
              </tr>
            </Thead>
            <Tbody>
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
                      <td className="font-medium">{model}</td>
                      <td className="num text-pass">{b.passed}</td>
                      <td className="num text-fail">{b.failed}</td>
                      <td className="num">{b.error}</td>
                      <td className="num">{b.timeout}</td>
                      <td className="num font-semibold">{rate}</td>
                    </tr>
                  );
                })}
            </Tbody>
          </Table>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Executions</CardTitle>
          <CardHint>
            One row per test-case-completed event, in arrival order.
          </CardHint>
        </CardHeader>
        <Table data-testid="executions-table">
          <Thead>
            <tr>
              <th>Model</th>
              <th>Suite</th>
              <th>Test Case</th>
              <th className="num">Rep</th>
              <th>Status</th>
              <th className="num">TTFT (ms)</th>
              <th className="num">Total (ms)</th>
              <th className="num">Tokens/s</th>
              <th>Metrics</th>
            </tr>
          </Thead>
          <Tbody>
            {state.rows.length === 0 ? (
              <tr>
                <td colSpan={9} className="text-center text-fg-muted">
                  No results yet.
                </td>
              </tr>
            ) : (
              state.rows.map((row, idx) => (
                <tr
                  key={`${row.model}-${row.suite}-${row.test_case_id}-${row.repetition}-${idx}`}
                  data-testid="execution-row"
                >
                  <td>{row.model}</td>
                  <td>{row.suite}</td>
                  <td className="font-mono text-xs text-fg-muted">
                    {row.test_case_id}
                  </td>
                  <td className="num">{row.repetition}</td>
                  <td>
                    <Pill
                      tone={
                        row.status === "pass"
                          ? "pass"
                          : row.status === "fail"
                            ? "fail"
                            : row.status === "error" || row.status === "timeout"
                              ? "warn"
                              : "neutral"
                      }
                    >
                      {row.status}
                    </Pill>
                  </td>
                  <td className="num">{row.ttft_ms ?? "—"}</td>
                  <td className="num">{row.total_ms}</td>
                  <td className="num">{row.tokens_per_second ?? "—"}</td>
                  <td>
                    {row.metrics.map((m) => (
                      <span
                        key={m.name}
                        className="inline-flex items-center gap-1 mr-2 text-xs"
                      >
                        <span className="text-fg-muted">{m.name}:</span>
                        <span className={cn("num", m.passed ? "text-pass" : "text-fail")}>
                          {m.score}
                          {m.passed ? " ✓" : " ✗"}
                        </span>
                      </span>
                    ))}
                  </td>
                </tr>
              ))
            )}
          </Tbody>
        </Table>
      </Card>

      {terminal ? (
        <section data-testid="terminal-report">
          <h2 className="text-lg font-semibold tracking-tight mt-8 mb-3">
            Report
          </h2>
          <Card>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-fg-muted">Download:</span>
              <a
                href={reportJsonHref}
                data-testid="report-json-link"
                className={cn(
                  "inline-flex items-center gap-2 rounded-md border border-border",
                  "bg-bg text-fg px-3 py-1.5 text-xs font-medium",
                  "hover:bg-bg-alt hover:border-border-strong transition-colors",
                )}
              >
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                report.json
              </a>
              <a
                href={reportMdHref}
                data-testid="report-md-link"
                className={cn(
                  "inline-flex items-center gap-2 rounded-md border border-border",
                  "bg-bg text-fg px-3 py-1.5 text-xs font-medium",
                  "hover:bg-bg-alt hover:border-border-strong transition-colors",
                )}
              >
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                report.md
              </a>
            </div>
          </Card>
          {reportQuery.isLoading ? (
            <p className="text-sm text-fg-muted">Loading report…</p>
          ) : null}
          {reportQuery.isError ? (
            <p role="alert" className="text-sm text-fail">
              Failed to load report.
            </p>
          ) : null}
          {reportQuery.data ? (
            <TerminalReport report={reportQuery.data} />
          ) : null}
        </section>
      ) : null}
    </section>
  );
}

function Kpi({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | string;
  tone?: "pass" | "fail";
}): JSX.Element {
  return (
    <div className="rounded-md border border-border bg-bg-alt px-4 py-3">
      <p className="text-xxs uppercase tracking-wider font-semibold text-fg-muted">
        {label}
      </p>
      <p
        className={cn(
          "text-2xl font-semibold num mt-0.5 leading-tight",
          tone === "pass" && "text-pass",
          tone === "fail" && "text-fail",
        )}
      >
        {value}
      </p>
    </div>
  );
}

/** Terminal-report detail view (after Run finishes). */
function TerminalReport({ report }: { report: RunReport }): JSX.Element {
  type Bucket = {
    passed: number;
    failed: number;
    errored: number;
    timedOut: number;
  };
  const newBucket = (): Bucket => ({
    passed: 0,
    failed: 0,
    errored: 0,
    timedOut: 0,
  });

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
      <Card>
        <p className="text-xs text-fg-muted mb-3 num">
          Started: {report.started_at}
          {report.ended_at ? ` · Ended: ${report.ended_at}` : null}
        </p>

        <CardHeader>
          <CardTitle>Summary by model</CardTitle>
        </CardHeader>
        <Table data-testid="summary-by-model">
          <Thead>
            <tr>
              <th>Model</th>
              <th className="num">Passed</th>
              <th className="num">Failed</th>
              <th className="num">Errored</th>
              <th className="num">Timed out</th>
              <th className="num">Pass rate</th>
            </tr>
          </Thead>
          <Tbody>
            {modelOrder.map((model) => {
              const b = byModel[model];
              return (
                <tr
                  key={model}
                  data-testid={`summary-model-${model}`}
                >
                  <td className="font-medium">{model}</td>
                  <td className="num text-pass">{b.passed}</td>
                  <td className="num text-fail">{b.failed}</td>
                  <td className="num">{b.errored}</td>
                  <td className="num">{b.timedOut}</td>
                  <td className="num font-semibold">{fmtRate(b)}</td>
                </tr>
              );
            })}
          </Tbody>
        </Table>
      </Card>

      {modelOrder.length > 0 && suiteOrder.length > 1 ? (
        <Card>
          <CardHeader>
            <CardTitle>Model × Suite breakdown</CardTitle>
          </CardHeader>
          <Table data-testid="summary-model-suite">
            <Thead>
              <tr>
                <th>Model</th>
                <th>Suite</th>
                <th className="num">Passed</th>
                <th className="num">Failed</th>
                <th className="num">Errored</th>
                <th className="num">Timed out</th>
                <th className="num">Pass rate</th>
              </tr>
            </Thead>
            <Tbody>
              {modelOrder.flatMap((model) =>
                suiteOrder.flatMap((suite) => {
                  const b = byModelSuite[model]?.[suite];
                  if (!b) return [];
                  return [
                    <tr
                      key={`${model}::${suite}`}
                      data-testid={`summary-cell-${model}-${suite}`}
                    >
                      <td className="font-medium">{model}</td>
                      <td>{suite}</td>
                      <td className="num text-pass">{b.passed}</td>
                      <td className="num text-fail">{b.failed}</td>
                      <td className="num">{b.errored}</td>
                      <td className="num">{b.timedOut}</td>
                      <td className="num font-semibold">{fmtRate(b)}</td>
                    </tr>,
                  ];
                }),
              )}
            </Tbody>
          </Table>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Per-test-case results</CardTitle>
        </CardHeader>
        <Table>
          <Thead>
            <tr>
              <th>Model</th>
              <th>Suite</th>
              <th>Test Case</th>
              <th className="num">Rep</th>
              <th>Status</th>
              <th>Response</th>
              <th>Metrics</th>
            </tr>
          </Thead>
          <Tbody>
            {report.results.map((r, idx) => (
              <tr
                key={`${r.model}-${r.suite}-${r.test_case_id}-${r.repetition}-${idx}`}
                data-testid="report-row"
              >
                <td>{r.model}</td>
                <td>{r.suite}</td>
                <td className="font-mono text-xs text-fg-muted">
                  {r.test_case_id}
                </td>
                <td className="num">{r.repetition}</td>
                <td>
                  <Pill
                    tone={
                      r.status === "pass"
                        ? "pass"
                        : r.status === "fail"
                          ? "fail"
                          : "warn"
                    }
                  >
                    {r.status}
                  </Pill>
                </td>
                <td>
                  <pre className="whitespace-pre-wrap text-xs text-fg-muted m-0 max-w-md">
                    {r.response ?? r.error_message ?? ""}
                  </pre>
                </td>
                <td>
                  {r.metrics.map((m) => (
                    <span
                      key={m.name}
                      className="inline-flex items-center gap-1 mr-2 text-xs"
                    >
                      <span className="text-fg-muted">{m.name}:</span>
                      <span className={cn("num", m.passed ? "text-pass" : "text-fail")}>
                        {m.score}
                        {m.passed ? " ✓" : " ✗"}
                      </span>
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </Tbody>
        </Table>
      </Card>

      {report.error_summary.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Errors</CardTitle>
          </CardHeader>
          <ul className="space-y-1 text-sm">
            {report.error_summary.map((e, idx) => (
              <li
                key={`${e.test_case_id}-${idx}`}
                className="text-fg-muted"
              >
                <span className="font-mono text-xs text-fg">
                  {e.model} / {e.suite} / {e.test_case_id} #{e.repetition}
                </span>
                : {e.error_message}
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
    </div>
  );
}
