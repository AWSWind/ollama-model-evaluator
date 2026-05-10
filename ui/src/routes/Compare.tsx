import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import {
  ApiError,
  compareRuns,
  type ComparisonReport,
} from "../api/apiClient";
import { errorMessages } from "../api/errorMessages";
import {
  Card,
  CardHeader,
  CardHint,
  CardTitle,
  Table,
  Tbody,
  Thead,
  cn,
} from "../ui";

/**
 * Compare view (Requirement 16.4).
 *
 * Reads ``a`` and ``b`` from the URL's ``?`` parameters, fetches
 * ``GET /api/compare``, and renders the metric and performance diff
 * tables. A ``400 no_common_dimensions`` envelope is rendered verbatim
 * so the user can tell exactly why no comparison is available.
 */
export function Compare(): JSX.Element {
  const [search] = useSearchParams();
  const a = search.get("a") ?? "";
  const b = search.get("b") ?? "";

  const compareQuery = useQuery({
    queryKey: ["compare", a, b],
    queryFn: () => compareRuns(a, b),
    enabled: a !== "" && b !== "",
    retry: false,
  });

  if (a === "" || b === "") {
    return (
      <section aria-labelledby="compare-heading">
        <h1
          id="compare-heading"
          className="text-[1.4rem] font-semibold tracking-tight mb-1"
        >
          Compare
        </h1>
        <Card>
          <p role="alert" data-testid="compare-missing-ids" className="text-sm text-fg-muted">
            Select exactly two runs from the History view to compare.
          </p>
        </Card>
      </section>
    );
  }

  if (compareQuery.isLoading) {
    return (
      <section aria-labelledby="compare-heading">
        <h1
          id="compare-heading"
          className="text-[1.4rem] font-semibold tracking-tight mb-1"
        >
          Compare
        </h1>
        <Card>
          <p className="text-sm text-fg-muted">Loading comparison…</p>
        </Card>
      </section>
    );
  }

  if (compareQuery.isError) {
    const err = compareQuery.error;
    const message =
      err instanceof ApiError && err.envelope
        ? err.envelope.message || errorMessages[err.envelope.error_code]
        : "Failed to load comparison.";
    return (
      <section aria-labelledby="compare-heading">
        <h1
          id="compare-heading"
          className="text-[1.4rem] font-semibold tracking-tight mb-1"
        >
          Compare
        </h1>
        <Card>
          <p role="alert" data-testid="compare-error" className="text-sm text-fail">
            {message}
          </p>
        </Card>
      </section>
    );
  }

  const report = compareQuery.data;
  if (!report) {
    return (
      <section aria-labelledby="compare-heading">
        <h1
          id="compare-heading"
          className="text-[1.4rem] font-semibold tracking-tight mb-1"
        >
          Compare
        </h1>
      </section>
    );
  }

  return (
    <section aria-labelledby="compare-heading">
      <h1
        id="compare-heading"
        className="text-[1.4rem] font-semibold tracking-tight mb-1"
      >
        Compare
      </h1>
      <p className="text-sm text-fg-muted mb-6">
        Run A: <code className="font-mono text-fg">{report.run_a}</code> · Run
        B: <code className="font-mono text-fg">{report.run_b}</code>
      </p>
      <CompareTables report={report} />
    </section>
  );
}

/**
 * Rendered exactly once per successful fetch so the property test can
 * assert row-count parity with the input report.
 */
export function CompareTables({
  report,
}: {
  report: ComparisonReport;
}): JSX.Element {
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Metric diffs</CardTitle>
          <CardHint>
            Intersection of (model, metric) pairs present in both runs. Positive
            <span className="text-pass font-semibold"> Δ</span> means Run B
            scored higher.
          </CardHint>
        </CardHeader>
        <Table data-testid="metric-diffs-table">
          <Thead>
            <tr>
              <th>Model</th>
              <th>Metric</th>
              <th className="num">Mean A</th>
              <th className="num">Mean B</th>
              <th className="num">Δ</th>
            </tr>
          </Thead>
          <Tbody>
            {report.metric_diffs.length === 0 ? (
              <tr>
                <td colSpan={5} className="text-center text-fg-muted italic">
                  No common metric dimensions.
                </td>
              </tr>
            ) : (
              report.metric_diffs.map((row, idx) => (
                <tr
                  key={`${row.model}-${row.metric}-${idx}`}
                  data-testid="metric-diff-row"
                >
                  <td className="font-medium">{row.model}</td>
                  <td>{row.metric}</td>
                  <td className="num">{row.mean_a}</td>
                  <td className="num">{row.mean_b}</td>
                  <td
                    className={cn(
                      "num font-semibold",
                      row.diff > 0 && "text-pass",
                      row.diff < 0 && "text-fail",
                    )}
                  >
                    {row.diff > 0 ? "+" : ""}
                    {row.diff}
                  </td>
                </tr>
              ))
            )}
          </Tbody>
        </Table>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Performance diffs</CardTitle>
          <CardHint>Per-model performance delta (B minus A).</CardHint>
        </CardHeader>
        <Table data-testid="performance-diffs-table">
          <Thead>
            <tr>
              <th>Model</th>
              <th className="num">tok/s A</th>
              <th className="num">tok/s B</th>
              <th className="num">ms A</th>
              <th className="num">ms B</th>
              <th className="num">Δ tok/s</th>
              <th className="num">Δ ms</th>
            </tr>
          </Thead>
          <Tbody>
            {report.performance_diffs.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center text-fg-muted italic">
                  No common performance dimensions.
                </td>
              </tr>
            ) : (
              report.performance_diffs.map((row, idx) => (
                <tr
                  key={`${row.model}-${idx}`}
                  data-testid="performance-diff-row"
                >
                  <td className="font-medium">{row.model}</td>
                  <td className="num">{row.mean_tokens_per_second_a ?? "—"}</td>
                  <td className="num">{row.mean_tokens_per_second_b ?? "—"}</td>
                  <td className="num">{row.mean_total_ms_a}</td>
                  <td className="num">{row.mean_total_ms_b}</td>
                  <td
                    className={cn(
                      "num font-semibold",
                      typeof row.tps_diff === "number" && row.tps_diff > 0 && "text-pass",
                      typeof row.tps_diff === "number" && row.tps_diff < 0 && "text-fail",
                    )}
                  >
                    {row.tps_diff ?? "—"}
                  </td>
                  <td
                    className={cn(
                      "num font-semibold",
                      row.total_ms_diff > 0 && "text-fail",
                      row.total_ms_diff < 0 && "text-pass",
                    )}
                  >
                    {row.total_ms_diff}
                  </td>
                </tr>
              ))
            )}
          </Tbody>
        </Table>
      </Card>
    </>
  );
}
