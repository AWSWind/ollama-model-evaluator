import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import {
  ApiError,
  compareRuns,
  type ComparisonReport,
} from "../api/apiClient";
import { errorMessages } from "../api/errorMessages";

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
        <h2 id="compare-heading">Compare</h2>
        <p role="alert" data-testid="compare-missing-ids">
          Select exactly two runs from the History view to compare.
        </p>
      </section>
    );
  }

  if (compareQuery.isLoading) {
    return (
      <section aria-labelledby="compare-heading">
        <h2 id="compare-heading">Compare</h2>
        <p>Loading comparison…</p>
      </section>
    );
  }

  if (compareQuery.isError) {
    const err = compareQuery.error;
    if (err instanceof ApiError && err.envelope) {
      const env = err.envelope;
      const message = env.message || errorMessages[env.error_code];
      return (
        <section aria-labelledby="compare-heading">
          <h2 id="compare-heading">Compare</h2>
          <p role="alert" data-testid="compare-error">
            {message}
          </p>
        </section>
      );
    }
    return (
      <section aria-labelledby="compare-heading">
        <h2 id="compare-heading">Compare</h2>
        <p role="alert" data-testid="compare-error">
          Failed to load comparison.
        </p>
      </section>
    );
  }

  const report = compareQuery.data;
  if (!report) {
    return (
      <section aria-labelledby="compare-heading">
        <h2 id="compare-heading">Compare</h2>
      </section>
    );
  }
  return (
    <section aria-labelledby="compare-heading">
      <h2 id="compare-heading">Compare</h2>
      <p>
        Run A: <code>{report.run_a}</code> · Run B: <code>{report.run_b}</code>
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
      <h3>Metric diffs</h3>
      <table data-testid="metric-diffs-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Metric</th>
            <th>Mean A</th>
            <th>Mean B</th>
            <th>Diff</th>
          </tr>
        </thead>
        <tbody>
          {report.metric_diffs.map((row, idx) => (
            <tr
              key={`${row.model}-${row.metric}-${idx}`}
              data-testid="metric-diff-row"
            >
              <td>{row.model}</td>
              <td>{row.metric}</td>
              <td>{row.mean_a}</td>
              <td>{row.mean_b}</td>
              <td>{row.diff}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Performance diffs</h3>
      <table data-testid="performance-diffs-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Mean tok/s A</th>
            <th>Mean tok/s B</th>
            <th>Mean total ms A</th>
            <th>Mean total ms B</th>
            <th>Δ tok/s</th>
            <th>Δ total ms</th>
          </tr>
        </thead>
        <tbody>
          {report.performance_diffs.map((row, idx) => (
            <tr key={`${row.model}-${idx}`} data-testid="performance-diff-row">
              <td>{row.model}</td>
              <td>{row.mean_tokens_per_second_a ?? "—"}</td>
              <td>{row.mean_tokens_per_second_b ?? "—"}</td>
              <td>{row.mean_total_ms_a}</td>
              <td>{row.mean_total_ms_b}</td>
              <td>{row.tps_diff ?? "—"}</td>
              <td>{row.total_ms_diff}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

