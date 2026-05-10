import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import {
  getBaseUrl,
  listRuns,
  type RunListFilter,
  type RunReport,
} from "../api/apiClient";

/**
 * Canonical set of filter keys supported by the History view.
 *
 * We keep the list explicit so the URL-encoding helpers can round-trip
 * strictly: unknown keys in ``?search`` are stripped by the decoder and
 * the encoder never writes keys outside this set (Property 39, Task
 * 24.4).
 */
const FILTER_KEYS = ["model", "suite", "status", "since", "until"] as const;
type FilterKey = (typeof FILTER_KEYS)[number];

/**
 * Serialise a :class:`RunListFilter` to a ``URLSearchParams``-style
 * string (no leading ``?``). Keys with ``undefined`` or empty-string
 * values are omitted so the round-trip collapses the "absent" and
 * "empty string" cases — the Backend filter parser treats them the
 * same way.
 */
export function encodeHistoryFilters(filters: RunListFilter): string {
  const params = new URLSearchParams();
  for (const key of FILTER_KEYS) {
    const value = filters[key];
    if (value !== undefined && value !== "") {
      params.append(key, value);
    }
  }
  return params.toString();
}

/**
 * Parse a query-string (with or without the leading ``?``) back into a
 * :class:`RunListFilter`. Unknown keys are dropped so the round-trip is
 * stable under arbitrary extra parameters (e.g. the ``a``/``b`` keys
 * used by the Compare route).
 */
export function decodeHistoryFilters(search: string): RunListFilter {
  const trimmed = search.startsWith("?") ? search.slice(1) : search;
  const params = new URLSearchParams(trimmed);
  const out: RunListFilter = {};
  for (const key of FILTER_KEYS) {
    const value = params.get(key);
    if (value !== null && value !== "") {
      out[key] = value;
    }
  }
  return out;
}

/**
 * Aggregate pass rate across every :class:`ModelAggregate` in a
 * :class:`RunReport`. Returns ``0`` when the denominator is zero
 * (e.g. a ``failed`` Run that produced no results).
 */
export function aggregatePassRate(report: RunReport): number {
  let passed = 0;
  let total = 0;
  for (const agg of report.aggregates) {
    passed += agg.passed;
    total += agg.passed + agg.failed + agg.errored + agg.timed_out;
  }
  if (total === 0) {
    return 0;
  }
  return passed / total;
}

function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

function modelsLabel(report: RunReport): string {
  return report.models.map((m) => m.name).join(", ");
}

function suitesLabel(report: RunReport): string {
  return report.config.run.suites.join(", ");
}

/**
 * History view (Requirements 16.1, 16.2, 16.4, 16.5).
 */
export function History(): JSX.Element {
  const [search, setSearch] = useSearchParams();
  const filters = useMemo<RunListFilter>(
    () => decodeHistoryFilters(search.toString()),
    [search],
  );

  const runsQuery = useQuery({
    queryKey: ["runs", filters],
    queryFn: () => listRuns(filters),
  });

  const [selected, setSelected] = useState<Set<string>>(new Set());

  function updateFilter(key: FilterKey, value: string): void {
    const next = new URLSearchParams(search);
    if (value === "") {
      next.delete(key);
    } else {
      next.set(key, value);
    }
    setSearch(next, { replace: true });
  }

  function toggleSelect(id: string): void {
    setSelected((prev) => {
      const copy = new Set(prev);
      if (copy.has(id)) {
        copy.delete(id);
      } else {
        copy.add(id);
      }
      return copy;
    });
  }

  const rows = runsQuery.data ?? [];
  const selectedArr = Array.from(selected);
  const canCompare = selectedArr.length === 2;

  return (
    <section aria-labelledby="history-heading">
      <h2 id="history-heading">History</h2>

      <form data-testid="history-filters" onSubmit={(e) => e.preventDefault()}>
        {FILTER_KEYS.map((key) => (
          <label key={key} style={{ marginRight: "1rem" }}>
            {key}
            <input
              type="text"
              value={filters[key] ?? ""}
              onChange={(e) => updateFilter(key, e.target.value)}
              data-filter={key}
            />
          </label>
        ))}
      </form>

      {runsQuery.isLoading ? <p>Loading runs…</p> : null}
      {runsQuery.isError ? <p role="alert">Failed to load runs.</p> : null}

      {canCompare ? (
        <Link
          to={`/compare?a=${encodeURIComponent(selectedArr[0]!)}&b=${encodeURIComponent(
            selectedArr[1]!,
          )}`}
          data-testid="compare-link"
        >
          Compare selected
        </Link>
      ) : null}

      <table data-testid="history-table">
        <thead>
          <tr>
            <th>Select</th>
            <th>Started</th>
            <th>Status</th>
            <th>Models</th>
            <th>Suites</th>
            <th>Pass rate</th>
            <th>Report</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((report) => (
            <tr key={report.run_id} data-testid="history-row">
              <td>
                <input
                  type="checkbox"
                  checked={selected.has(report.run_id)}
                  onChange={() => toggleSelect(report.run_id)}
                  aria-label={`Select ${report.run_id}`}
                />
              </td>
              <td data-field="started_at">{report.started_at}</td>
              <td data-field="status">{report.status}</td>
              <td data-field="models">{modelsLabel(report)}</td>
              <td data-field="suites">{suitesLabel(report)}</td>
              <td data-field="pass_rate">
                {formatPercent(aggregatePassRate(report))}
              </td>
              <td>
                <a
                  href={`${getBaseUrl()}/api/runs/${encodeURIComponent(
                    report.run_id,
                  )}`}
                  data-testid="row-report-json"
                >
                  {report.run_id}/report.json
                </a>{" "}
                |{" "}
                <a
                  href={`${getBaseUrl()}/api/runs/${encodeURIComponent(
                    report.run_id,
                  )}/report.md`}
                  data-testid="row-report-md"
                >
                  {report.run_id}/report.md
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

