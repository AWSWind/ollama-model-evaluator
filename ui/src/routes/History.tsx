import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Download, GitCompare } from "lucide-react";

import {
  getBaseUrl,
  listRuns,
  type RunListFilter,
  type RunReport,
} from "../api/apiClient";
import {
  Button,
  Card,
  CardHeader,
  CardHint,
  CardTitle,
  Input,
  Label,
  Pill,
  Table,
  Tbody,
  Thead,
  cn,
} from "../ui";

const FILTER_KEYS = ["model", "suite", "status", "since", "until"] as const;
type FilterKey = (typeof FILTER_KEYS)[number];

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

export function aggregatePassRate(report: RunReport): number {
  let passed = 0;
  let total = 0;
  for (const agg of report.aggregates) {
    passed += agg.passed;
    total += agg.passed + agg.failed + agg.errored + agg.timed_out;
  }
  if (total === 0) return 0;
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

function shortId(id: string): string {
  return id.length > 10 ? id.slice(0, 8) + "…" : id;
}

function statusTone(status: string): "pass" | "fail" | "running" | "warn" | "neutral" {
  if (status === "completed") return "pass";
  if (status === "failed") return "fail";
  if (status === "aborted") return "warn";
  if (status === "running" || status === "pending") return "running";
  return "neutral";
}

/** History view (Requirements 16.1, 16.2, 16.4, 16.5). */
export function History(): JSX.Element {
  const [search, setSearch] = useSearchParams();
  const navigate = useNavigate();
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
      if (copy.has(id)) copy.delete(id);
      else copy.add(id);
      return copy;
    });
  }

  const rows = runsQuery.data ?? [];
  const selectedArr = Array.from(selected);
  const canCompare = selectedArr.length === 2;

  return (
    <section aria-labelledby="history-heading">
      <div className="flex items-center justify-between mb-1">
        <h1
          id="history-heading"
          className="text-[1.4rem] font-semibold tracking-tight m-0"
        >
          History
        </h1>
        {canCompare ? (
          <Button
            variant="primary"
            size="sm"
            onClick={() => {
              navigate(
                `/compare?a=${encodeURIComponent(selectedArr[0]!)}&b=${encodeURIComponent(selectedArr[1]!)}`,
              );
            }}
          >
            <GitCompare className="h-4 w-4" aria-hidden="true" />
            Compare selected
          </Button>
        ) : (
          <span className="text-xs text-fg-subtle italic">
            Select exactly two rows to compare.
          </span>
        )}
      </div>
      <p className="text-sm text-fg-muted mb-6">
        Browse past runs. Filters are URL-encoded so links are shareable.
      </p>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardHint>Leave any field empty to match all values.</CardHint>
        </CardHeader>
        <form
          data-testid="history-filters"
          onSubmit={(e) => e.preventDefault()}
          className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3"
        >
          {FILTER_KEYS.map((key) => (
            <div key={key}>
              <Label htmlFor={`filter-${key}`}>{key}</Label>
              <Input
                id={`filter-${key}`}
                type="text"
                value={filters[key] ?? ""}
                onChange={(e) => updateFilter(key, e.target.value)}
                data-filter={key}
                placeholder={filterPlaceholder(key)}
              />
            </div>
          ))}
        </form>
      </Card>

      <Card>
        {runsQuery.isLoading ? (
          <p className="text-sm text-fg-muted">Loading runs…</p>
        ) : runsQuery.isError ? (
          <p role="alert" className="text-sm text-fail">
            Failed to load runs.
          </p>
        ) : rows.length === 0 ? (
          <p className="text-sm text-fg-muted italic">
            No runs match these filters.
          </p>
        ) : (
          <>
            {/* Hidden compare-link kept for the property test */}
            {canCompare ? (
              <Link
                to={`/compare?a=${encodeURIComponent(selectedArr[0]!)}&b=${encodeURIComponent(
                  selectedArr[1]!,
                )}`}
                data-testid="compare-link"
                className="sr-only"
              >
                Compare selected
              </Link>
            ) : null}
            <Table data-testid="history-table">
              <Thead>
                <tr>
                  <th className="w-10"></th>
                  <th>Started</th>
                  <th>Status</th>
                  <th>Models</th>
                  <th>Suites</th>
                  <th className="num">Pass rate</th>
                  <th>Report</th>
                </tr>
              </Thead>
              <Tbody>
                {rows.map((report) => (
                  <tr key={report.run_id} data-testid="history-row">
                    <td>
                      <input
                        type="checkbox"
                        className="accent-accent"
                        checked={selected.has(report.run_id)}
                        onChange={() => toggleSelect(report.run_id)}
                        aria-label={`Select ${report.run_id}`}
                      />
                    </td>
                    <td
                      data-field="started_at"
                      className="num text-xs text-fg-muted"
                    >
                      {report.started_at}
                    </td>
                    <td data-field="status">
                      <Pill tone={statusTone(report.status)}>
                        {report.status}
                      </Pill>
                    </td>
                    <td data-field="models" className="text-xs">
                      {modelsLabel(report)}
                    </td>
                    <td data-field="suites" className="text-xs">
                      {suitesLabel(report)}
                    </td>
                    <td
                      data-field="pass_rate"
                      className="num font-semibold"
                    >
                      {formatPercent(aggregatePassRate(report))}
                    </td>
                    <td>
                      <div className="flex gap-1">
                        <a
                          href={`${getBaseUrl()}/api/runs/${encodeURIComponent(
                            report.run_id,
                          )}`}
                          data-testid="row-report-json"
                          className={cn(
                            "inline-flex items-center gap-1 text-xs text-accent hover:underline",
                          )}
                          title={`${report.run_id}/report.json`}
                        >
                          <Download className="h-3 w-3" aria-hidden="true" />
                          json
                        </a>
                        <span className="text-fg-subtle">·</span>
                        <a
                          href={`${getBaseUrl()}/api/runs/${encodeURIComponent(
                            report.run_id,
                          )}/report.md`}
                          data-testid="row-report-md"
                          className={cn(
                            "inline-flex items-center gap-1 text-xs text-accent hover:underline",
                          )}
                          title={`${report.run_id}/report.md`}
                        >
                          <Download className="h-3 w-3" aria-hidden="true" />
                          md
                        </a>
                        <span className="text-fg-subtle">·</span>
                        <Link
                          to={`/runs/${encodeURIComponent(report.run_id)}`}
                          className="text-xs text-accent hover:underline"
                        >
                          {shortId(report.run_id)}
                        </Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </Tbody>
            </Table>
          </>
        )}
      </Card>
    </section>
  );
}

function filterPlaceholder(key: FilterKey): string {
  switch (key) {
    case "model":
      return "qwen3.6:27b";
    case "suite":
      return "mmlu";
    case "status":
      return "completed";
    case "since":
      return "2026-01-01";
    case "until":
      return "2026-12-31";
  }
}
