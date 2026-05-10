import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import fc from "fast-check";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import {
  render,
  screen,
  waitFor,
  cleanup,
  within,
} from "@testing-library/react";

import { History } from "./History";
import type { RunReport } from "../api/apiClient";
import * as apiClient from "../api/apiClient";

/**
 * Property — UI history row completeness (Task 24.3).
 *
 * For every generated array of ``RunReport`` values, the History view
 * renders one row per report and every row carries the columns the
 * requirement enumerates: ``started_at``, ``status``, model names,
 * suite names, and an aggregate pass rate.
 *
 * Validates: Requirement 16.1.
 */

const statusArb = fc.constantFrom(
  "pending",
  "running",
  "completed",
  "aborted",
  "failed",
) as fc.Arbitrary<RunReport["status"]>;

const modelInfoArb = fc.record({
  name: fc.constantFrom("llama3:8b", "mistral:7b", "phi3:14b"),
  digest: fc.constant(null),
  parameter_size: fc.constant(null),
});

const metricAggregateArb = fc.record({
  mean: fc.double({ min: 0, max: 1, noNaN: true }),
  stddev: fc.double({ min: 0, max: 1, noNaN: true }),
  count: fc.nat({ max: 10 }),
  metric: fc.constantFrom("exact-match", "regex-match"),
});

const modelAggregateArb = fc.record({
  model: fc.constantFrom("llama3:8b", "mistral:7b"),
  passed: fc.nat({ max: 10 }),
  failed: fc.nat({ max: 10 }),
  errored: fc.nat({ max: 5 }),
  timed_out: fc.nat({ max: 5 }),
  mean_ttft_ms: fc.constant(null),
  mean_total_ms: fc.double({ min: 0.1, max: 1_000, noNaN: true }),
  mean_tokens_per_second: fc.constant(null),
  metric_aggregates: fc.dictionary(
    fc.constantFrom("exact-match", "regex-match"),
    metricAggregateArb,
    { maxKeys: 2 },
  ),
});

const runConfigArb = fc.record({
  models: fc.array(fc.constantFrom("llama3:8b", "mistral:7b"), {
    minLength: 1,
    maxLength: 3,
  }),
  suites: fc.array(fc.constantFrom("reasoning-basics", "code-generation"), {
    minLength: 1,
    maxLength: 3,
  }),
  repetitions: fc.integer({ min: 1, max: 3 }),
  concurrency: fc.integer({ min: 1, max: 3 }),
});

const configFileArb = fc.record({
  run: runConfigArb,
  suites_dir: fc.constant("/tmp/suites"),
});

const runReportArb: fc.Arbitrary<RunReport> = fc.record({
  run_id: fc.uuid().map((u) => `run-${u}`),
  status: statusArb,
  started_at: fc.constant("2024-01-01T00:00:00Z"),
  ended_at: fc.constant(null),
  backend_version: fc.constant("0.1.0"),
  ollama_version: fc.constant(null),
  config: configFileArb as unknown as fc.Arbitrary<RunReport["config"]>,
  models: fc.array(modelInfoArb, { minLength: 1, maxLength: 3 }),
  results: fc.constant([]),
  aggregates: fc.array(modelAggregateArb, { minLength: 0, maxLength: 3 }),
  error_summary: fc.constant([]),
}) as unknown as fc.Arbitrary<RunReport>;

describe("History row completeness", () => {
  beforeEach(() => {
    // Clear previously registered spies between property iterations.
    vi.restoreAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  /** Validates: Requirement 16.1. */
  test("renders the required columns for every RunReport", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(runReportArb, { minLength: 1, maxLength: 4 }),
        async (reports) => {
          vi.spyOn(apiClient, "listRuns").mockResolvedValue(reports);

          const queryClient = new QueryClient({
            defaultOptions: { queries: { retry: false } },
          });
          const { unmount } = render(
            <QueryClientProvider client={queryClient}>
              <MemoryRouter initialEntries={["/history"]}>
                <History />
              </MemoryRouter>
            </QueryClientProvider>,
          );

          await waitFor(() => {
            expect(screen.getAllByTestId("history-row").length).toBe(
              reports.length,
            );
          });

          const rows = screen.getAllByTestId("history-row");
          rows.forEach((row, idx) => {
            const scope = within(row);
            const expected = reports[idx]!;

            expect(scope.getByText(expected.started_at)).toBeInTheDocument();
            expect(scope.getByText(expected.status)).toBeInTheDocument();

            const modelLabel = expected.models.map((m) => m.name).join(", ");
            expect(scope.getByText(modelLabel)).toBeInTheDocument();

            const suiteLabel = expected.config.run.suites.join(", ");
            expect(scope.getByText(suiteLabel)).toBeInTheDocument();

            // Pass-rate cell always present — just require a '%' suffix,
            // the exact value comes from the aggregate computation and
            // is validated elsewhere.
            const rateCell = row.querySelector('[data-field="pass_rate"]');
            expect(rateCell).not.toBeNull();
            expect(rateCell!.textContent ?? "").toMatch(/%$/);
          });

          unmount();
        },
      ),
      { numRuns: 5 },
    );
  });
});

