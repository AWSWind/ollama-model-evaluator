import { describe, expect, test, afterEach } from "vitest";
import fc from "fast-check";
import { render, cleanup } from "@testing-library/react";

import { CompareTables } from "./Compare";
import type { ComparisonReport } from "../api/apiClient";

/**
 * Property — UI comparison table rendering (Task 24.5).
 *
 * For any generated :class:`ComparisonReport`, :func:`CompareTables`
 * renders exactly ``|metric_diffs|`` rows in the metric table and
 * ``|performance_diffs|`` rows in the performance table.
 *
 * Validates: Requirement 16.4.
 */

const modelArb = fc.constantFrom("llama3:8b", "mistral:7b", "phi3:14b");
const metricArb = fc.constantFrom("exact-match", "regex-match", "contains");

const metricDiffArb = fc.record({
  model: modelArb,
  metric: metricArb,
  mean_a: fc.double({ min: 0, max: 1, noNaN: true }),
  mean_b: fc.double({ min: 0, max: 1, noNaN: true }),
  diff: fc.double({ min: -1, max: 1, noNaN: true }),
});

const perfDiffArb = fc.record({
  model: modelArb,
  mean_tokens_per_second_a: fc.option(
    fc.double({ min: 0, max: 500, noNaN: true }),
    { nil: null },
  ),
  mean_tokens_per_second_b: fc.option(
    fc.double({ min: 0, max: 500, noNaN: true }),
    { nil: null },
  ),
  mean_total_ms_a: fc.double({ min: 0.1, max: 10_000, noNaN: true }),
  mean_total_ms_b: fc.double({ min: 0.1, max: 10_000, noNaN: true }),
  tps_diff: fc.option(fc.double({ min: -500, max: 500, noNaN: true }), {
    nil: null,
  }),
  total_ms_diff: fc.double({ min: -10_000, max: 10_000, noNaN: true }),
});

const comparisonArb: fc.Arbitrary<ComparisonReport> = fc.record({
  run_a: fc.constant("run-a"),
  run_b: fc.constant("run-b"),
  metric_diffs: fc.array(metricDiffArb, { maxLength: 6 }),
  performance_diffs: fc.array(perfDiffArb, { maxLength: 6 }),
});

describe("CompareTables row counts", () => {
  afterEach(() => {
    cleanup();
  });

  /** Validates: Requirement 16.4. */
  test("renders |metric_diffs| and |performance_diffs| rows", () => {
    fc.assert(
      fc.property(comparisonArb, (report) => {
        const { container, unmount } = render(<CompareTables report={report} />);
        const metricRows = container.querySelectorAll(
          '[data-testid="metric-diff-row"]',
        );
        const perfRows = container.querySelectorAll(
          '[data-testid="performance-diff-row"]',
        );
        expect(metricRows.length).toBe(report.metric_diffs.length);
        expect(perfRows.length).toBe(report.performance_diffs.length);
        unmount();
      }),
      { numRuns: 20 },
    );
  });
});

