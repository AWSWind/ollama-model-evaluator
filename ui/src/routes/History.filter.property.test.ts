import { describe, expect, test } from "vitest";
import fc from "fast-check";

import {
  decodeHistoryFilters,
  encodeHistoryFilters,
} from "./History";
import type { RunListFilter } from "../api/apiClient";

/**
 * Property — UI filter query round-trip (Task 24.4).
 *
 * ``decodeHistoryFilters(encodeHistoryFilters(filters)) === filters``
 * for every :class:`RunListFilter`.
 *
 * The encoder normalises the "absent" and "empty string" cases (both
 * omit the key) so the property is phrased against the *normalised*
 * view of the filter — any key whose value is ``undefined`` or ``""``
 * is removed. This mirrors the Backend's filter parser, which treats
 * missing and blank parameters identically.
 *
 * Validates: Requirement 16.2.
 */

const safeValueArb = fc
  .string({ minLength: 1, maxLength: 20 })
  // Avoid characters that produce ambiguous encoding (``&`` and ``=``
  // round-trip fine through URLSearchParams but reduce reader signal).
  .map((s) => s.replace(/[&=#]/g, "a"));

const filterArb: fc.Arbitrary<RunListFilter> = fc.record(
  {
    model: fc.option(safeValueArb, { nil: undefined }),
    suite: fc.option(safeValueArb, { nil: undefined }),
    status: fc.option(
      fc.constantFrom("pending", "running", "completed", "aborted", "failed"),
      { nil: undefined },
    ),
    since: fc.option(fc.constant("2024-01-01T00:00:00Z"), { nil: undefined }),
    until: fc.option(fc.constant("2024-12-31T00:00:00Z"), { nil: undefined }),
  },
  { requiredKeys: [] },
);

function normalise(f: RunListFilter): RunListFilter {
  const out: RunListFilter = {};
  for (const key of ["model", "suite", "status", "since", "until"] as const) {
    const value = f[key];
    if (value !== undefined && value !== "") {
      out[key] = value;
    }
  }
  return out;
}

describe("encodeHistoryFilters / decodeHistoryFilters", () => {
  /** Validates: Requirement 16.2. */
  test("encode then decode is the identity on normalised filters", () => {
    fc.assert(
      fc.property(filterArb, (filters) => {
        const encoded = encodeHistoryFilters(filters);
        const decoded = decodeHistoryFilters(encoded);
        expect(decoded).toEqual(normalise(filters));
      }),
      { numRuns: 20 },
    );
  });

  test("decoder ignores unknown keys", () => {
    const encoded = "model=llama3:8b&foo=bar&baz=qux";
    expect(decodeHistoryFilters(encoded)).toEqual({ model: "llama3:8b" });
  });

  test("leading ? is tolerated", () => {
    const encoded = "?status=completed";
    expect(decodeHistoryFilters(encoded)).toEqual({ status: "completed" });
  });
});

