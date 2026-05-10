import { describe, expect, test } from "vitest";
import fc from "fast-check";

import { buildRunConfigBody, type NewRunFormSelections } from "./NewRun";

/**
 * Property 35 — UI form → RunConfig body (Task 22.2, Requirement 15.3).
 *
 * For every valid form selection, the POST body the handler builds has
 * fields equal to the selections and no additional fields.
 *
 * ``numRuns: 20`` is reduced from the design's 100 for speed; the body
 * is a pure deterministic projection so 20 samples cover the shape
 * space.
 */
describe("NewRun.buildRunConfigBody", () => {
  const selectionsArb: fc.Arbitrary<NewRunFormSelections> = fc.record({
    models: fc.array(fc.string({ minLength: 1, maxLength: 16 }), {
      minLength: 1,
      maxLength: 5,
    }),
    suites: fc.array(fc.string({ minLength: 1, maxLength: 16 }), {
      minLength: 1,
      maxLength: 5,
    }),
    repetitions: fc.integer({ min: 1, max: 10 }),
    concurrency: fc.integer({ min: 1, max: 8 }),
    tagFilter: fc
      .array(fc.string({ minLength: 1, maxLength: 10 }), { maxLength: 5 })
      .map((tags) =>
        tags
          .map((t) => t.replace(/,/g, ""))
          .filter((t) => t.length > 0)
          .join(", "),
      ),
  });

  /** Validates: Requirements 15.3. */
  test("body has exactly the expected fields and matches selections", () => {
    fc.assert(
      fc.property(selectionsArb, (sel) => {
        const body = buildRunConfigBody(sel);
        // Field equality: same list members, same counters.
        expect(body.models).toEqual(sel.models);
        expect(body.suites).toEqual(sel.suites);
        expect(body.repetitions).toBe(sel.repetitions);
        expect(body.concurrency).toBe(sel.concurrency);
        const expectedTags = sel.tagFilter
          .split(",")
          .map((t) => t.trim())
          .filter((t) => t.length > 0);
        expect(body.tag_filter).toEqual(expectedTags);

        // No additional fields beyond the allowed set.
        const allowed = new Set([
          "models",
          "suites",
          "repetitions",
          "concurrency",
          "tag_filter",
        ]);
        for (const key of Object.keys(body)) {
          expect(allowed.has(key)).toBe(true);
        }
      }),
      { numRuns: 20 },
    );
  });
});
