import { describe, expect, test } from "vitest";
import fc from "fast-check";

import {
  foldRunEvents,
  initialRunState,
  runEventReducer,
  type RunEventUnion,
} from "./runEventState";

/**
 * Property 37 — UI event-stream state reducer (Task 23.4).
 *
 * For any well-formed event sequence, applying :func:`runEventReducer`
 * left-to-right produces the same state as applying it again after
 * dropping any prefix and replaying it. In other words, given
 * ``events = prefix ++ suffix``:
 *
 *     fold(events) == fold(suffix, fold(prefix))
 *
 * which is a standard left-fold associativity / replay-determinism
 * check. Because the reducer is pure, this reduces to the statement
 * "the reducer never observes external state".
 *
 * Validates: Requirements 15.6, 15.7, 16.3, 16.6.
 */

const runIdArb = fc.constant("run-1");
const tsArb = fc.constant("2024-01-01T00:00:00Z");
const seqArb = fc.nat({ max: 1_000 });
const modelArb = fc.constantFrom("llama3:8b", "mistral:7b");
const suiteArb = fc.constantFrom("reasoning-basics", "code-generation");
const caseIdArb = fc.stringMatching(/^[a-z0-9-]{1,10}$/);

const metricResultArb = fc.record({
  name: fc.constantFrom("exact-match", "regex-match", "contains"),
  score: fc.double({ min: 0, max: 1, noNaN: true }),
  passed: fc.boolean(),
  threshold: fc.option(fc.double({ min: 0, max: 1, noNaN: true }), {
    nil: null,
  }),
  details: fc.constant({}),
  error: fc.option(fc.string({ maxLength: 30 }), { nil: null }),
});

const performanceArb = fc.record({
  ttft_ms: fc.option(fc.double({ min: 0, max: 5_000, noNaN: true }), {
    nil: null,
  }),
  total_ms: fc.double({ min: 0.001, max: 10_000, noNaN: true }),
  prompt_tokens: fc.option(fc.nat({ max: 1_000 }), { nil: null }),
  response_tokens: fc.option(fc.nat({ max: 1_000 }), { nil: null }),
  tokens_per_second: fc.option(fc.double({ min: 0, max: 500, noNaN: true }), {
    nil: null,
  }),
});

const testCaseResultArb = fc.record({
  model: modelArb,
  suite: suiteArb,
  test_case_id: caseIdArb,
  repetition: fc.integer({ min: 1, max: 5 }),
  status: fc.constantFrom("pass", "fail", "error", "timeout") as fc.Arbitrary<
    "pass" | "fail" | "error" | "timeout"
  >,
  response: fc.option(fc.string({ maxLength: 30 }), { nil: null }),
  error_message: fc.option(fc.string({ maxLength: 30 }), { nil: null }),
  performance: performanceArb,
  metrics: fc.array(metricResultArb, { minLength: 0, maxLength: 3 }),
});

const runStartedArb = fc.record({
  type: fc.constant("run-started" as const),
  run_id: runIdArb,
  seq: seqArb,
  ts: tsArb,
  planned_executions: fc.nat({ max: 50 }),
});

const runProgressArb = fc.record({
  type: fc.constant("run-progress" as const),
  run_id: runIdArb,
  seq: seqArb,
  ts: tsArb,
  completed: fc.nat({ max: 50 }),
  in_progress: fc.nat({ max: 5 }),
  pending: fc.nat({ max: 50 }),
});

const testCaseCompletedArb = fc.record({
  type: fc.constant("test-case-completed" as const),
  run_id: runIdArb,
  seq: seqArb,
  ts: tsArb,
  result: testCaseResultArb,
});

const runSummaryArb = fc.record({
  planned_executions: fc.nat({ max: 50 }),
  completed_executions: fc.nat({ max: 50 }),
  passed: fc.nat({ max: 50 }),
  failed: fc.nat({ max: 50 }),
  errored: fc.nat({ max: 50 }),
  timed_out: fc.nat({ max: 50 }),
});

const runCompletedArb = fc.record({
  type: fc.constant("run-completed" as const),
  run_id: runIdArb,
  seq: seqArb,
  ts: tsArb,
  summary: runSummaryArb,
});

const runAbortedArb = fc.record({
  type: fc.constant("run-aborted" as const),
  run_id: runIdArb,
  seq: seqArb,
  ts: tsArb,
  reason: fc.string({ minLength: 1, maxLength: 40 }),
});

const runFailedArb = fc.record({
  type: fc.constant("run-failed" as const),
  run_id: runIdArb,
  seq: seqArb,
  ts: tsArb,
  error_code: fc.constantFrom("ollama_unreachable", "dataset_fetch_failed"),
  message: fc.string({ minLength: 1, maxLength: 40 }),
});

const eventArb: fc.Arbitrary<RunEventUnion> = fc.oneof(
  runStartedArb,
  runProgressArb,
  testCaseCompletedArb,
  runCompletedArb,
  runAbortedArb,
  runFailedArb,
) as fc.Arbitrary<RunEventUnion>;

describe("runEventReducer", () => {
  /** Validates: Requirements 15.6, 15.7, 16.3, 16.6. */
  test("fold is replay-deterministic under any prefix split", () => {
    fc.assert(
      fc.property(
        fc.array(eventArb, { maxLength: 30 }),
        fc.nat(),
        (events, cut) => {
          const split = events.length === 0 ? 0 : cut % (events.length + 1);
          const prefix = events.slice(0, split);
          const suffix = events.slice(split);

          const whole = foldRunEvents(events);
          const replayed = foldRunEvents(suffix, foldRunEvents(prefix));

          // Structural equality; both branches share ``initialRunState``
          // and the reducer returns plain objects so Vitest's deep
          // matcher is the right tool.
          expect(replayed).toEqual(whole);
        },
      ),
      { numRuns: 20 },
    );
  });

  /** Sanity: empty sequence returns the initial state verbatim. */
  test("empty fold equals initialRunState", () => {
    expect(foldRunEvents([])).toEqual(initialRunState);
    // Single-event fold is the same as one reducer step.
    const event: RunEventUnion = {
      type: "run-started",
      run_id: "r",
      seq: 0,
      ts: "2024-01-01T00:00:00Z",
      planned_executions: 3,
    };
    expect(foldRunEvents([event])).toEqual(
      runEventReducer(initialRunState, event),
    );
  });
});

