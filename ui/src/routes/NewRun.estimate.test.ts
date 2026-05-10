/**
 * Unit tests for the NewRun-route estimate helpers.
 *
 * These helpers are pure, so property-based testing would be overkill —
 * a handful of table-driven cases is enough to lock in the shape of the
 * output (string format, edge cases like zero and large durations).
 */

import { describe, expect, it } from "vitest";

import { estimateRunSeconds, formatDuration } from "./NewRun";

describe("estimateRunSeconds", () => {
  it("returns 0 when any multiplicative input is non-positive", () => {
    expect(estimateRunSeconds(0, 10, 1, 1)).toBe(0);
    expect(estimateRunSeconds(1, 0, 1, 1)).toBe(0);
    expect(estimateRunSeconds(1, 10, 0, 1)).toBe(0);
  });

  it("scales linearly with models, cases, and reps", () => {
    // 1 model × 10 cases × 1 rep / 1 concurrency × 15s = 150s
    expect(estimateRunSeconds(1, 10, 1, 1, 15)).toBe(150);
    // 2 models × 10 cases × 1 rep / 1 concurrency × 15s = 300s
    expect(estimateRunSeconds(2, 10, 1, 1, 15)).toBe(300);
    // 1 model × 10 cases × 3 reps / 1 concurrency × 15s = 450s
    expect(estimateRunSeconds(1, 10, 3, 1, 15)).toBe(450);
  });

  it("divides serial time by concurrency (ceiling division)", () => {
    // 1 × 10 × 1 / 3 = ceil(3.33) = 4 batches × 15s
    expect(estimateRunSeconds(1, 10, 1, 3, 15)).toBe(60);
    // 1 × 9 × 1 / 3 = 3 batches exactly × 15s
    expect(estimateRunSeconds(1, 9, 1, 3, 15)).toBe(45);
  });

  it("clamps concurrency to a minimum of 1", () => {
    // concurrency 0 would be a divide-by-zero; must degrade to serial
    expect(estimateRunSeconds(1, 5, 1, 0, 10)).toBe(50);
    expect(estimateRunSeconds(1, 5, 1, -3, 10)).toBe(50);
  });
});

describe("formatDuration", () => {
  it("returns 0s for zero / negative / non-finite inputs", () => {
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(-5)).toBe("0s");
    expect(formatDuration(Number.NaN)).toBe("0s");
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe("0s");
  });

  it("formats sub-minute durations in seconds", () => {
    expect(formatDuration(1)).toBe("1s");
    expect(formatDuration(59)).toBe("59s");
  });

  it("formats minute-range durations with optional seconds", () => {
    expect(formatDuration(60)).toBe("1m");
    expect(formatDuration(65)).toBe("1m 5s");
    expect(formatDuration(600)).toBe("10m");
    expect(formatDuration(754)).toBe("12m 34s");
  });

  it("formats hour-range durations as H + rounded M", () => {
    expect(formatDuration(3600)).toBe("1h");
    expect(formatDuration(3660)).toBe("1h 1m");
    expect(formatDuration(5400)).toBe("1h 30m");
    expect(formatDuration(7200)).toBe("2h");
  });
});
