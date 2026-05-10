import { describe, expect, test } from "vitest";
import fc from "fast-check";

import {
  POLL_INTERVAL_MS,
  RECONNECT_DELAYS_MS,
  RunEventStream,
  type RunEventStreamStatus,
  type RunEventStreamWebSocket,
  type SetTimeoutFn,
  type ClearTimeoutFn,
  type TimerHandle,
} from "./runEvents";

/**
 * Property 38 — UI reconnect schedule and polling fallback (Task 23.5).
 *
 * Given a :class:`RunEventStream` backed by a fake ``WebSocket`` and
 * fake timers, we assert:
 *
 * * At most 5 reconnect attempts are scheduled.
 * * The scheduled delays are exactly ``[1000, 2000, 4000, 8000, 16000]``
 *   in order.
 * * After the 5th failed attempt the stream transitions to the
 *   ``"polling"`` status and poll timers are scheduled on a
 *   ``POLL_INTERVAL_MS`` (5s) cadence.
 *
 * ``numRuns`` is small (10) — the stream is deterministic and the
 * assertion space is finite.
 *
 * Validates: Requirements 15.8, 15.9.
 */

interface ScheduledTimer {
  handle: number;
  delay: number;
  fn: () => void;
  cancelled: boolean;
}

interface FakeClock {
  setTimeout: SetTimeoutFn;
  clearTimeout: ClearTimeoutFn;
  timers: ScheduledTimer[];
  /** Run the earliest pending timer if any. */
  advance(): ScheduledTimer | null;
}

function makeFakeClock(): FakeClock {
  let nextHandle = 1;
  const timers: ScheduledTimer[] = [];

  const setTimeoutFn: SetTimeoutFn = (fn, delay) => {
    const timer: ScheduledTimer = {
      handle: nextHandle++,
      delay,
      fn,
      cancelled: false,
    };
    timers.push(timer);
    return timer.handle as unknown as TimerHandle;
  };
  const clearTimeoutFn: ClearTimeoutFn = (handle) => {
    if (handle == null) {
      return;
    }
    const found = timers.find(
      (t) => (t.handle as unknown as typeof handle) === handle,
    );
    if (found) {
      found.cancelled = true;
    }
  };

  return {
    setTimeout: setTimeoutFn,
    clearTimeout: clearTimeoutFn,
    timers,
    advance(): ScheduledTimer | null {
      const pending = timers.filter((t) => !t.cancelled && !("fired" in t));
      if (pending.length === 0) {
        return null;
      }
      const next = pending[0]!;
      (next as ScheduledTimer & { fired: boolean }).fired = true;
      next.fn();
      return next;
    },
  };
}

interface FakeSocket extends RunEventStreamWebSocket {
  url: string;
  triggerClose(code: number): void;
}

function makeFakeWebSocketFactory(): {
  ctor: new (url: string) => FakeSocket;
  sockets: FakeSocket[];
} {
  const sockets: FakeSocket[] = [];
  class FakeWS implements FakeSocket {
    readonly url: string;
    readyState: number = 0;
    onopen: ((event: unknown) => void) | null = null;
    onclose: ((event: { code: number; reason?: string }) => void) | null = null;
    onerror: ((event: unknown) => void) | null = null;
    onmessage: ((event: { data: unknown }) => void) | null = null;

    constructor(url: string) {
      this.url = url;
      sockets.push(this);
    }

    close(): void {
      // No-op: the test drives closes via ``triggerClose``.
    }

    triggerClose(code: number): void {
      this.readyState = 3;
      this.onclose?.({ code });
    }
  }
  return { ctor: FakeWS, sockets };
}

describe("RunEventStream reconnect schedule", () => {
  /** Validates: Requirements 15.8, 15.9. */
  test(
    "schedules at most 5 reconnects with the documented delays then switches to polling",
    () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 4001, max: 4999 }).filter((c) => c !== 1000),
          (closeCode) => {
            const clock = makeFakeClock();
            const { ctor, sockets } = makeFakeWebSocketFactory();
            const pollCalls: string[] = [];
            const statuses: RunEventStreamStatus[] = [];

            const stream = new RunEventStream(
              "run-1",
              () => {
                // Events not exercised in this test.
              },
              (s) => {
                statuses.push(s);
              },
              {
                WebSocket: ctor,
                setTimeout: clock.setTimeout,
                clearTimeout: clock.clearTimeout,
                baseUrl: "http://localhost:8765",
                poll: async (id) => {
                  pollCalls.push(id);
                  return {};
                },
              },
            );

            // First connection attempt opens socket 0; close it with a
            // non-1000 code to trigger the reconnect ladder.
            expect(sockets.length).toBe(1);
            sockets[0]!.triggerClose(closeCode);

            const observedDelays: number[] = [];
            for (let i = 0; i < RECONNECT_DELAYS_MS.length; i += 1) {
              // Only the latest pending timer should be a reconnect
              // timer at this point (no poll timers yet).
              const pending = clock.timers.filter(
                (t) => !t.cancelled && !("fired" in t),
              );
              expect(pending.length).toBe(1);
              observedDelays.push(pending[0]!.delay);

              // Fire the reconnect delay — the stream opens a fresh
              // socket, which we close with the same non-1000 code.
              clock.advance();
              expect(sockets.length).toBe(i + 2);
              sockets[i + 1]!.triggerClose(closeCode);
            }

            expect(observedDelays).toEqual(RECONNECT_DELAYS_MS);

            // After 5 failures, no further reconnect timers, and the
            // stream has entered polling. A single poll tick has fired
            // synchronously (the ``tick()`` entry), scheduling the next
            // poll on a 5-second interval.
            const afterFailures = clock.timers.filter(
              (t) => !t.cancelled && !("fired" in t),
            );
            expect(afterFailures.length).toBe(1);
            expect(afterFailures[0]!.delay).toBe(POLL_INTERVAL_MS);
            expect(pollCalls).toEqual(["run-1"]);
            expect(statuses.at(-1)).toBe("polling");

            // Fire the next poll tick; delay should stay 5s and no new
            // WebSocket is constructed.
            clock.advance();
            expect(sockets.length).toBe(RECONNECT_DELAYS_MS.length + 1);
            expect(pollCalls.length).toBe(2);

            // Ensure overall reconnect attempts equal exactly 5.
            expect(sockets.length - 1).toBe(RECONNECT_DELAYS_MS.length);

            stream.close();
          },
        ),
        { numRuns: 10 },
      );
    },
  );
});

