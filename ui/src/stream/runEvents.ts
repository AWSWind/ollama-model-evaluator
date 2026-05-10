/**
 * WebSocket-based live Run event client with reconnect-then-poll fallback.
 *
 * ``RunEventStream`` owns the full lifecycle of a single subscription to
 * ``GET /api/runs/{id}/events`` and transparently falls back to REST
 * polling when the WebSocket cannot be re-established.
 *
 * Lifecycle (Requirements 15.8, 15.9):
 *
 * 1. Open a WebSocket to ``{wsBaseUrl}/api/runs/{runId}/events``.
 * 2. On each successful connection emit ``onStatus("connected")`` and
 *    forward every parsed event via ``onEvent``.
 * 3. On a non-1000 close, schedule a reconnect with delays
 *    ``[1s, 2s, 4s, 8s, 16s]`` (capped at 5 attempts). Between attempts
 *    emit ``onStatus("disconnected")``.
 * 4. After 5 consecutive failed attempts, switch to polling
 *    ``GET /api/runs/{id}`` every 5 seconds and emit
 *    ``onStatus("polling")`` once per poll tick. Polling never calls
 *    ``onEvent`` because we do not have raw events from a REST response;
 *    callers are expected to refresh their view from their own RunReport
 *    query when they see ``status === "polling"``.
 * 5. A terminal WebSocket close with code 1000 transitions the status
 *    to ``"closed"`` and does not reconnect.
 *
 * The WebSocket constructor, ``setTimeout``, and ``clearTimeout`` may be
 * injected via the ``options`` bag so property tests can drive the
 * reconnect schedule under fake timers deterministically (Task 23.5,
 * Property 38).
 */

import { getBaseUrl } from "../api/apiClient";

/**
 * Observable status of a :class:`RunEventStream`.
 *
 * * ``connecting`` — a WebSocket ``open`` is outstanding.
 * * ``connected`` — the WebSocket is open and forwarding events.
 * * ``disconnected`` — the WebSocket closed with a non-1000 code and
 *   the next reconnect is scheduled (or being attempted).
 * * ``polling`` — reconnect budget exhausted; falling back to REST.
 * * ``closed`` — the WebSocket closed cleanly (code 1000) or the
 *   consumer called :meth:`close`; no further callbacks will fire.
 */
export type RunEventStreamStatus =
  | "connecting"
  | "connected"
  | "disconnected"
  | "polling"
  | "closed";

/**
 * Minimal :class:`WebSocket`-compatible interface consumed by
 * :class:`RunEventStream`. The real ``WebSocket`` class satisfies this
 * shape; tests can also supply a fake.
 */
export interface RunEventStreamWebSocket {
  readonly readyState: number;
  close(code?: number, reason?: string): void;
  onopen: ((event: unknown) => void) | null;
  onclose: ((event: { code: number; reason?: string }) => void) | null;
  onerror: ((event: unknown) => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
}

/**
 * WebSocket constructor. Mirrors the native ``WebSocket`` signature so
 * ``window.WebSocket`` can be passed in directly.
 */
export type RunEventStreamWebSocketCtor = new (
  url: string,
) => RunEventStreamWebSocket;

/**
 * Node/browser ``setTimeout`` signature, narrowed so tests can inject a
 * fake-timers implementation that returns an arbitrary handle type.
 */
export type TimerHandle = ReturnType<typeof setTimeout>;
export type SetTimeoutFn = (handler: () => void, ms: number) => TimerHandle;
export type ClearTimeoutFn = (handle: TimerHandle | null | undefined) => void;

/**
 * Optional injection points so tests can drive the reconnect schedule
 * with fake timers and a controllable ``WebSocket``. All fields default
 * to the corresponding globals at construction time.
 */
export interface RunEventStreamOptions {
  /** Constructor used to open WebSocket connections. */
  WebSocket?: RunEventStreamWebSocketCtor;
  /** Implementation used to schedule reconnect and poll delays. */
  setTimeout?: SetTimeoutFn;
  /** Implementation used to cancel scheduled delays. */
  clearTimeout?: ClearTimeoutFn;
  /** Override the base URL (defaults to :func:`getBaseUrl`). */
  baseUrl?: string;
  /**
   * Override the function used to poll for Run state while in the
   * polling fallback. Defaults to ``fetch(baseUrl + '/api/runs/:id')``.
   * The return value is ignored by the stream; consumers should
   * refresh their own RunReport query when they observe the
   * ``"polling"`` status.
   */
  poll?: (runId: string) => Promise<unknown>;
}

/**
 * Reconnect backoff schedule, in milliseconds (Requirement 15.8).
 *
 * Exposed so Property 38's fast-check test can assert the exact values.
 */
export const RECONNECT_DELAYS_MS: readonly number[] = [
  1_000, 2_000, 4_000, 8_000, 16_000,
] as const;

/**
 * Polling interval used in the REST fallback (Requirement 15.9).
 */
export const POLL_INTERVAL_MS = 5_000;

/**
 * Per-run event-stream client.
 *
 * The class is intentionally small: it owns a single WebSocket (or a
 * single polling timer), wires up reconnect, and translates transport
 * events into ``onEvent``/``onStatus`` callbacks. State fold-ing is
 * performed by :mod:`./runEventState`, not here, so this file stays
 * testable without a React tree.
 */
export class RunEventStream {
  private readonly runId: string;
  private readonly onEvent: (event: unknown) => void;
  private readonly onStatus: (status: RunEventStreamStatus) => void;

  private readonly ws_ctor: RunEventStreamWebSocketCtor;
  private readonly setTimeoutFn: SetTimeoutFn;
  private readonly clearTimeoutFn: ClearTimeoutFn;
  private readonly baseUrl: string;
  private readonly poll: (runId: string) => Promise<unknown>;

  private socket: RunEventStreamWebSocket | null = null;
  private reconnectTimer: TimerHandle | null = null;
  private pollTimer: TimerHandle | null = null;

  /** Number of consecutive failed reconnect attempts. */
  private attempt: number = 0;
  /** Set once :meth:`close` has been invoked by the consumer. */
  private closed: boolean = false;

  constructor(
    runId: string,
    onEvent: (event: unknown) => void,
    onStatus: (status: RunEventStreamStatus) => void,
    options: RunEventStreamOptions = {},
  ) {
    this.runId = runId;
    this.onEvent = onEvent;
    this.onStatus = onStatus;

    // Resolve injection points once at construction; binding globals
    // eagerly makes the behaviour easier to reason about under tests
    // that swap ``globalThis.setTimeout`` *after* the stream starts.
    const defaultWs = (globalThis as { WebSocket?: RunEventStreamWebSocketCtor })
      .WebSocket;
    this.ws_ctor =
      options.WebSocket ?? (defaultWs as RunEventStreamWebSocketCtor);
    this.setTimeoutFn =
      options.setTimeout ??
      (((fn: () => void, ms: number) =>
        setTimeout(fn, ms)) as SetTimeoutFn);
    this.clearTimeoutFn =
      options.clearTimeout ??
      (((handle: TimerHandle | null | undefined) => {
        if (handle != null) {
          clearTimeout(handle);
        }
      }) as ClearTimeoutFn);

    this.baseUrl = options.baseUrl ?? getBaseUrl();
    this.poll =
      options.poll ??
      (async (id: string) => {
        const res = await fetch(`${this.baseUrl}/api/runs/${encodeURIComponent(id)}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        return (await res.json()) as unknown;
      });

    this.connect();
  }

  /**
   * Cancel any pending reconnect/poll timers, close the underlying
   * WebSocket (with code 1000), and prevent any further callbacks.
   */
  close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.cancelReconnect();
    this.cancelPoll();
    if (this.socket) {
      // Detach handlers before closing so a late ``onclose`` from the
      // browser does not re-enter our state machine after ``close()``.
      this.socket.onopen = null;
      this.socket.onclose = null;
      this.socket.onerror = null;
      this.socket.onmessage = null;
      try {
        this.socket.close(1000, "client-closed");
      } catch {
        // Swallow: some fake WebSocket implementations throw on close.
      }
      this.socket = null;
    }
    this.onStatus("closed");
  }

  private wsUrl(): string {
    const httpBase = this.baseUrl;
    const wsBase = httpBase.replace(/^http/i, (match) =>
      match.toLowerCase() === "http" ? "ws" : "wss",
    );
    return `${wsBase}/api/runs/${encodeURIComponent(this.runId)}/events`;
  }

  private connect(): void {
    if (this.closed) {
      return;
    }
    this.onStatus("connecting");
    let socket: RunEventStreamWebSocket;
    try {
      socket = new this.ws_ctor(this.wsUrl());
    } catch {
      // Constructor threw (e.g. invalid URL). Treat as an immediate
      // failure and honour the reconnect schedule.
      this.handleClose(4000);
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      if (this.closed) {
        return;
      }
      // Successful open resets the reconnect budget for the *next*
      // disconnection: a run that drops after a healthy period gets a
      // fresh 5-attempt window rather than inheriting old failures.
      this.attempt = 0;
      this.onStatus("connected");
    };

    socket.onmessage = (event) => {
      if (this.closed) {
        return;
      }
      const data = event.data;
      if (typeof data !== "string") {
        return;
      }
      try {
        const parsed = JSON.parse(data) as unknown;
        this.onEvent(parsed);
      } catch {
        // Ignore malformed frames rather than tearing the stream down;
        // the Backend owns the wire format and any parse error here
        // points at the server side.
      }
    };

    socket.onerror = () => {
      // ``onclose`` follows an error, which is where we actually
      // schedule the next attempt. Keeping ``onerror`` as a no-op
      // avoids counting a single failure twice.
    };

    socket.onclose = (event) => {
      if (this.closed) {
        return;
      }
      this.handleClose(event?.code ?? 4000);
    };
  }

  private handleClose(code: number): void {
    this.socket = null;
    if (code === 1000) {
      // Clean close: the Backend signals end-of-stream after the
      // terminal event. Transition to ``closed`` and stop.
      this.closed = true;
      this.onStatus("closed");
      return;
    }
    if (this.attempt >= RECONNECT_DELAYS_MS.length) {
      this.startPolling();
      return;
    }
    this.onStatus("disconnected");
    const delay = RECONNECT_DELAYS_MS[this.attempt];
    this.attempt += 1;
    this.reconnectTimer = this.setTimeoutFn(() => {
      this.reconnectTimer = null;
      if (this.closed) {
        return;
      }
      this.connect();
    }, delay);
  }

  private startPolling(): void {
    this.cancelReconnect();
    const tick = (): void => {
      if (this.closed) {
        return;
      }
      // Emit the status *per tick* (not only on entry) because
      // Requirement 15.9 describes a sustained polling state that the
      // UI uses to keep a visible banner on screen. The status callback
      // is idempotent for consumers — they render from the latest
      // reported value.
      this.onStatus("polling");
      void this.poll(this.runId).catch(() => {
        // Polling errors are silent: the UI already has a "polling"
        // indicator and should continue trying at the fixed cadence.
      });
      this.pollTimer = this.setTimeoutFn(tick, POLL_INTERVAL_MS);
    };
    tick();
  }

  private cancelReconnect(): void {
    if (this.reconnectTimer != null) {
      this.clearTimeoutFn(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private cancelPoll(): void {
    if (this.pollTimer != null) {
      this.clearTimeoutFn(this.pollTimer);
      this.pollTimer = null;
    }
  }
}

