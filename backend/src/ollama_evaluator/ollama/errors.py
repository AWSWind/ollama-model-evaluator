"""Exception types raised by :mod:`ollama_evaluator.ollama.client`.

These exceptions are the narrow error surface the :class:`OllamaClient`
translates HTTP and network failures into. They are deliberately simple
record classes rather than a rich hierarchy because the retry and
lifecycle policies that act on them live one layer up, in the scheduler
(Task 12.2) and the preflight step (Task 12.4):

* :class:`OllamaHTTPError` — the Ollama_Server returned a non-2xx HTTP
  response. Carries the numeric status, the request URL, and the
  decoded response body so callers (and log output) can classify the
  failure without re-issuing the request. The scheduler retries a
  subset of statuses (502/503/504) and treats the rest as terminal
  ``error`` on the affected Test_Case (Requirements 1.5, 5.6, 11.1,
  11.2).

* :class:`OllamaConnectionError` — a network-layer failure that the
  caller chose to wrap. The client itself **does not** wrap
  ``httpx.ConnectError``/``httpx.ReadError``/``httpx.TimeoutException``
  — those propagate unchanged so the retry policy in Task 12.2 can
  pattern-match on the original httpx types. This class is provided as
  a convenience for higher layers (for example, the preflight step
  in Task 12.4) that prefer a single wrapped error when reporting
  ``ollama_unreachable`` to the user (Requirement 1.3).

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Retry and timeout policy.
"""

from __future__ import annotations


class OllamaHTTPError(Exception):
    """Raised when the Ollama_Server returns a non-2xx HTTP response.

    Populated by :class:`~ollama_evaluator.ollama.client.OllamaClient`
    for every REST and streaming call. The scheduler (Task 12.2)
    classifies the error by ``status``:

    * ``502``, ``503``, ``504`` → retriable up to ``retry_max_attempts``.
    * Every other status (notably the 4xx family) → terminal ``error``.

    The full response ``body`` is preserved verbatim so operators
    debugging a failure can read the Ollama_Server's own message
    without enabling extra logging. No body truncation is performed
    at this layer; callers that log exceptions are responsible for
    clipping if needed.

    Attributes:
        status: HTTP status code returned by the Ollama_Server.
        url: Path (relative to ``base_url``) or absolute URL the client
            dispatched the request to. Uses the path as provided to the
            client, so log output matches the call-site.
        body: Decoded response body as a string. Best-effort UTF-8
            decoding with ``errors="replace"`` is applied upstream so
            this field is always a valid Python ``str``.
    """

    def __init__(self, status: int, url: str, body: str) -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"Ollama HTTP {status} at {url}: {body}")


class OllamaConnectionError(Exception):
    """Wrapper for network-level failures reaching the Ollama_Server.

    The :class:`~ollama_evaluator.ollama.client.OllamaClient` does
    *not* raise this directly — it lets ``httpx.ConnectError``,
    ``httpx.ReadError``, and ``httpx.TimeoutException`` propagate so
    the retry/timeout policy in Task 12.2 can classify them against
    the native httpx types. Higher layers (preflight, CLI error
    surface) can catch one of those and re-raise as
    :class:`OllamaConnectionError` when they prefer a single, named
    exception for the ``ollama_unreachable`` error code
    (Requirement 1.3).

    Attributes:
        url: The ``base_url`` (or endpoint URL) the caller was trying
            to reach. Present so the wrapped message matches
            Requirement 1.3's "identifies the unreachable URL".
        cause: Optional underlying exception the caller is wrapping.
            Preserved on the instance so log handlers can choose to
            render both the wrapper message and the original traceback.
    """

    def __init__(self, url: str, cause: Exception | None = None) -> None:
        self.url = url
        self.cause = cause
        msg = f"Could not connect to Ollama at {url}"
        if cause is not None:
            msg = f"{msg}: {cause}"
        super().__init__(msg)


__all__ = ["OllamaConnectionError", "OllamaHTTPError"]
