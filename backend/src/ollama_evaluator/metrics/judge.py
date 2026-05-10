"""LLM-as-judge scoring metric (Task 5.3).

This module implements the ``llm-as-judge`` metric listed in the
design document's §Metric framework pass/score table (Requirement
7.2). The metric asks a second Ollama model (the "judge") to score a
candidate response against a natural-language rubric and returns the
parsed score normalised to ``[0, 1]``.

The metric satisfies the :class:`Metric` protocol defined in
:mod:`ollama_evaluator.metrics.base`: it is a stateless class with a
``name`` attribute and an ``async score(response, ctx)`` method. All
per-invocation configuration is pulled from
:attr:`MetricContext.metric_config.params`; the judge client and model
tag are carried on :attr:`MetricContext.judge_client` and
:attr:`MetricContext.judge_model` so that purely local metrics can
share the same context type without requiring a judge dependency they
never use (see :mod:`ollama_evaluator.metrics.base`).

Error-handling contract (Requirement 7.5)
-----------------------------------------

Two distinct classes of error are handled differently, matching the
built-in metric policy in :mod:`ollama_evaluator.metrics.builtin`:

* **Metric configuration errors** — missing/malformed ``rubric``,
  missing ``judge_client`` or ``judge_model`` on the context — **raise**
  :class:`ValueError`. The runner catches these at the metric-call
  boundary and converts them into a :class:`MetricResult` with
  ``error`` populated and ``passed=False``. Raising keeps each metric's
  implementation small and uniform — the runner owns the "never crash
  a Run because of one broken metric" policy, not each metric.

* **Judge response errors** — malformed ``"Score: X/Y"`` output, or a
  well-formed line whose numbers violate ``0 ≤ X ≤ Y`` (``X > Y``) or
  ``Y > 0`` (``Y == 0``) — **return** a :class:`MetricResult` with
  ``score=0.0``, ``passed=False``, ``error`` set to a short diagnostic
  string, and ``details.judge_response`` carrying the raw judge text.
  These are failures of the candidate model or judge, not metric
  configuration bugs, so they belong in the per-``TestCaseResult``
  error field rather than aborting via an exception.

Prompt shape
------------

The prompt is intentionally boring and asks the judge for a single-line
``"Score: X/Y"`` answer. Two design decisions:

* A compact, greppable response line keeps the parser trivial: a single
  regex that is hard to false-match.
* Embedding the rubric, the test-case prompt, and the expected output
  (when present) gives the judge enough context to score without
  round-tripping additional state.

The default scale bound ``scale_max`` is used **only for documentation**
— the metric does not enforce ``Y == scale_max``. If a judge returns a
higher-denominator score that still satisfies ``0 ≤ X ≤ Y``, the metric
normalises verbatim and records ``x``/``y`` in ``details`` so the user
can inspect the judge's choice.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import MetricResult
from ..ollama.types import GenerateOptions
from .base import MetricContext

# The parser is intentionally a single-line regex that only looks for the
# literal ``Score:`` token followed by an integer / integer pair. We use
# :func:`re.search` so the judge may wrap the score in explanatory text
# (e.g. ``"Reasoning: … Score: 4/5"``) without breaking the extractor.
_SCORE_PATTERN = re.compile(r"Score:\s*(\d+)\s*/\s*(\d+)")


class LlmAsJudge:
    """LLM-based scoring metric that delegates to a configured judge model.

    Parameters (from ``MetricConfig.params``):

    * ``rubric`` (``str``, required): free-form text describing how the
      judge should evaluate the response. Embedded verbatim into the
      judge prompt.
    * ``threshold`` (``float``, default ``0.7``): minimum normalised
      score for ``passed=True``. Applied as ``passed = score >=
      threshold`` after the judge output is parsed and normalised.
    * ``scale_max`` (``int``, default ``100``): upper bound ``Y``
      advertised to the judge in the prompt. Documentation only — the
      metric does not reject ``Y != scale_max`` at parse time.
    * ``judge_options`` (``dict | None``, default ``None``): overrides
      passed through to :class:`GenerateOptions` when calling the judge.
      ``None`` (the default) sends ``{"temperature": 0.0}`` so the
      judge produces deterministic output; passing an explicit dict
      takes full control and opts out of that default entirely.

    Context (from :class:`MetricContext`):

    * ``judge_client`` must be an async-iterator-returning Ollama
      client (typically :class:`OllamaClient`) exposing
      ``generate(model, prompt, options=...) -> AsyncIterator[GenerateChunk]``.
    * ``judge_model`` must be a non-empty model tag string
      (e.g. ``"llama3:8b"``).

    Missing or malformed required parameters raise :class:`ValueError`.
    Malformed judge output or parsed scores that violate ``0 ≤ X ≤ Y``
    or ``Y > 0`` produce a :class:`MetricResult` with ``error`` set
    (see module docstring).
    """

    name = "llm-as-judge"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        params = ctx.metric_config.params

        # --- parameter validation (metric config errors raise) ---------------

        rubric = self._require_rubric(params)
        threshold = self._optional_threshold(params)
        scale_max = self._optional_scale_max(params)
        judge_options = self._optional_judge_options(params)

        # --- context validation (metric config errors raise) ----------------

        if ctx.judge_client is None:
            raise ValueError(
                "llm-as-judge requires MetricContext.judge_client to be set"
            )
        if not isinstance(ctx.judge_model, str) or not ctx.judge_model:
            raise ValueError(
                "llm-as-judge requires MetricContext.judge_model to be a "
                "non-empty string"
            )

        # --- prompt construction --------------------------------------------

        prompt = self._build_prompt(
            rubric=rubric,
            test_case_prompt=ctx.test_case.prompt,
            expected_output=ctx.test_case.expected_output,
            response=response,
            scale_max=scale_max,
        )

        # --- call the judge --------------------------------------------------
        #
        # ``judge_options or {"temperature": 0.0}`` falls back to a
        # deterministic-temperature default when the caller passes ``None``
        # *or* an empty dict. The spec in tasks.md is explicit about that
        # semantics — an empty dict is treated the same as ``None`` here.

        options = GenerateOptions(**(judge_options or {"temperature": 0.0}))
        chunks: list[str] = []
        async for chunk in ctx.judge_client.generate(
            ctx.judge_model, prompt, options=options
        ):
            chunks.append(chunk.response)
        raw = "".join(chunks)

        # --- parse and validate the judge output ----------------------------

        match = _SCORE_PATTERN.search(raw)
        if match is None:
            return self._error_result(
                metric_name=ctx.metric_config.name,
                threshold=threshold,
                message="malformed judge output",
                details={"judge_response": raw},
            )

        x = int(match.group(1))
        y = int(match.group(2))

        # Check ``Y == 0`` before ``X > Y`` because a zero denominator
        # makes ``X / Y`` undefined regardless of ``X``.
        if y == 0:
            return self._error_result(
                metric_name=ctx.metric_config.name,
                threshold=threshold,
                message="scale is zero",
                details={"judge_response": raw, "x": x, "y": y},
            )
        if x > y:
            return self._error_result(
                metric_name=ctx.metric_config.name,
                threshold=threshold,
                message="score exceeds scale",
                details={"judge_response": raw, "x": x, "y": y},
            )

        normalised = x / y
        return MetricResult(
            name=ctx.metric_config.name,
            score=normalised,
            passed=normalised >= threshold,
            threshold=threshold,
            details={"judge_response": raw, "x": x, "y": y},
        )

    # ------------------------------------------------------------------
    # Parameter parsing helpers (kept private so the module surface area
    # stays just the class plus :func:`register_judge_metric`).
    # ------------------------------------------------------------------

    @staticmethod
    def _require_rubric(params: dict[str, Any]) -> str:
        if "rubric" not in params:
            raise ValueError("llm-as-judge metric parameter 'rubric' is required")
        value = params["rubric"]
        if not isinstance(value, str):
            raise ValueError(
                "llm-as-judge metric parameter 'rubric' must be of type str "
                f"(got {type(value).__name__})"
            )
        if not value.strip():
            raise ValueError(
                "llm-as-judge metric parameter 'rubric' must be a non-empty string"
            )
        return value

    @staticmethod
    def _optional_threshold(params: dict[str, Any]) -> float:
        value = params.get("threshold", 0.7)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # ``bool`` is a subclass of ``int`` in Python — exclude it
            # explicitly so ``threshold=True`` doesn't silently mean 1.0.
            raise ValueError(
                "llm-as-judge metric parameter 'threshold' must be a real number "
                f"(got {type(value).__name__})"
            )
        if value != value:  # NaN guard — NaN breaks the ``>=`` comparison.
            raise ValueError(
                "llm-as-judge metric parameter 'threshold' must not be NaN"
            )
        return float(value)

    @staticmethod
    def _optional_scale_max(params: dict[str, Any]) -> int:
        value = params.get("scale_max", 100)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "llm-as-judge metric parameter 'scale_max' must be an int "
                f"(got {type(value).__name__})"
            )
        if value <= 0:
            raise ValueError(
                f"llm-as-judge metric parameter 'scale_max' must be > 0 (got {value})"
            )
        return value

    @staticmethod
    def _optional_judge_options(params: dict[str, Any]) -> dict[str, Any] | None:
        value = params.get("judge_options")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(
                "llm-as-judge metric parameter 'judge_options' must be a dict "
                f"(got {type(value).__name__})"
            )
        return value

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        *,
        rubric: str,
        test_case_prompt: str,
        expected_output: str | None,
        response: str,
        scale_max: int,
    ) -> str:
        """Build the prompt handed to the judge model.

        The exact wording matches the template in Task 5.3 in
        ``tasks.md`` so that fixture-based tests can assert the
        structure of the prompt without pinning to an incidentally
        equivalent rewording.
        """
        expected_rendered = (
            expected_output if expected_output is not None else "(none provided)"
        )
        return (
            "Evaluate the following response against the rubric.\n"
            "\n"
            f"Rubric:\n{rubric}\n"
            "\n"
            f"Test case prompt:\n{test_case_prompt}\n"
            "\n"
            f"Expected output (if any):\n{expected_rendered}\n"
            "\n"
            f"Response:\n{response}\n"
            "\n"
            'Respond with a one-line score in the format: "Score: X/Y"\n'
            f"where 0 ≤ X ≤ Y ≤ {scale_max}."
        )

    # ------------------------------------------------------------------
    # Error-result helper
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(
        *,
        metric_name: str,
        threshold: float,
        message: str,
        details: dict[str, Any],
    ) -> MetricResult:
        """Shorthand for the three judge-output failure shapes.

        Keeps the three return sites (``malformed``, ``scale is zero``,
        ``score exceeds scale``) identical so the contract is obvious
        and the diagnostic fields (``judge_response`` and, when
        available, ``x`` / ``y``) land in the same places.
        """
        return MetricResult(
            name=metric_name,
            score=0.0,
            passed=False,
            threshold=threshold,
            error=message,
            details=details,
        )


def register_judge_metric() -> None:
    """Register the :class:`LlmAsJudge` singleton in the process registry.

    Idempotent: the registry has replacement semantics (documented in
    :mod:`ollama_evaluator.metrics`) so a second call simply installs a
    fresh singleton that behaves identically to the first — no
    ``AlreadyRegistered`` errors to guard against. Invoked at import
    time from the package ``__init__`` alongside
    :func:`register_builtin_metrics` so that ``from ollama_evaluator
    import metrics`` is enough to make ``llm-as-judge`` resolvable via
    :func:`get_metric`.
    """
    # Local import to avoid a circular import at module load time:
    # ``metrics/__init__.py`` imports this module to auto-register, so
    # importing ``register_metric`` at module top-level would loop.
    from . import register_metric

    register_metric(LlmAsJudge())


__all__ = [
    "LlmAsJudge",
    "register_judge_metric",
]
