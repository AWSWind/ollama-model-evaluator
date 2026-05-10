"""Property 13: llm-as-judge score normalisation.

Two complementary properties hold for the ``llm-as-judge`` metric:

1. **Well-formed branch.** For any judge output containing a line
   ``Score: X/Y`` with ``0 ≤ X ≤ Y ≤ 100`` and ``Y > 0``,
   ``llm-as-judge`` returns a :class:`MetricResult` with
   ``score == X / Y``, ``error is None``, and ``passed == (score >=
   threshold)``.

2. **Malformed branch.** For any judge output that either:

   * does not contain a ``Score:`` line at all,
   * has a non-integer numerator / denominator,
   * has ``Y == 0`` (undefined division), or
   * has ``X > Y`` (out-of-range score),

   ``llm-as-judge`` returns a :class:`MetricResult` with ``error`` set
   to a non-``None`` diagnostic string and ``passed == False``. The raw
   judge text is preserved in ``details.judge_response``.

The property is stated in ``.kiro/specs/ollama-model-evaluator/design.md``
§Correctness Properties as Property 13 and is driven by Requirement 7.2
(LLM-as-judge scoring and error handling).

Approach
--------
The metric calls into an Ollama client via ``ctx.judge_client``. These
tests inject a *fake* client that yields a fixed sequence of
:class:`GenerateChunk` objects with ``response`` equal to the drawn
judge text. No network I/O is performed, and the metric's parameter
parsing, prompt construction, regex search, and result construction
paths are all exercised end-to-end.

``asyncio.run`` is used rather than a pytest-asyncio coroutine test
because Hypothesis drives synchronous test bodies; the metric's
``score`` method is ``async`` so we wrap the invocation in
``asyncio.run`` per example.

``max_examples=20`` and ``deadline=None`` match the testing-strategy
floor set in ``design.md``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from string import ascii_letters, digits

from hypothesis import given, settings
from hypothesis import strategies as st

from ollama_evaluator.metrics.base import MetricContext
from ollama_evaluator.metrics.judge import LlmAsJudge
from ollama_evaluator.models import MetricResult
from ollama_evaluator.ollama.types import GenerateChunk, GenerateOptions
from ollama_evaluator.suites.models import MetricConfig, TestCase

# ---------------------------------------------------------------------------
# Fake Ollama client for injection via MetricContext.judge_client
# ---------------------------------------------------------------------------


class _FakeOllamaClient:
    """Minimal async Ollama client stand-in that yields a canned response.

    The real :class:`OllamaClient.generate` is an *async generator
    function* — calling it returns an async iterator directly (no
    ``await`` needed). The judge metric iterates the result with
    ``async for``:

    .. code-block:: python

        async for chunk in ctx.judge_client.generate(
            ctx.judge_model, prompt, options=options
        ):
            ...

    so this fake mirrors the async-generator shape (``async def ... yield``)
    rather than returning an iterator. It emits the canned text as one
    content chunk followed by a ``done`` final chunk, matching the real
    streaming shape (non-final chunks carry text, the final chunk is
    empty but marks completion). One-chunk and two-chunk streams are
    both valid representations of the same string because the metric
    joins chunks via ``"".join(chunk.response for ...)``.
    """

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.call_count = 0

    async def generate(  # noqa: D401 — mirrors the real client's signature.
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """Yield a canned chunk stream. ``model``/``prompt``/``options`` are ignored."""
        del model, prompt, system, options
        self.call_count += 1
        now = datetime.now(UTC)
        yield GenerateChunk(
            model="fake",
            created_at=now,
            response=self._response_text,
            done=False,
        )
        yield GenerateChunk(
            model="fake",
            created_at=now,
            response="",
            done=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_ALPHABET = ascii_letters + digits + " .,-_"


def _build_ctx(client: _FakeOllamaClient, threshold: float = 0.7) -> MetricContext:
    """Build a :class:`MetricContext` with the fake judge client attached."""
    params: dict[str, object] = {
        "rubric": "score the response",
        "threshold": threshold,
    }
    metric_config = MetricConfig(name="llm-as-judge", params=params)
    return MetricContext(
        model="llama3:8b",
        suite="s",
        test_case=TestCase(
            id="c1",
            prompt="prompt",
            expected_output="expected",
            metrics=[metric_config],
        ),
        metric_config=metric_config,
        judge_client=client,
        judge_model="judge:latest",
    )


def _run_score(judge_text: str, threshold: float = 0.7) -> MetricResult:
    """Invoke ``llm-as-judge`` with ``judge_text`` and return the result."""
    metric = LlmAsJudge()
    client = _FakeOllamaClient(judge_text)
    ctx = _build_ctx(client, threshold=threshold)
    return asyncio.run(metric.score("response-under-test", ctx))


# ---------------------------------------------------------------------------
# Well-formed branch
# ---------------------------------------------------------------------------


@st.composite
def _well_formed_judge_outputs(draw: st.DrawFn) -> tuple[str, int, int, float]:
    """Draw ``(judge_text, x, y, threshold)`` with ``0 ≤ x ≤ y ≤ 100``, ``y > 0``.

    The numerator/denominator pair drives the normalisation assertion.
    Optional leading / trailing text exercises the ``re.search``
    wrapping behaviour the metric relies on (the judge may prepend
    reasoning like ``"Reasoning: ... Score: 4/5"``) without changing
    the expected result.

    The suffix alphabet excludes digits so adjacent characters cannot
    accidentally extend the regex match (e.g. ``Score: 1/1`` followed
    by a literal ``0`` would otherwise parse as ``1/10``). The prefix
    excludes the literal ``Score:`` token so an earlier ``Score: ...``
    substring cannot hijack the ``re.search`` match.
    """
    y = draw(st.integers(min_value=1, max_value=100))
    x = draw(st.integers(min_value=0, max_value=y))
    # Non-digit alphabet for prefix/suffix: letters + punctuation only.
    non_digit_alphabet = ascii_letters + " .,-_"
    prefix = draw(
        st.text(alphabet=non_digit_alphabet, min_size=0, max_size=20).filter(
            lambda s: "Score:" not in s
        )
    )
    suffix = draw(st.text(alphabet=non_digit_alphabet, min_size=0, max_size=20))
    judge_text = f"{prefix}Score: {x}/{y}{suffix}"
    threshold = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return judge_text, x, y, threshold


@given(data=_well_formed_judge_outputs())
@settings(max_examples=20, deadline=None)
def test_well_formed_judge_output_normalises_to_x_over_y(
    data: tuple[str, int, int, float],
) -> None:
    """**Validates: Requirement 7.2**

    Well-formed ``Score: X/Y`` outputs with ``0 ≤ X ≤ Y`` and ``Y > 0``
    produce ``score == X / Y`` with no error. ``passed`` tracks the
    ``score >= threshold`` rule stated in the design document's
    §Metric framework pass/score table.
    """
    judge_text, x, y, threshold = data
    result = _run_score(judge_text, threshold=threshold)

    assert result.error is None
    assert result.threshold == threshold
    assert math.isclose(result.score, x / y)
    assert result.passed is (result.score >= threshold)
    # The raw judge text is preserved in ``details.judge_response`` so
    # downstream tooling (UI, report) can render it verbatim.
    assert result.details["judge_response"] == judge_text
    assert result.details["x"] == x
    assert result.details["y"] == y


# ---------------------------------------------------------------------------
# Malformed branch
# ---------------------------------------------------------------------------


@st.composite
def _malformed_judge_outputs(draw: st.DrawFn) -> str:
    """Draw judge text that falls into one of four malformed categories.

    Categories:

    1. **No ``Score:`` line.** Plain text without the ``Score:`` token.
       The regex search returns ``None`` and the metric records
       ``error == "malformed judge output"``.

    2. **``Y == 0``.** Well-formed ``Score:`` line but the denominator
       is zero — division would be undefined. The metric records
       ``error == "scale is zero"``.

    3. **``X > Y``.** Well-formed ``Score:`` line but the numerator
       exceeds the denominator. The metric records
       ``error == "score exceeds scale"``.

    4. **Non-integer numerator / denominator.** A ``Score:`` token
       followed by non-digit characters. ``re.search`` does not find a
       match so the metric falls through to the "malformed" error.

    Each category fires with roughly equal probability across 100
    examples; the assertion only checks the common contract
    (``error != None``, ``passed == False``), so the individual
    failure categories share a single property.
    """
    category = draw(st.sampled_from(["no-score", "y-zero", "x-gt-y", "non-integer"]))
    if category == "no-score":
        # Plain text guaranteed *not* to contain the literal "Score:" token.
        # Filter keeps the alphabet loose but rejects the rare draw that
        # accidentally contains ``Score:`` as a substring.
        text = draw(
            st.text(alphabet=_SIMPLE_ALPHABET, min_size=0, max_size=40).filter(
                lambda s: "Score:" not in s
            )
        )
        return text
    if category == "y-zero":
        x = draw(st.integers(min_value=0, max_value=100))
        return f"Score: {x}/0"
    if category == "x-gt-y":
        y = draw(st.integers(min_value=1, max_value=99))
        x = draw(st.integers(min_value=y + 1, max_value=y + 100))
        return f"Score: {x}/{y}"
    # "non-integer": the ``Score:`` token is present but followed by
    # non-digit content so the regex does not match.
    garbage = draw(
        st.sampled_from(
            [
                "Score: ??/??",
                "Score: four/five",
                "Score: x/y",
                "Score: -/-",
                "Score: NaN/NaN",
            ]
        )
    )
    return garbage


@given(judge_text=_malformed_judge_outputs())
@settings(max_examples=20, deadline=None)
def test_malformed_judge_output_sets_error_and_fails(judge_text: str) -> None:
    """**Validates: Requirement 7.2**

    Malformed judge outputs produce a :class:`MetricResult` with
    ``error != None`` and ``passed == False``. The raw judge text is
    preserved in ``details.judge_response`` so the UI and error
    summary can display it verbatim.
    """
    result = _run_score(judge_text)

    assert result.error is not None
    assert result.passed is False
    assert result.score == 0.0
    assert result.details.get("judge_response") == judge_text
