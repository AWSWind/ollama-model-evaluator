"""Built-in scoring metrics (Task 5.2, 5.4).

This module implements the five string/structural metrics listed in the
design document's §Metric framework pass/score table plus the HumanEval
v1 capture/reserved pair:

* :class:`ExactMatch` — ``exact-match``
* :class:`RegexMatch` — ``regex-match``
* :class:`Contains` — ``contains``
* :class:`JsonSchemaValid` — ``json-schema-valid``
* :class:`LengthRange` — ``length-range``
* :class:`ResponseCapture` — ``response-capture`` (HumanEval v1 default)
* :class:`HumanevalExecReserved` — ``humaneval-exec`` (reserved name, v2+)

Each metric is a small class satisfying the :class:`Metric` protocol
defined in :mod:`ollama_evaluator.metrics.base`. The metric
implementations are intentionally stateless: all per-invocation
configuration is pulled from :attr:`MetricContext.metric_config.params`
on every call. This keeps a single module-level singleton reusable
across every registered suite and lets tests install, inspect, and
replace metrics through the public registry without thinking about
instance state.

Parameter-validation contract (Requirement 7.5 and the sibling Task 5.3
``llm-as-judge`` implementation)
--------------------------------------------------------------------

The metric implementations in this module **raise** ``ValueError`` on
missing or malformed required parameters (for example, ``pattern`` on
``regex-match`` or ``schema`` on ``json-schema-valid``). That is by
design: the runner wraps every metric call in ``try/except`` and
converts the raised exception into a :class:`MetricResult` with
``error`` populated and ``passed=False``. Raising here keeps the metric
code path small and uniform — the runner owns the "never crash the Run
because of one broken metric" policy, not each metric.

Metrics do *not* raise on malformed *response* content. A response that
is not JSON, a response that fails a schema, a response that is too
short — these are all valid inputs that produce ``score=0.0``,
``passed=False`` with a diagnostic ``details`` field. The distinction is
"the metric was mis-configured" (raise) vs. "the response didn't score
well" (return a failing result).

Registration
-----------

:func:`register_builtin_metrics` installs one singleton of each class
into the process-global registry under the names shown above. It is
idempotent: calling it twice replaces the prior singletons with fresh
ones (the registry already has replacement semantics — see the
``metrics/__init__.py`` docstring) without raising. The package's
``__init__.py`` calls it at import time so that ``from
ollama_evaluator import metrics`` is enough to make every built-in
metric resolvable via :func:`get_metric`.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from ..models import MetricResult
from .base import MetricContext

# ---------------------------------------------------------------------------
# Small parameter-parsing helpers
# ---------------------------------------------------------------------------


def _require_param(
    params: dict[str, Any], key: str, expected_type: type | tuple[type, ...]
) -> Any:
    """Pull ``key`` from ``params`` and check its type.

    Raises ``ValueError`` with a message naming the metric parameter
    that was missing or wrongly typed. The runner turns these raises
    into a :class:`MetricResult` with ``error`` populated, so the user
    sees the bad parameter name verbatim rather than a generic
    ``KeyError``.
    """
    if key not in params:
        raise ValueError(f"metric parameter {key!r} is required")
    value = params[key]
    if not isinstance(value, expected_type):
        # Mention the expected type(s) so the message is self-contained.
        type_names = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise ValueError(
            f"metric parameter {key!r} must be of type {type_names} "
            f"(got {type(value).__name__})"
        )
    return value


def _optional_param(
    params: dict[str, Any],
    key: str,
    expected_type: type | tuple[type, ...],
    default: Any,
) -> Any:
    """Pull an optional parameter, falling back to ``default``.

    ``None`` is returned verbatim when stored; callers that want to
    reject ``None`` should use :func:`_require_param`. Type checking is
    skipped when the value is absent or explicitly ``None`` and the
    default allows it.
    """
    if key not in params:
        return default
    value = params[key]
    # Allow explicit None for optional params that accept it.
    if value is None:
        return None
    if not isinstance(value, expected_type):
        type_names = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise ValueError(
            f"metric parameter {key!r} must be of type {type_names} "
            f"(got {type(value).__name__})"
        )
    return value


# ---------------------------------------------------------------------------
# exact-match
# ---------------------------------------------------------------------------


class ExactMatch:
    """Score ``response`` as ``1.0`` iff it equals ``test_case.expected_output``.

    Parameters (from ``MetricConfig.params``):

    * ``case_sensitive`` (``bool``, default ``True``): when ``False``,
      comparison is done after lowercasing both strings.
    * ``trim`` (``bool``, default ``True``): when ``True``, strip
      surrounding whitespace from both strings before comparing. The
      default is ``True`` because Ollama responses frequently include
      a trailing newline that users almost never mean to treat as part
      of the answer; users who explicitly care about whitespace can
      turn trimming off.

    ``expected_output`` comes from the :class:`TestCase`, not from
    ``params``. Missing ``expected_output`` is a Test_Case authoring
    error and surfaces as a ``ValueError`` at score time — see the
    module docstring for the policy.
    """

    name = "exact-match"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        params = ctx.metric_config.params
        case_sensitive = _optional_param(params, "case_sensitive", bool, True)
        trim = _optional_param(params, "trim", bool, True)

        expected = ctx.test_case.expected_output
        if expected is None:
            raise ValueError(
                "exact-match metric requires test_case.expected_output to be set"
            )

        actual_cmp = response
        expected_cmp = expected
        if trim:
            actual_cmp = actual_cmp.strip()
            expected_cmp = expected_cmp.strip()
        if not case_sensitive:
            actual_cmp = actual_cmp.lower()
            expected_cmp = expected_cmp.lower()

        matched = actual_cmp == expected_cmp
        score = 1.0 if matched else 0.0
        return MetricResult(
            name=ctx.metric_config.name,
            score=score,
            passed=score == 1.0,
            threshold=1.0,
            details={
                "matched": matched,
                "case_sensitive": case_sensitive,
                "trim": trim,
                "expected": expected,
                "actual": response,
            },
        )


# ---------------------------------------------------------------------------
# regex-match
# ---------------------------------------------------------------------------


_FLAG_MAP: dict[str, int] = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
}


def _parse_regex_flags(flags_spec: str) -> int:
    """Translate a short flag spec (``"i"``, ``"ims"``, ``""``) into :mod:`re` flags.

    Only ``i``, ``m``, and ``s`` are supported per the design document.
    Unknown characters raise ``ValueError`` so users get immediate
    feedback on typos rather than a silently disabled flag.
    """
    combined = 0
    for ch in flags_spec:
        if ch not in _FLAG_MAP:
            raise ValueError(
                f"regex-match flag {ch!r} is not supported (allowed: i, m, s)"
            )
        combined |= _FLAG_MAP[ch]
    return combined


class RegexMatch:
    """Score ``response`` as ``1.0`` iff ``re.search(pattern, response)`` matches.

    Parameters (from ``MetricConfig.params``):

    * ``pattern`` (``str``, required): a Python regex passed directly to
      :func:`re.search`. ``re.search`` (not ``re.match`` or
      ``re.fullmatch``) is used intentionally so that the metric finds
      an answer anywhere in the response, which matches how the
      benchmark adapters extract letters / numbers from free-form
      model output.
    * ``flags`` (``str``, default ``""``): combination of the characters
      ``i`` (:data:`re.IGNORECASE`), ``m`` (:data:`re.MULTILINE`), and
      ``s`` (:data:`re.DOTALL`).

    Any :class:`re.error` from :func:`re.compile` propagates as a
    :class:`ValueError` so the runner records a metric error rather than
    crashing the whole Run.
    """

    name = "regex-match"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        params = ctx.metric_config.params
        pattern_str = _require_param(params, "pattern", str)
        flags_spec = _optional_param(params, "flags", str, "")

        flags = _parse_regex_flags(flags_spec)
        try:
            pattern = re.compile(pattern_str, flags)
        except re.error as exc:
            raise ValueError(
                f"regex-match pattern {pattern_str!r} is invalid: {exc}"
            ) from exc

        match = pattern.search(response)
        if match is None:
            return MetricResult(
                name=ctx.metric_config.name,
                score=0.0,
                passed=False,
                threshold=1.0,
                details={
                    "matched": False,
                    "pattern": pattern_str,
                    "flags": flags_spec,
                },
            )

        return MetricResult(
            name=ctx.metric_config.name,
            score=1.0,
            passed=True,
            threshold=1.0,
            details={
                "matched": True,
                "pattern": pattern_str,
                "flags": flags_spec,
                "match": match.group(0),
                # ``groups()`` is a tuple; cast to list so the JSON
                # serialiser in ``MetricResult.details`` emits a stable
                # array representation.
                "groups": list(match.groups()),
            },
        )


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


class Contains:
    """Score ``response`` by the fraction of ``substrings`` it contains.

    Parameters (from ``MetricConfig.params``):

    * ``substrings`` (``list[str]``, required, non-empty): the list of
      substrings to search for. An empty list is rejected because the
      fraction would be undefined.
    * ``mode`` (``"any" | "all"``, default ``"all"``): the semantic
      label attached to ``threshold``'s default and the way the result
      is rendered in diagnostics. ``mode`` does *not* change how
      ``score`` is computed — score is always the fraction of
      substrings that matched. What ``mode`` changes is the default
      threshold: ``"all"`` defaults to ``1.0`` (every substring must
      match), ``"any"`` defaults to "any single match is enough" by
      clamping to a small positive epsilon in the threshold so the
      ``passed = score >= threshold`` check fires on one match.
    * ``threshold`` (``float``, default ``1.0`` in ``"all"`` mode,
      ``epsilon`` in ``"any"`` mode): the threshold compared against
      the computed fraction.

    Case-sensitivity is not part of the spec; matches are case-sensitive
    Python ``in`` checks.
    """

    name = "contains"

    # Any-mode fires on a single match in a list of up to 1e6 substrings;
    # a small positive threshold is easier to reason about than ``score > 0``.
    _ANY_EPSILON = 1e-9

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        params = ctx.metric_config.params
        substrings = _require_param(params, "substrings", list)
        if len(substrings) == 0:
            raise ValueError("contains metric requires a non-empty 'substrings' list")
        # Defer element-type validation to a loop so the error identifies
        # the offending index.
        for idx, s in enumerate(substrings):
            if not isinstance(s, str):
                raise ValueError(
                    f"contains metric 'substrings[{idx}]' must be a str "
                    f"(got {type(s).__name__})"
                )

        mode = _optional_param(params, "mode", str, "all")
        if mode not in ("any", "all"):
            raise ValueError(
                f"contains metric 'mode' must be 'any' or 'all' (got {mode!r})"
            )

        default_threshold = 1.0 if mode == "all" else self._ANY_EPSILON
        threshold = _optional_param(
            params, "threshold", (int, float), default_threshold
        )
        # Reject NaN/inf explicitly: they make the ``>=`` comparison useless.
        if not isinstance(threshold, (int, float)) or threshold != threshold:
            raise ValueError(
                "contains metric 'threshold' must be a real number "
                f"(got {threshold!r})"
            )

        matches = [sub for sub in substrings if sub in response]
        fraction = len(matches) / len(substrings)
        passed = fraction >= float(threshold)

        return MetricResult(
            name=ctx.metric_config.name,
            score=fraction,
            passed=passed,
            threshold=float(threshold),
            details={
                "mode": mode,
                "substrings": list(substrings),
                "matched": matches,
                "matched_count": len(matches),
                "total": len(substrings),
            },
        )


# ---------------------------------------------------------------------------
# json-schema-valid
# ---------------------------------------------------------------------------


class JsonSchemaValid:
    """Score ``response`` as ``1.0`` iff it parses as JSON and validates against ``schema``.

    Parameters (from ``MetricConfig.params``):

    * ``schema`` (``dict``, required): a JSON Schema document handed to
      :func:`jsonschema.validate`.

    Behaviour:

    * If :func:`json.loads` raises, ``score=0.0``, ``passed=False``, and
      ``details["parse_error"]`` carries the exception message. The
      metric does not distinguish "not JSON at all" from "syntactically
      invalid JSON" — either is a failure.
    * If the parsed JSON does not validate, ``score=0.0``,
      ``passed=False``, and ``details["validation_error"]`` carries the
      jsonschema error message. ``details["parsed"]`` is included so
      the UI can render the actual object that failed.
    * An invalid ``schema`` itself (``SchemaError``) is a metric
      configuration bug, not a response failure, so it is re-raised as
      a :class:`ValueError` and picked up by the runner.
    """

    name = "json-schema-valid"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        params = ctx.metric_config.params
        schema = _require_param(params, "schema", dict)

        try:
            parsed = json.loads(response.strip())
        except json.JSONDecodeError as exc:
            return MetricResult(
                name=ctx.metric_config.name,
                score=0.0,
                passed=False,
                threshold=1.0,
                details={"parse_error": str(exc)},
            )

        try:
            jsonschema.validate(instance=parsed, schema=schema)
        except JsonSchemaValidationError as exc:
            return MetricResult(
                name=ctx.metric_config.name,
                score=0.0,
                passed=False,
                threshold=1.0,
                details={
                    "validation_error": exc.message,
                    "parsed": parsed,
                },
            )
        except JsonSchemaSchemaError as exc:
            # The *schema* itself is invalid — that is a metric
            # configuration bug, not a response failure. Surface it as
            # a ValueError so the runner records a metric error.
            raise ValueError(
                f"json-schema-valid received an invalid schema: {exc.message}"
            ) from exc

        return MetricResult(
            name=ctx.metric_config.name,
            score=1.0,
            passed=True,
            threshold=1.0,
            details={"parsed": parsed},
        )


# ---------------------------------------------------------------------------
# length-range
# ---------------------------------------------------------------------------


class LengthRange:
    """Score ``response`` as ``1.0`` iff its length is within ``[min, max]``.

    Parameters (from ``MetricConfig.params``):

    * ``min`` (``int | None``, default ``None``): inclusive lower bound.
      ``None`` means "no lower bound".
    * ``max`` (``int | None``, default ``None``): inclusive upper bound.
      ``None`` means "no upper bound".
    * ``unit`` (``"chars" | "tokens"``, default ``"chars"``): whether
      length is measured in characters (``len(response)``) or
      whitespace-delimited tokens (``len(response.split())``). The
      tokens unit is explicitly an **approximation** — there is no
      tokenizer dependency in v1, so the definition is "whatever
      ``str.split()`` returns". Downstream consumers that need
      model-specific tokenisation should implement their own metric.

    At least one of ``min`` / ``max`` must be set; ``(None, None)`` is
    rejected as it would produce ``1.0`` unconditionally.
    """

    name = "length-range"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        params = ctx.metric_config.params
        min_bound = _optional_param(params, "min", int, None)
        max_bound = _optional_param(params, "max", int, None)
        unit = _optional_param(params, "unit", str, "chars")

        if unit not in ("chars", "tokens"):
            raise ValueError(
                f"length-range 'unit' must be 'chars' or 'tokens' (got {unit!r})"
            )
        if min_bound is None and max_bound is None:
            raise ValueError(
                "length-range requires at least one of 'min' or 'max' to be set"
            )
        if min_bound is not None and min_bound < 0:
            raise ValueError(
                f"length-range 'min' must be >= 0 (got {min_bound})"
            )
        if max_bound is not None and max_bound < 0:
            raise ValueError(
                f"length-range 'max' must be >= 0 (got {max_bound})"
            )
        if (
            min_bound is not None
            and max_bound is not None
            and min_bound > max_bound
        ):
            raise ValueError(
                f"length-range 'min' ({min_bound}) must be <= 'max' ({max_bound})"
            )

        # Whitespace split approximates tokens without pulling in a
        # tokenizer dependency; see the class docstring.
        length = len(response) if unit == "chars" else len(response.split())

        in_range = True
        if min_bound is not None and length < min_bound:
            in_range = False
        if max_bound is not None and length > max_bound:
            in_range = False

        score = 1.0 if in_range else 0.0
        return MetricResult(
            name=ctx.metric_config.name,
            score=score,
            passed=score == 1.0,
            threshold=1.0,
            details={
                "unit": unit,
                "length": length,
                "min": min_bound,
                "max": max_bound,
                "in_range": in_range,
            },
        )


# ---------------------------------------------------------------------------
# response-capture  (HumanEval v1 default; see design.md §HumanEval execution
# mode and Requirement 17.9)
# ---------------------------------------------------------------------------


class ResponseCapture:
    """Record the raw model response without scoring it.

    Score table entry (design.md §Metric framework):

    ================= =====================
    Inputs            —
    Score             always ``0.0``
    Pass condition    always ``True``
    ================= =====================

    Used by the HumanEval v1 adapter per Requirement 17.9: HumanEval's
    canonical metric (``pass@k``) requires executing untrusted model
    output inside a sandbox, which is deferred to v2. Until then the
    adapter uses this metric to retain every response verbatim in
    ``MetricResult.details.response`` so an external grader can score
    the Run later. The fixed ``score=0.0`` means "no score reported";
    ``passed=True`` keeps the Run from classifying the Test_Case as a
    failure just because scoring was deferred.

    The metric takes no parameters and never raises — it is safe to
    attach to any Test_Case regardless of configuration.
    """

    name = "response-capture"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        return MetricResult(
            name=ctx.metric_config.name,
            score=0.0,
            passed=True,
            threshold=None,
            details={"response": response},
        )


# ---------------------------------------------------------------------------
# humaneval-exec  (reserved name; see design.md §HumanEval execution mode and
# Requirement 17.9)
# ---------------------------------------------------------------------------


class HumanevalExecReserved:
    """Placeholder for the post-v1 sandboxed HumanEval execution metric.

    The design document reserves the ``humaneval-exec`` metric name so
    a future version can add sandboxed-execution ``pass@1`` scoring
    without a configuration-schema break (Requirement 17.9). Until
    that lands, the metric is registered but any attempt to score with
    it raises :class:`NotImplementedError`.

    Registering a raising stub — rather than leaving the name
    unregistered — is load-bearing: if the name were unregistered, the
    suite loader would reject Test_Cases that reference it at load
    time, defeating the "reserve the name" intent. Registering it and
    raising at ``score()`` time means a Test_Case can *declare* the
    metric and the runner's per-metric ``try/except`` wrapper
    (Requirement 7.5) will convert the raise into a
    :class:`MetricResult` with ``error`` populated, so the Run still
    completes.
    """

    name = "humaneval-exec"

    async def score(self, response: str, ctx: MetricContext) -> MetricResult:
        raise NotImplementedError(
            "humaneval-exec is reserved for future sandboxed-execution "
            "implementation (v1 uses response-capture)"
        )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_builtin_metrics() -> None:
    """Register one singleton of each built-in metric in the process registry.

    Idempotent: calling this twice is safe because the registry has
    replacement semantics (documented in
    :mod:`ollama_evaluator.metrics`). The second call installs fresh
    singletons that behave identically to the first — no
    ``AlreadyRegistered`` errors to guard against.

    Invoked at import time from the package ``__init__`` so that
    ``from ollama_evaluator import metrics`` is enough to make every
    built-in metric resolvable via :func:`get_metric`. Tests that clear
    the registry (e.g. via a ``clean_registry`` fixture) can call this
    function again to repopulate it.
    """
    # Local import to avoid a circular import at module load time:
    # ``metrics/__init__.py`` imports this module to auto-register, so
    # importing ``register_metric`` at module top-level would loop.
    from . import register_metric

    register_metric(ExactMatch())
    register_metric(RegexMatch())
    register_metric(Contains())
    register_metric(JsonSchemaValid())
    register_metric(LengthRange())
    register_metric(ResponseCapture())
    register_metric(HumanevalExecReserved())


__all__ = [
    "Contains",
    "ExactMatch",
    "HumanevalExecReserved",
    "JsonSchemaValid",
    "LengthRange",
    "RegexMatch",
    "ResponseCapture",
    "register_builtin_metrics",
]
