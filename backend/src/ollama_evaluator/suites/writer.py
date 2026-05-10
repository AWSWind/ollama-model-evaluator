"""Canonical serialisation of :class:`EvaluationSuite` objects.

This module is the counterpart to :mod:`ollama_evaluator.suites.loader`.
It exports a single public function, :func:`dump_suite`, which turns a
validated :class:`EvaluationSuite` back into either a YAML or JSON
string.

Canonicalisation guarantees (Requirement 4.2; Design Property 1):

* **Sorted keys.** Every mapping is rendered with its keys in
  lexicographic order, recursively. This means output is deterministic
  for a given input model (identical across Python processes,
  invocations, and library versions that preserve insertion order).
* **Two-space indent.** Mappings indent by two spaces; block sequences
  use a four-space indent with a two-space offset so the ``-`` marker
  sits two spaces past the enclosing key.
* **Block style.** YAML output never falls back to flow style for
  non-empty containers. Empty ``{}`` / ``[]`` collections are still
  emitted in flow form because that is the only representation YAML
  offers for them.
* **No comments, no YAML directives, no tags.** The writer uses
  ``ruamel.yaml.YAML(typ="safe")`` which produces a plain YAML 1.2
  document suitable for hand-editing.

Because YAML permits many byte-equivalent renderings of the same
document (quoting style, scalar folding, flow vs. block style), the
round-trip property (Requirement 4.3) is defined at the *Pydantic model*
level — ``load_suite_from_string(dump_suite(s, f), f) == s`` — not at
the byte level. The stability guarantees above are sufficient to make
``dump_suite`` itself byte-deterministic for a fixed input, but the
loader is free to re-parse any byte-equivalent rewrite of the same
document.

Design references: ``.kiro/specs/ollama-model-evaluator/design.md``
§Components §2 "Suite Loader / Writer" and §Correctness Properties,
Property 1. Requirement 4.2 is the direct driver for this module.
"""

from __future__ import annotations

import io
import json
from typing import Any, Literal

from ruamel.yaml import YAML

from .models import EvaluationSuite


def dump_suite(
    suite: EvaluationSuite, fmt: Literal["yaml", "json"]
) -> str:
    """Serialise ``suite`` into a canonical ``fmt`` string.

    Args:
        suite: A validated :class:`EvaluationSuite` instance. Any
            structure that survives :class:`EvaluationSuite`
            validation can be dumped.
        fmt: Output format selector. Must be ``"yaml"`` or ``"json"``.
            Any other value raises :class:`ValueError`.

    Returns:
        The serialised suite as a string. For YAML the trailing
        newline from ``ruamel.yaml`` is preserved; for JSON the
        output matches :func:`json.dumps` with no trailing newline.

    Raises:
        ValueError: ``fmt`` is not one of the two supported values.
    """
    if fmt == "yaml":
        return _dump_yaml(suite)
    if fmt == "json":
        return _dump_json(suite)
    raise ValueError(
        f"Unsupported suite format {fmt!r}; expected 'yaml' or 'json'"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dump_yaml(suite: EvaluationSuite) -> str:
    """Render ``suite`` as a canonical YAML document.

    Uses ``model_dump(mode="json")`` to obtain a plain-Python payload
    (so ``ruamel.yaml(typ="safe")`` can serialise it without needing
    to register custom representers), then sorts keys recursively so
    iteration order is deterministic, then dumps with block style and
    a two-space mapping indent.
    """
    data = _sort_recursive(suite.model_dump(mode="json"))
    yaml = YAML(typ="safe")
    # Force block-style output. ``default_flow_style=None`` (ruamel's
    # default for ``typ="safe"``) auto-selects between block and flow,
    # which breaks determinism for small containers.
    yaml.default_flow_style = False
    # Two-space mapping indent; the sequence indent / offset values
    # below place the ``-`` marker two spaces past the enclosing key
    # name, matching the most common hand-authored YAML layout.
    yaml.indent(mapping=2, sequence=4, offset=2)
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def _dump_json(suite: EvaluationSuite) -> str:
    """Render ``suite`` as a canonical pretty JSON document.

    ``json.dumps(..., indent=2, sort_keys=True)`` already produces a
    byte-deterministic output for any JSON-serialisable payload, so we
    rely on that directly rather than pre-sorting the dict ourselves.
    ``model_dump(mode="json")`` converts any non-primitive field (e.g.
    :class:`pathlib.Path`) to a JSON-native scalar first.
    """
    data = suite.model_dump(mode="json")
    return json.dumps(data, indent=2, sort_keys=True)


def _sort_recursive(value: Any) -> Any:
    """Return ``value`` with every nested mapping's keys sorted.

    List order is preserved (test-case order is semantically
    significant per Requirement 3.3); only mapping key order is
    normalised. Scalars pass through unchanged.
    """
    if isinstance(value, dict):
        return {k: _sort_recursive(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_recursive(item) for item in value]
    return value


__all__ = ["dump_suite"]
