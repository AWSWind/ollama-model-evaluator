"""Unit tests for :mod:`ollama_evaluator.suites.writer` (Task 3.2).

Covers the example-level behaviours listed in the task description:

* :func:`dump_suite` with ``fmt="yaml"`` emits block-style YAML with
  sorted keys and a trailing newline.
* :func:`dump_suite` with ``fmt="json"`` emits a pretty JSON document
  with sorted keys and a 2-space indent.
* The same input produces byte-identical output across multiple calls
  (determinism).
* The output is loadable by :func:`load_suite_from_string` and yields
  a model equal to the original suite (Property 1 round-trip at the
  example level; the formal property test lives in Task 3.3).
* An unknown ``fmt`` raises :class:`ValueError`.

The tests intentionally avoid Hypothesis so they stay fast and
debuggable. The generative round-trip invariant is covered by
``tests/property/test_property_1_suite_roundtrip.py`` in Task 3.3.
"""

from __future__ import annotations

import json
import re

import pytest

from ollama_evaluator.suites import (
    dump_suite,
    load_suite_from_string,
)
from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)

# ---------------------------------------------------------------------------
# Fixture suites
# ---------------------------------------------------------------------------


def _minimal_suite() -> EvaluationSuite:
    """Smallest legal suite: one case, one metric, default everything."""
    return EvaluationSuite(
        name="smoke",
        test_cases=[
            TestCase(
                id="c1",
                prompt="Hello world",
                metrics=[MetricConfig(name="exact-match")],
            )
        ],
    )


def _rich_suite() -> EvaluationSuite:
    """A suite exercising every optional field.

    Designed so the round-trip assertion exercises all serialised
    shapes: optional strings, optional lists, explicit ``[]`` vs. the
    ``None`` sentinel for ``stop_sequences``, nested reference data,
    multiple metrics per case, multiple tags, per-case generation
    overrides, and a non-default :class:`GenerationDefaults`.
    """
    return EvaluationSuite(
        name="reasoning-basics",
        version="1.2",
        description="Smoke suite covering optional fields.",
        defaults=GenerationDefaults(
            temperature=0.2,
            max_tokens=512,
            stop_sequences=["\n\n", "###"],
        ),
        test_cases=[
            TestCase(
                id="c1",
                prompt="What is 2 + 2?",
                system_prompt="You are a careful arithmetic tutor.",
                expected_output="4",
                reference_data={"topic": "arithmetic", "difficulty": 1},
                tags=["math", "easy"],
                temperature=0.0,
                max_tokens=32,
                stop_sequences=[],  # explicit [] vs the None sentinel
                metrics=[
                    MetricConfig(
                        name="exact-match",
                        params={"case_sensitive": False, "trim": True},
                    ),
                    MetricConfig(
                        name="regex-match",
                        params={"pattern": r"^\s*4\s*$", "flags": "i"},
                    ),
                ],
            ),
            TestCase(
                id="c2",
                prompt="Name a prime number.",
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={"pattern": r"^\d+$"},
                    )
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# YAML output shape
# ---------------------------------------------------------------------------


class TestDumpYaml:
    def test_returns_string_with_trailing_newline(self) -> None:
        out = dump_suite(_minimal_suite(), "yaml")
        assert isinstance(out, str)
        assert out.endswith("\n")

    def test_is_deterministic_across_calls(self) -> None:
        """Same input → byte-identical output on every invocation."""
        suite = _rich_suite()
        outs = [dump_suite(suite, "yaml") for _ in range(5)]
        assert len(set(outs)) == 1, "YAML output is not deterministic"

    def test_mapping_keys_are_sorted(self) -> None:
        """Top-level mapping keys appear in sorted order."""
        out = dump_suite(_rich_suite(), "yaml")
        # Collect the first token on each line that looks like a
        # top-level key (zero indent, ends with ``:``). Sorted order is
        # the whole point of the canonicalisation rule.
        top_keys = re.findall(r"^([A-Za-z_][A-Za-z0-9_]*):", out, flags=re.MULTILINE)
        assert top_keys == sorted(top_keys)
        # Sanity check: every top-level model field is present.
        assert set(top_keys) >= {
            "defaults",
            "description",
            "name",
            "test_cases",
            "version",
        }

    def test_uses_two_space_indent_and_block_style(self) -> None:
        out = dump_suite(_rich_suite(), "yaml")
        # Test cases are rendered as a block sequence under
        # ``test_cases:``. The ``-`` marker lives two spaces in; the
        # keys beneath it live four spaces in (2 indent + 2 offset for
        # ruamel's sequence layout).
        assert "test_cases:\n  - " in out
        # No flow-style mapping for the test_cases value (``{`` only
        # appears inside string scalars, never as the mapping opener
        # for a populated container).
        assert "test_cases: {" not in out
        assert "test_cases: [" not in out

    def test_no_yaml_directives_or_tags(self) -> None:
        """No ``%YAML`` directive, no ``---`` doc marker, no ``!`` tags."""
        out = dump_suite(_rich_suite(), "yaml")
        assert not out.lstrip().startswith("%YAML")
        assert not out.lstrip().startswith("---")
        # Ruamel in safe mode never emits ``!!python/...`` tags; guard
        # against that explicitly so any future regression is caught.
        assert "!!" not in out

    def test_unknown_fmt_raises(self) -> None:
        with pytest.raises(ValueError, match="yaml.*json|Unsupported suite format"):
            dump_suite(_minimal_suite(), "toml")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JSON output shape
# ---------------------------------------------------------------------------


class TestDumpJson:
    def test_returns_pretty_json_string(self) -> None:
        out = dump_suite(_minimal_suite(), "json")
        assert isinstance(out, str)
        # ``json.dumps(..., indent=2)`` does not append a trailing
        # newline. Codify that behaviour here so downstream tooling
        # knows what to expect.
        assert not out.endswith("\n")
        # Must be parseable as JSON.
        parsed = json.loads(out)
        assert parsed["name"] == "smoke"

    def test_is_deterministic_across_calls(self) -> None:
        suite = _rich_suite()
        outs = [dump_suite(suite, "json") for _ in range(5)]
        assert len(set(outs)) == 1, "JSON output is not deterministic"

    def test_mapping_keys_are_sorted(self) -> None:
        """``sort_keys=True`` applies recursively in json.dumps."""
        out = dump_suite(_rich_suite(), "json")
        parsed = json.loads(out)
        # Top-level keys appear in sorted order in the raw text.
        top_keys = [
            line.split('"')[1]
            for line in out.splitlines()
            if line.startswith('  "')
        ]
        assert top_keys == sorted(top_keys)
        # And semantically the payload covers every expected field.
        assert set(parsed.keys()) >= {
            "defaults",
            "description",
            "name",
            "test_cases",
            "version",
        }

    def test_uses_two_space_indent(self) -> None:
        out = dump_suite(_rich_suite(), "json")
        # The second line of a pretty-printed JSON object is the first
        # key at exactly two spaces of indent.
        second_line = out.splitlines()[1]
        assert second_line.startswith('  "')
        assert not second_line.startswith('    "')

    def test_unknown_fmt_raises(self) -> None:
        with pytest.raises(ValueError):
            dump_suite(_minimal_suite(), "xml")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip (example-level; property test lives in Task 3.3)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_yaml_minimal_suite_round_trips(self) -> None:
        suite = _minimal_suite()
        assert load_suite_from_string(dump_suite(suite, "yaml"), "yaml") == suite

    def test_json_minimal_suite_round_trips(self) -> None:
        suite = _minimal_suite()
        assert load_suite_from_string(dump_suite(suite, "json"), "json") == suite

    def test_yaml_rich_suite_round_trips(self) -> None:
        suite = _rich_suite()
        assert load_suite_from_string(dump_suite(suite, "yaml"), "yaml") == suite

    def test_json_rich_suite_round_trips(self) -> None:
        suite = _rich_suite()
        assert load_suite_from_string(dump_suite(suite, "json"), "json") == suite

    def test_round_trip_preserves_test_case_order(self) -> None:
        """Test-case list order is semantically significant."""
        suite = _rich_suite()
        reloaded_yaml = load_suite_from_string(dump_suite(suite, "yaml"), "yaml")
        reloaded_json = load_suite_from_string(dump_suite(suite, "json"), "json")
        expected = [tc.id for tc in suite.test_cases]
        assert [tc.id for tc in reloaded_yaml.test_cases] == expected
        assert [tc.id for tc in reloaded_json.test_cases] == expected

    def test_round_trip_preserves_stop_sequences_none_vs_empty(self) -> None:
        """``None`` and ``[]`` on ``stop_sequences`` mean different things."""
        suite = _rich_suite()
        # c1 explicitly sets stop_sequences=[] (override → no stops).
        # c2 leaves it as None (inherit from defaults).
        assert suite.test_cases[0].stop_sequences == []
        assert suite.test_cases[1].stop_sequences is None
        for fmt in ("yaml", "json"):
            reloaded = load_suite_from_string(dump_suite(suite, fmt), fmt)  # type: ignore[arg-type]
            assert reloaded.test_cases[0].stop_sequences == []
            assert reloaded.test_cases[1].stop_sequences is None
