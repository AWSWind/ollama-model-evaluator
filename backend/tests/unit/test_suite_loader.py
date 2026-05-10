"""Unit tests for :mod:`ollama_evaluator.suites.loader` (Task 3.1).

Covers the example-level behaviours listed in the task description:

* A valid YAML suite round-trips through ``load_suite`` and
  ``dump_suite``.
* A valid JSON suite loads successfully.
* Malformed YAML raises :class:`SuiteValidationError` with
  ``line > 0``.
* A missing required ``prompt`` yields
  ``missing_field == "test_cases.0.prompt"`` and a message matching
  ``"Field required"``.
* A duplicate ``TestCase.id`` yields a message containing the
  duplicate id verbatim.
* :func:`discover_suites` returns suites in deterministic sorted
  order and skips non-suite files (``.txt``, ``.md``).
* An unsupported extension raises :class:`SuiteValidationError` with
  a matching message.

Example-level coverage only; the Hypothesis round-trip and validation
properties live in ``tests/property/test_property_1_suite_roundtrip.py``
and ``tests/property/test_property_2_suite_validation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ollama_evaluator.suites.loader import (
    SuiteValidationError,
    discover_suites,
    load_suite,
    load_suite_from_string,
)
from ollama_evaluator.suites.writer import dump_suite

# ---------------------------------------------------------------------------
# Fixture suite payloads
# ---------------------------------------------------------------------------

# A minimally-valid suite shared by several tests. Keeping it in one
# place keeps the tests readable and localises any future schema
# changes.
_VALID_SUITE_DICT: dict[str, object] = {
    "name": "reasoning-basics",
    "version": "1.0",
    "description": "Small smoke suite",
    "defaults": {"temperature": 0.0, "max_tokens": None, "stop_sequences": []},
    "test_cases": [
        {
            "id": "c1",
            "prompt": "What is 2 + 2?",
            "expected_output": "4",
            "tags": ["math"],
            "metrics": [
                {
                    "name": "exact-match",
                    "params": {"case_sensitive": False, "trim": True},
                }
            ],
        },
        {
            "id": "c2",
            "prompt": "Name a prime number.",
            "metrics": [
                {"name": "regex-match", "params": {"pattern": r"^\d+$"}}
            ],
        },
    ],
}


def _write_minimal_suite(path: Path, name: str = "reasoning-basics") -> None:
    """Write a minimal valid suite with the given ``name`` to ``path``.

    Format is chosen from the suffix; any other suffix falls through
    to the YAML branch so tests can deliberately exercise the
    "unsupported extension" path by writing YAML text to e.g. a
    ``.txt`` file.
    """
    payload: dict[str, object] = {
        "name": name,
        "test_cases": [
            {
                "id": "c1",
                "prompt": "Hi",
                "metrics": [{"name": "exact-match"}],
            }
        ],
    }
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(
            f"name: {name}\n"
            "test_cases:\n"
            "  - id: c1\n"
            "    prompt: Hi\n"
            "    metrics:\n"
            "      - name: exact-match\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Happy path: round-trip through writer → loader
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Load a YAML/JSON suite and assert model-level equality."""

    def test_yaml_round_trip_from_tempfile(self, tmp_path: Path) -> None:
        # Build the suite in-memory so we can compare to the reloaded
        # instance directly. dump_suite emits canonical YAML; load_suite
        # should return an equal model.
        original = load_suite_from_string(json.dumps(_VALID_SUITE_DICT), "json")
        path = tmp_path / "reasoning.yaml"
        path.write_text(dump_suite(original, "yaml"), encoding="utf-8")
        reloaded = load_suite(path)
        assert reloaded == original
        assert [tc.id for tc in reloaded.test_cases] == ["c1", "c2"]

    def test_yml_extension_also_supported(self, tmp_path: Path) -> None:
        path = tmp_path / "reasoning.yml"
        _write_minimal_suite(path)
        suite = load_suite(path)
        assert suite.name == "reasoning-basics"

    def test_json_round_trip_from_tempfile(self, tmp_path: Path) -> None:
        original = load_suite_from_string(json.dumps(_VALID_SUITE_DICT), "json")
        path = tmp_path / "reasoning.json"
        path.write_text(dump_suite(original, "json"), encoding="utf-8")
        reloaded = load_suite(path)
        assert reloaded == original


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


class TestSyntaxErrors:
    def test_malformed_yaml_reports_positive_line(self, tmp_path: Path) -> None:
        # A YAML scanner error: a bare ``:`` at the start of a line is
        # a flow-entry mark without an owning mapping.
        bad_yaml = "name: s\n  : bad\n: ["
        path = tmp_path / "bad.yaml"
        path.write_text(bad_yaml, encoding="utf-8")
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite(path)
        err = excinfo.value
        assert err.path == path
        assert err.line is not None
        assert err.line > 0
        assert "YAML" in err.message

    def test_malformed_yaml_from_string_reports_positive_line(self) -> None:
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite_from_string("name: s\n  : bad\n: [", "yaml")
        err = excinfo.value
        # No path when loading from a string.
        assert err.path is None
        assert err.line is not None
        assert err.line > 0

    def test_malformed_json_reports_positive_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"name": "s", "test_cases": [}', encoding="utf-8")
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite(path)
        err = excinfo.value
        assert err.path == path
        assert err.line is not None
        assert err.line > 0


# ---------------------------------------------------------------------------
# Schema validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_missing_prompt_reports_dotted_path(self, tmp_path: Path) -> None:
        bad = {
            "name": "s",
            "test_cases": [
                {"id": "c1", "metrics": [{"name": "exact-match"}]},
            ],
        }
        path = tmp_path / "missing_prompt.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite(path)
        err = excinfo.value
        assert err.path == path
        assert err.missing_field == "test_cases.0.prompt"
        assert err.test_case_id == "c1"
        # The task spec requires the raw Pydantic ``msg`` ("Field
        # required") to flow through unchanged.
        assert "Field required" in err.message

    def test_missing_metrics_reports_dotted_path(self) -> None:
        bad = {
            "name": "s",
            "test_cases": [{"id": "c1", "prompt": "Hi"}],
        }
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite_from_string(json.dumps(bad), "json")
        err = excinfo.value
        assert err.missing_field == "test_cases.0.metrics"
        assert err.test_case_id == "c1"

    def test_missing_top_level_name(self) -> None:
        bad = {
            "test_cases": [
                {"id": "c1", "prompt": "Hi", "metrics": [{"name": "exact-match"}]}
            ]
        }
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite_from_string(json.dumps(bad), "json")
        err = excinfo.value
        assert err.missing_field == "name"
        # The top-level ``name`` field has no enclosing test case.
        assert err.test_case_id is None

    def test_duplicate_test_case_id_names_the_duplicate(
        self, tmp_path: Path
    ) -> None:
        duplicate_id = "case-dup"
        bad = {
            "name": "s",
            "test_cases": [
                {
                    "id": duplicate_id,
                    "prompt": "Hi",
                    "metrics": [{"name": "exact-match"}],
                },
                {
                    "id": duplicate_id,
                    "prompt": "Hi",
                    "metrics": [{"name": "exact-match"}],
                },
            ],
        }
        path = tmp_path / "dupes.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite(path)
        err = excinfo.value
        assert err.path == path
        # The duplicate id must appear verbatim in the human-readable
        # message so the CLI / UI can surface it (Requirement 3.5).
        assert duplicate_id in err.message


# ---------------------------------------------------------------------------
# Extension handling and format dispatch
# ---------------------------------------------------------------------------


class TestExtensionHandling:
    def test_unknown_extension_raises_with_matching_message(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "reasoning.txt"
        _write_minimal_suite(path)
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite(path)
        err = excinfo.value
        assert err.path == path
        assert err.message == "Unsupported suite file extension: .txt"

    def test_extensionless_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "no_extension"
        _write_minimal_suite(path)
        with pytest.raises(SuiteValidationError):
            load_suite(path)

    def test_missing_file_raises_not_found(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.yaml"
        with pytest.raises(SuiteValidationError) as excinfo:
            load_suite(path)
        err = excinfo.value
        assert err.path == path
        assert err.message == "Suite file not found"
        assert err.line is None

    def test_unknown_format_string_raises(self) -> None:
        with pytest.raises(SuiteValidationError):
            # The Literal type annotation does not enforce at runtime;
            # the loader must reject unknown format strings itself.
            load_suite_from_string("name: s\n", "toml")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# discover_suites
# ---------------------------------------------------------------------------


class TestDiscoverSuites:
    def test_returns_suites_in_sorted_filename_order(self, tmp_path: Path) -> None:
        # Intentionally write files in non-alphabetical order.
        _write_minimal_suite(tmp_path / "charlie.yaml", "charlie")
        _write_minimal_suite(tmp_path / "alpha.yaml", "alpha")
        _write_minimal_suite(tmp_path / "bravo.json", "bravo")
        suites = discover_suites(tmp_path)
        # ``sorted()`` on Path uses the full path string, so
        # ``alpha.yaml`` < ``bravo.json`` < ``charlie.yaml``.
        assert [s.name for s in suites] == ["alpha", "bravo", "charlie"]

    def test_ignores_non_suite_files(self, tmp_path: Path) -> None:
        _write_minimal_suite(tmp_path / "alpha.yaml", "alpha")
        (tmp_path / "README.md").write_text("# Suites directory", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("scratch", encoding="utf-8")
        (tmp_path / ".hidden.swp").write_text("editor backup", encoding="utf-8")
        suites = discover_suites(tmp_path)
        assert [s.name for s in suites] == ["alpha"]

    def test_mixed_yaml_and_json_extensions(self, tmp_path: Path) -> None:
        _write_minimal_suite(tmp_path / "a.yaml", "a")
        _write_minimal_suite(tmp_path / "b.yml", "b")
        _write_minimal_suite(tmp_path / "c.json", "c")
        suites = discover_suites(tmp_path)
        assert {s.name for s in suites} == {"a", "b", "c"}
        # Non-recursive: a nested directory must be skipped entirely.
        nested = tmp_path / "nested"
        nested.mkdir()
        _write_minimal_suite(nested / "d.yaml", "d")
        suites_after = discover_suites(tmp_path)
        assert {s.name for s in suites_after} == {"a", "b", "c"}

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert discover_suites(tmp_path) == []

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(SuiteValidationError) as excinfo:
            discover_suites(missing)
        err = excinfo.value
        assert err.path == missing
        assert err.message == "Suite directory not found"

    def test_invalid_file_propagates_suite_validation_error(
        self, tmp_path: Path
    ) -> None:
        _write_minimal_suite(tmp_path / "good.yaml", "good")
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SuiteValidationError) as excinfo:
            discover_suites(tmp_path)
        err = excinfo.value
        assert err.path is not None
        assert err.path.name == "bad.json"
