"""Load Evaluation_Suite files from disk or in-memory strings.

This module implements the user-facing entry points for reading suite
files into strongly-typed :class:`EvaluationSuite` objects:

* :func:`load_suite` — read and parse a file on disk, dispatching on
  extension (``.yaml``/``.yml`` → YAML, ``.json`` → JSON).
* :func:`load_suite_from_string` — parse an in-memory string given an
  explicit ``"yaml"`` or ``"json"`` format.
* :func:`discover_suites` — non-recursive directory scan that loads
  every ``.yaml``/``.yml``/``.json`` file in deterministic filename
  order.
* :class:`SuiteValidationError` — a single error type raised for all
  failure modes (missing file, unknown extension, malformed syntax,
  Pydantic validation failure). Carries the file path, the offending
  test-case id, the dotted-path name of the offending field, a
  human-readable message, and a 1-based line number when available.

YAML parsing goes through ``ruamel.yaml`` in round-trip mode so the
parser attaches ``problem_mark`` metadata to :class:`ScannerError` /
:class:`ParserError` instances — that is the source of the 1-based
line for YAML syntax errors. JSON uses the standard library's
:mod:`json` module, which exposes a line number only for syntax errors
via :attr:`json.JSONDecodeError.lineno`.

Design references: ``.kiro/specs/ollama-model-evaluator/design.md``
§Components §2 "Suite Loader / Writer"; Requirements 3.1, 3.2, 3.5,
4.1, 4.4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.parser import ParserError
from ruamel.yaml.scanner import ScannerError

from .models import EvaluationSuite

# File extensions accepted by :func:`load_suite` and :func:`discover_suites`.
# Stored here rather than inline so both functions stay in lockstep.
_YAML_EXTS: frozenset[str] = frozenset({".yaml", ".yml"})
_JSON_EXTS: frozenset[str] = frozenset({".json"})
_SUPPORTED_EXTS: frozenset[str] = _YAML_EXTS | _JSON_EXTS


class SuiteValidationError(Exception):
    """Raised for any failure while loading an Evaluation_Suite file.

    The constructor exposes all five fields required by Requirement
    3.5 (``path``, ``test_case_id``, ``missing_field``, ``message``)
    plus Requirement 4.4 (``line``). Any of the first three may be
    ``None`` depending on the failure mode — for example a missing
    file has no offending test case, and a JSON validation error has
    no line number.

    Attributes:
        path: Path of the file that failed to load, or ``None`` when
            the caller supplied an in-memory string.
        test_case_id: ``TestCase.id`` of the offending test case, when
            the Pydantic error path traverses ``test_cases[i]`` and
            the id is present in the raw document.
        missing_field: Dotted path of the offending field joined from
            the first Pydantic error's ``loc`` (for example
            ``"test_cases.0.prompt"``). ``None`` for non-field errors
            (syntax errors, model-level validators, file-not-found).
        message: Human-readable description of the error. For
            schema-level failures this is the raw Pydantic ``msg`` so
            downstream callers can render it verbatim.
        line: 1-based line number of the offending element when the
            underlying parser can supply one (YAML/JSON syntax
            errors). ``None`` otherwise.
    """

    def __init__(
        self,
        path: Path | None,
        test_case_id: str | None,
        missing_field: str | None,
        message: str,
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.test_case_id = test_case_id
        self.missing_field = missing_field
        self.message = message
        self.line = line

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def load_suite(path: Path) -> EvaluationSuite:
    """Load and validate an Evaluation_Suite from ``path``.

    The file format is chosen from the path's extension: ``.yaml`` or
    ``.yml`` is parsed as YAML via ``ruamel.yaml``; ``.json`` is parsed
    via :mod:`json`. Any other extension raises
    :class:`SuiteValidationError`. All errors carry ``path`` so users
    can locate the offending file from the error alone.

    Args:
        path: Filesystem path to the suite file.

    Returns:
        A validated :class:`EvaluationSuite` instance.

    Raises:
        SuiteValidationError: The extension is unsupported, the file
            cannot be read, the file contains invalid syntax, or the
            parsed document fails Pydantic validation.
    """
    ext = path.suffix.lower()
    if ext in _YAML_EXTS:
        fmt: Literal["yaml", "json"] = "yaml"
    elif ext in _JSON_EXTS:
        fmt = "json"
    else:
        raise SuiteValidationError(
            path=path,
            test_case_id=None,
            missing_field=None,
            message=f"Unsupported suite file extension: {path.suffix}",
            line=None,
        )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SuiteValidationError(
            path=path,
            test_case_id=None,
            missing_field=None,
            message="Suite file not found",
            line=None,
        ) from exc
    except OSError as exc:
        raise SuiteValidationError(
            path=path,
            test_case_id=None,
            missing_field=None,
            message=f"Failed to read suite file: {exc}",
            line=None,
        ) from exc
    try:
        return load_suite_from_string(text, fmt)
    except SuiteValidationError as exc:
        # Re-raise the same structured error with the file path
        # attached so callers always see where the failure originated.
        raise SuiteValidationError(
            path=path,
            test_case_id=exc.test_case_id,
            missing_field=exc.missing_field,
            message=exc.message,
            line=exc.line,
        ) from exc.__cause__ or exc


def load_suite_from_string(
    text: str, fmt: Literal["yaml", "json"]
) -> EvaluationSuite:
    """Parse and validate an Evaluation_Suite from an in-memory string.

    No file I/O is performed; the ``path`` field on any raised
    :class:`SuiteValidationError` is ``None``. Used by CLI
    subcommands, adapter tests, and :func:`load_suite` itself.

    Args:
        text: The YAML or JSON document to parse.
        fmt: Explicit format selector; must be ``"yaml"`` or
            ``"json"``. The loader does **not** infer the format from
            content because YAML is a superset of JSON and such
            inference would silently mask format confusions.

    Returns:
        A validated :class:`EvaluationSuite` instance.

    Raises:
        SuiteValidationError: The string contains invalid syntax or
            fails Pydantic validation, or ``fmt`` is not one of the
            supported values. Syntax errors carry a 1-based ``line``;
            validation errors leave ``line`` as ``None``.
    """
    if fmt == "yaml":
        data = _parse_yaml(text)
    elif fmt == "json":
        data = _parse_json(text)
    else:
        raise SuiteValidationError(
            path=None,
            test_case_id=None,
            missing_field=None,
            message=f"Unsupported suite format: {fmt!r}; expected 'yaml' or 'json'",
            line=None,
        )

    try:
        return EvaluationSuite.model_validate(data)
    except ValidationError as exc:
        raise _wrap_validation_error(exc, data=data) from exc


def discover_suites(dir: Path) -> list[EvaluationSuite]:  # noqa: A002 - design API
    """Load every suite file in ``dir`` in deterministic filename order.

    The scan is **non-recursive**. Files whose extension is not one of
    ``.yaml``, ``.yml``, or ``.json`` are silently ignored so the
    suites directory may coexist with README files, editor backups,
    and similar siblings (Requirement 3.1).

    Args:
        dir: Directory to scan. Must exist and be readable.

    Returns:
        A list of :class:`EvaluationSuite` instances sorted by the
        case-sensitive filename of the source file. Order is stable
        across runs so callers may rely on it for reproducibility.

    Raises:
        SuiteValidationError: ``dir`` is not an existing directory,
            or any individual suite file fails to load. The loader is
            fail-fast: the first failure propagates and no further
            files are loaded.
    """
    if not dir.is_dir():
        raise SuiteValidationError(
            path=dir,
            test_case_id=None,
            missing_field=None,
            message="Suite directory not found",
            line=None,
        )
    candidates = sorted(
        p
        for p in dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
    )
    return [load_suite(p) for p in candidates]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_yaml(text: str) -> Any:
    """Parse ``text`` as YAML using ``ruamel.yaml`` round-trip mode.

    Round-trip mode preserves comments and mapping order on the parsed
    document; we only need the structural data here, but using the
    same loader keeps behaviour consistent with the writer.

    Raises:
        SuiteValidationError: The text is not valid YAML. Line number
            is pulled from the parser's ``problem_mark`` when
            available (1-indexed).
    """
    yaml = YAML(typ="rt")
    try:
        return yaml.load(text)
    except (ScannerError, ParserError) as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        raise SuiteValidationError(
            path=None,
            test_case_id=None,
            missing_field=None,
            message=f"Invalid YAML: {exc}",
            line=line,
        ) from exc
    except YAMLError as exc:
        raise SuiteValidationError(
            path=None,
            test_case_id=None,
            missing_field=None,
            message=f"Invalid YAML: {exc}",
            line=None,
        ) from exc


def _parse_json(text: str) -> Any:
    """Parse ``text`` as JSON using the standard library.

    Raises:
        SuiteValidationError: The text is not valid JSON. Line number
            comes from :attr:`json.JSONDecodeError.lineno` (1-indexed).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SuiteValidationError(
            path=None,
            test_case_id=None,
            missing_field=None,
            message=f"Invalid JSON: {exc.msg}",
            line=exc.lineno,
        ) from exc


def _wrap_validation_error(
    exc: ValidationError, *, data: Any
) -> SuiteValidationError:
    """Translate a Pydantic :class:`ValidationError` into our own error.

    Extracts the **first** error from ``exc.errors()`` and maps it to:

    * ``missing_field`` — the full dotted path of the Pydantic ``loc``
      components joined with ``.``. For example
      ``("test_cases", 0, "prompt") → "test_cases.0.prompt"``. ``None``
      when ``loc`` is empty (model-level validators raise with
      ``loc = ()``).
    * ``test_case_id`` — the ``id`` of the test case at
      ``data["test_cases"][i]`` when the Pydantic error path traverses
      an integer index under ``test_cases``, else ``None``.
    * ``message`` — the Pydantic ``msg`` field unchanged, so downstream
      tooling (CLI, UI) can render the same text users see from Pydantic
      in other contexts.
    * ``line`` — always ``None`` for validation errors; line numbers
      are only available for syntax errors.
    """
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc: tuple[Any, ...] = tuple(first.get("loc", ()))
        message = str(first.get("msg", "Validation failed"))
    else:  # pragma: no cover - Pydantic always produces at least one error
        loc = ()
        message = "Validation failed"

    missing_field = ".".join(str(component) for component in loc) if loc else None
    test_case_id = _extract_test_case_id(data, loc)

    return SuiteValidationError(
        path=None,
        test_case_id=test_case_id,
        missing_field=missing_field,
        message=message,
        line=None,
    )


def _extract_test_case_id(data: Any, loc: tuple[Any, ...]) -> str | None:
    """Return the ``TestCase.id`` pointed at by ``loc``, if any.

    When ``loc`` begins with ``("test_cases", idx, ...)`` where ``idx``
    is an integer, look up ``data["test_cases"][idx]["id"]`` in the
    raw parsed document (``CommentedMap`` from ruamel or plain ``dict``
    from json). Returns the id when it is present and a string; falls
    back to ``None`` when any step of the lookup fails (wrong type,
    missing key, missing id) so the loader never shadows the primary
    error with an id-extraction bug.
    """
    if len(loc) < 2 or loc[0] != "test_cases":
        return None
    idx = loc[1]
    if not isinstance(idx, int):
        return None
    try:
        test_cases = data["test_cases"]
        tc = test_cases[idx]
        tc_id = tc["id"]
    except (KeyError, IndexError, TypeError):
        return None
    return tc_id if isinstance(tc_id, str) else None


__all__ = [
    "SuiteValidationError",
    "discover_suites",
    "load_suite",
    "load_suite_from_string",
]
