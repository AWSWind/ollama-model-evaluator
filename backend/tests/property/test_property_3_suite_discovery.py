"""Property 3: Evaluation_Suite discovery.

For every directory containing a set of files, :func:`discover_suites`
returns exactly the suites produced by loading each file in that
directory whose extension is one of ``.yaml``, ``.yml``, or ``.json``,
in case-sensitive sorted filename order, and nothing else.

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness Properties
as Property 3 and is directly driven by **Requirement 3.1** — *"THE
Backend SHALL load Evaluation_Suites from files in a user-specified
directory"*. The other requirements in the §3 cluster cover the
parse-level behaviour that is exercised by Properties 1 and 2; here we
specifically assert the directory-level semantics:

1. **Inclusivity.** Every ``.yaml``/``.yml``/``.json`` file in the
   directory is loaded.
2. **Exclusivity.** Files with any other extension (``.md``, ``.txt``,
   dotfiles without a recognised suffix, etc.) are skipped.
3. **Ordering.** Loaded suites are returned in the deterministic order
   produced by ``sorted()`` over the containing ``Path`` objects,
   which — because all files in a single directory share the same
   parent — reduces to sorted filename order. To keep that ordering
   identical across OSes (``pathlib`` compares case-insensitively on
   Windows) the generated stems are restricted to a single case.

Approach
--------
Each Hypothesis example draws 0..5 :class:`EvaluationSuite` instances
with mutually distinct names plus a parallel list of unique
alphanumeric filename stems, each paired with a randomly chosen
extension from ``{.yaml, .yml, .json}``. Inside the test body we open a
fresh :func:`tempfile.TemporaryDirectory` (not pytest's ``tmp_path``
fixture, which Hypothesis cannot drive across multiple examples per
test), dump each suite with :func:`dump_suite` in the format that
matches its extension, drop a handful of fixed "noise" files whose
extensions are deliberately *not* in the supported set, and call
:func:`discover_suites` on the directory. We then assert the three
invariants above against the returned list.

``max_examples=20`` is lower than the 100 used by Properties 1 and 2
because every example performs real filesystem I/O (tempdir create,
up to five file writes, one directory scan, up to five file reads,
tempdir teardown). 30 is enough to shuffle through several hundred
distinct (suite-count, extension-mix, filename-ordering) combinations
— far more than the handful of edge cases a hand-authored test would
cover — without bloating the CI runtime. ``deadline=None`` avoids
false positives on slow disks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from string import ascii_lowercase, digits

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ollama_evaluator.suites import discover_suites, dump_suite
from ollama_evaluator.suites.models import EvaluationSuite

from .generators import evaluation_suites

# Lowercase-alphanumeric filename stems keep the generated tree
# portable across every filesystem Hypothesis might run on (ext4,
# NTFS, APFS) and the generated ordering deterministic across every
# OS:
#
# * Case-insensitive filesystems (NTFS, APFS default) collapse
#   ``A00.yaml`` and ``a00.yaml`` to the same on-disk entry; using
#   a single case eliminates the collision.
# * :class:`pathlib.PurePath` sort order is OS-dependent — Linux
#   compares bytes, Windows case-normalises via ``_str_normcase``.
#   Single-case stems make the two behaviours agree so the test's
#   expected-order assertion holds uniformly.
#
# Stems are also guaranteed not to start with ``.`` (no digits vs
# dot ambiguity) and cannot contain path separators.
_FILENAME_ALPHABET = ascii_lowercase + digits

# Supported extensions (must match ``_SUPPORTED_EXTS`` in ``suites/loader.py``).
# Maintained as a module-level constant so a future change to the
# loader has a single test-side counterpart to update.
_SUPPORTED_EXTENSIONS: tuple[str, ...] = (".yaml", ".yml", ".json")

# Noise files written alongside every generated suite. They exercise
# three distinct "should be ignored" shapes that real users' suite
# directories commonly contain:
#
# * ``README.md`` — a documentation file with a non-supported but
#   otherwise well-formed extension.
# * ``NOTES.txt`` — a plain text sidecar; the ``.txt`` suffix is the
#   most common "not a suite" extension in authored trees.
# * ``.gitkeep`` — a version-control placeholder with no extension at
#   all (``Path(".gitkeep").suffix == ""``). This is the hardest case
#   to silently include by accident: a lenient implementation that
#   fell back to ``suffix in ("", ".yaml", …)`` would try to load it
#   as YAML and blow up.
#
# The *contents* of these files are deliberately non-JSON and
# non-YAML-suite-shaped so that if the implementation ever regressed
# to scanning them, the regression would be loud (a
# :class:`SuiteValidationError`) rather than silent.
_NOISE_FILES: dict[str, str] = {
    "README.md": "This directory holds evaluation suites.\n",
    "NOTES.txt": "Authored by hand; do not commit secrets.\n",
    ".gitkeep": "",
}


@st.composite
def _suite_file_plans(
    draw: st.DrawFn,
) -> list[tuple[EvaluationSuite, str, str]]:
    """Draw a list of ``(suite, filename, fmt)`` plans for one test run.

    The list length is in ``0..5``. Suite *names* are unique (via
    Hypothesis' ``unique_by``) so that multiset assertions in the test
    body are unambiguous — two suites with identical content but
    different names would still compare equal under Pydantic's
    structural equality, so collapsing on ``name`` is the right level
    of uniqueness to enforce.

    Filename *stems* are drawn with ``unique=True`` so the tempdir
    never ends up with two files whose full names collide; this in
    turn means :func:`discover_suites`' sorted-filename order is
    well-defined (no ties to break).

    The format is derived from the extension rather than drawn
    independently: ``.json`` files must contain JSON (``fmt="json"``),
    and ``.yaml``/``.yml`` files must contain YAML (``fmt="yaml"``).
    Decoupling them would generate mismatched pairs that
    :func:`discover_suites` would reject on read — a shape that is
    already covered by Property 2.
    """
    suites: list[EvaluationSuite] = draw(
        st.lists(
            evaluation_suites(),
            min_size=0,
            max_size=5,
            unique_by=lambda s: s.name,
        )
    )
    n = len(suites)
    stems: list[str] = draw(
        st.lists(
            st.text(alphabet=_FILENAME_ALPHABET, min_size=3, max_size=10),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    extensions: list[str] = draw(
        st.lists(
            st.sampled_from(_SUPPORTED_EXTENSIONS),
            min_size=n,
            max_size=n,
        )
    )
    plans: list[tuple[EvaluationSuite, str, str]] = []
    for suite, stem, ext in zip(suites, stems, extensions, strict=True):
        fmt = "json" if ext == ".json" else "yaml"
        plans.append((suite, f"{stem}{ext}", fmt))
    return plans


@given(plans=_suite_file_plans())
@settings(
    max_examples=20,
    deadline=None,
    # Filesystem I/O inside the test body is intentionally variable;
    # suppress the ``too_slow`` health check so a loaded runner
    # doesn't cause spurious failures.
    suppress_health_check=[HealthCheck.too_slow],
)
def test_discover_suites_loads_exactly_supported_files_in_sorted_order(
    plans: list[tuple[EvaluationSuite, str, str]],
) -> None:
    """**Validates: Requirement 3.1**

    For any directory ``D`` containing the files described by
    ``plans`` plus a fixed set of noise files,
    :func:`discover_suites` returns a list that:

    1. has length equal to the number of suite files (noise is
       ignored),
    2. is a permutation of the generated suites — equivalently, the
       multiset of loaded suites equals the multiset of generated
       suites,
    3. is ordered by the source file's sorted filename (matching
       the loader's ``sorted(p for p in dir.iterdir() …)`` contract;
       on a single-directory scan this reduces to filename order).

    We use :func:`tempfile.TemporaryDirectory` rather than pytest's
    ``tmp_path`` fixture because Hypothesis invokes the test body
    once per generated example, and pytest fixtures are scoped to a
    single invocation — reusing ``tmp_path`` across examples would
    bleed files from earlier draws into later ones.
    """
    with tempfile.TemporaryDirectory() as raw_dir:
        dir_path = Path(raw_dir)

        # Materialise every planned suite file. ``dump_suite`` is
        # chosen to match the extension so the file is guaranteed to
        # round-trip through :func:`discover_suites` — Property 1
        # already covers ``load(dump(s)) == s`` at the single-file
        # level, which is the invariant we rely on here.
        for suite, filename, fmt in plans:
            (dir_path / filename).write_text(
                dump_suite(suite, fmt), encoding="utf-8"
            )

        # Drop the noise files. Their contents are unstructured text;
        # a regression that caused them to be loaded would surface
        # as a :class:`SuiteValidationError`, not a silent pass.
        for noise_name, noise_content in _NOISE_FILES.items():
            (dir_path / noise_name).write_text(
                noise_content, encoding="utf-8"
            )

        discovered = discover_suites(dir_path)

    # (1) Length: every suite file is loaded, no noise file is loaded.
    assert len(discovered) == len(plans), (
        f"Expected {len(plans)} discovered suite(s); got {len(discovered)}. "
        f"Noise files present in the directory: {sorted(_NOISE_FILES)}"
    )

    # (2) Multiset of suites matches. Suite names are unique per the
    # strategy, so comparing the sorted-by-name lists is equivalent
    # to multiset equality and gives a stable failure message.
    assert sorted((s for s, _, _ in plans), key=lambda s: s.name) == sorted(
        discovered, key=lambda s: s.name
    )

    # (3) Ordering: the loader sorts ``Path`` objects from
    # ``dir.iterdir()``. For files in the same directory this
    # collapses to sorted filename order, so we build the expected
    # ordering by sorting ``plans`` on the generated filename and
    # projecting the suite field.
    expected_in_order = [
        suite for suite, _, _ in sorted(plans, key=lambda plan: plan[1])
    ]
    assert discovered == expected_in_order
