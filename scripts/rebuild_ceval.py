"""Quick one-shot: rebuild just the ceval-mixed suite after fixing the
subject list."""

import importlib.util
import pathlib
import sys

# Load the materialise_tier3 module by absolute file path so this
# script works whether or not ``scripts/`` is a Python package.
_here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "materialise_tier3", _here / "materialise_tier3.py"
)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["materialise_tier3"] = mod
spec.loader.exec_module(mod)

mod._write_suite(mod.build_ceval_mixed())
