"""Evaluation suite loading, writing, and dataset adapters.

Re-exports the loader and writer entry points so callers can do
``from ollama_evaluator.suites import load_suite`` without reaching into
the private module layout. Concrete modules for the dataset adapters
(mmlu, hellaswag, truthfulqa, gsm8k, humaneval, huggingface) are added
in later tasks.
"""

from .loader import (
    SuiteValidationError,
    discover_suites,
    load_suite,
    load_suite_from_string,
)
from .writer import dump_suite

__all__ = [
    "SuiteValidationError",
    "discover_suites",
    "dump_suite",
    "load_suite",
    "load_suite_from_string",
]
