"""Materialise Tier-2 public-benchmark suites on .224.

Run this on the remote host; it downloads rows from HuggingFace using
each adapter's ``materialise(mode='remote', ...)`` entry point, then
writes a standard Evaluation_Suite YAML under ``examples/suites/``.

Deliberate choices
------------------
* Each suite is sub-sampled with ``limit`` + ``seed`` so downloads
  stay under control and runs on a 27B model finish in minutes.
* MMLU is too big to run unsliced (14k questions × 57 subjects);
  we materialise a single mixed 200-row sample across subjects using
  ``subject=None``.
* HumanEval is kept at its full 164 rows because it uses ``response-capture``
  (no execution scoring) and we may want the complete set available.

Invocation
----------
    cd /home/azurewind/workspaces/AI-Model-Evaluation
    .venv/bin/python scripts/materialise_tier2.py
"""

from __future__ import annotations

from pathlib import Path

from ollama_evaluator.suites.adapter_base import AdapterOptions, HFRef
from ollama_evaluator.suites.adapters import get_adapter
from ollama_evaluator.suites.writer import dump_suite


OUTPUT_DIR = Path("examples/suites")


def _write(adapter_name: str, opts: AdapterOptions) -> None:
    print(f"[{adapter_name}] fetching… (limit={opts.limit}, seed={opts.seed})")
    adapter = get_adapter(adapter_name)
    try:
        suite = adapter.materialise(mode="remote", opts=opts, cache_dir=None)
    except Exception as exc:  # noqa: BLE001 - user-facing error boundary
        print(f"[{adapter_name}] FAILED: {type(exc).__name__}: {exc}")
        return

    # Replace any '/' in the suite name with '-' for a clean filename.
    safe_name = suite.name.replace("/", "-")
    output_path = OUTPUT_DIR / f"{safe_name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_suite(suite, "yaml"), encoding="utf-8")
    print(f"[{adapter_name}] wrote {output_path} ({len(suite.test_cases)} test cases)")


def main() -> None:
    # Small, meaningful slices for benchmarking a single model in a few
    # minutes. Raise the limits later for a bigger production run.

    # MMLU — mixed subjects, 200 questions.
    _write("mmlu", AdapterOptions(limit=200, seed=42))

    # HellaSwag — commonsense completions, 200 rows.
    _write("hellaswag", AdapterOptions(limit=200, seed=42))

    # TruthfulQA — MC1 form only in v1, 200 rows.
    _write("truthfulqa", AdapterOptions(limit=200, seed=42, form="mc1"))

    # GSM8K — grade-school math, 100 rows.
    _write("gsm8k", AdapterOptions(limit=100, seed=42))

    # HumanEval — all 164 rows (response-capture, not scored here).
    _write("humaneval", AdapterOptions(limit=164, seed=42))


if __name__ == "__main__":
    main()
