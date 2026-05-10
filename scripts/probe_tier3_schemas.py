"""Probe schemas of the Tier-3 HuggingFace datasets so we can write
field maps confidently. Runs on .224 (which has internet + datasets
installed).

Emits one row preview per dataset. Any dataset that fails to load is
reported with the raw error so we know to skip it or pick a different
ref.
"""

from __future__ import annotations

from datasets import load_dataset

PROBES: list[tuple[str, str | None, str]] = [
    ("lukaemon/bbh", "logical_deduction_three_objects", "test"),
    ("allenai/ai2_arc", "ARC-Challenge", "test"),
    ("allenai/ai2_arc", "ARC-Easy", "test"),
    ("ybisk/piqa", None, "validation"),
    ("winogrande", "winogrande_xl", "validation"),
    ("ceval/ceval-exam", "computer_network", "val"),
    ("haonan-li/cmmlu", "agronomy", "test"),
    ("hendrycks/competition_math", None, "test"),
    ("google-research-datasets/mbpp", "full", "test"),
    ("rajpurkar/squad_v2", None, "validation"),
    ("google/IFEval", None, "train"),
    ("HuggingFaceH4/mt_bench_prompts", None, "train"),
]


def main() -> None:
    for repo, config, split in PROBES:
        print("=" * 70)
        print(f"repo={repo} config={config!r} split={split!r}")
        try:
            if config:
                ds = load_dataset(repo, config, split=split, streaming=True)
            else:
                ds = load_dataset(repo, split=split, streaming=True)
            first = next(iter(ds))
            print(f"  sample keys: {sorted(first.keys())}")
            for k, v in first.items():
                preview = repr(v)
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(f"    {k}: {preview}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
