"""Probe alternatives for the datasets that failed under datasets v4."""

from __future__ import annotations

from datasets import load_dataset

PROBES: list[tuple[str, str | None, str]] = [
    # PIQA alternatives (parquet-only mirrors)
    ("ybisk/piqa", "plain_text", "validation"),
    ("piqa", "plain_text", "validation"),
    # CMMLU alternatives
    ("lukaemon/cmmlu", "agronomy", "test"),
    # MATH alternatives
    ("HuggingFaceH4/MATH-500", None, "test"),
    ("lighteval/MATH", None, "test"),
    # BBH full config (default "all" maybe not set)
    ("lukaemon/bbh", None, "test"),
    # ToolBench (smaller variant)
    ("qiaojin/PubMedQA", "pqa_labeled", "train"),
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
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                print(f"    {k}: {preview}")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            print(f"  FAILED: {type(exc).__name__}: {msg}")


if __name__ == "__main__":
    main()
