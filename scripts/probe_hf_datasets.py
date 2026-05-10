"""Quick probe of the 5 HF datasets to see their current row schemas."""

from __future__ import annotations

from datasets import load_dataset

PROBES = [
    ("cais/mmlu", "all", "test"),
    ("Rowan/hellaswag", None, "validation"),
    ("truthful_qa", "multiple_choice", "validation"),
    ("openai/gsm8k", "main", "test"),
    ("openai_humaneval", None, "test"),
]


def main() -> None:
    for repo, config, split in PROBES:
        print("=" * 70)
        print(f"repo={repo} config={config!r} split={split!r}")
        try:
            ds = load_dataset(repo, config, split=split, streaming=True) if config else load_dataset(repo, split=split, streaming=True)
            it = iter(ds)
            first = next(it)
            print(f"  sample keys: {sorted(first.keys())}")
            for k, v in first.items():
                preview = repr(v)
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                print(f"    {k}: {preview}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
