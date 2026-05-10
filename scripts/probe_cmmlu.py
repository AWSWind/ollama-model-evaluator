from datasets import load_dataset

for repo, config in [
    ("haonan-li/cmmlu", "agronomy"),
    ("ADNYA/CMMLU", None),
    ("R1000/CMMLU", None),
    ("fzmnm/cmmlu_lite", None),
    ("vlsp-2023-vllm/CMMLU", None),
]:
    print(f"repo={repo} config={config!r}")
    try:
        if config:
            ds = load_dataset(repo, config, split="test", streaming=True)
        else:
            ds = load_dataset(repo, split="test", streaming=True)
        first = next(iter(ds))
        print("  keys:", sorted(first.keys()))
        for k, v in list(first.items())[:5]:
            preview = repr(v)[:150]
            print(f"    {k}: {preview}")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:150]}")
    print()
