"""Property 43: Local/remote mode equivalence.

For every adapter ``a`` (and HuggingFace spec), given a fixture row
list ``R``, the :class:`EvaluationSuite` produced by
``a.materialise("local", opts, cache_dir)`` equals the one produced
by ``a.materialise("remote", opts, cache_dir)`` up to
:class:`~ollama_evaluator.suites.models.TestCase` identity:

* Same suite ``name``.
* Same ordered list of ``TestCase.id``.
* Each paired test case has equal ``prompt``, ``system_prompt``,
  ``expected_output``, ``reference_data``, ``tags``, and ``metrics``.

The property is stated in
``.kiro/specs/ollama-model-evaluator/design.md`` §Correctness
Properties as Property 43 and validates Requirement 17.10.

Approach
--------
Each test drives both modes from the *same* row source:

* **Local mode.** Writes the fixture rows to a JSONL file under a
  temporary ``cache_dir`` using the naming convention expected by
  :func:`ollama_evaluator.suites.huggingface._read_local_rows`
  (``<cache_dir>/<adapter_name>/[<config>/]<split>.jsonl``), then
  calls ``adapter.materialise("local", ..., cache_dir)``.

* **Remote mode.** Patches
  :func:`ollama_evaluator.suites.huggingface._stream_remote_rows`
  to yield the same row list, then calls
  ``adapter.materialise("remote", ...)``. No real network I/O
  happens.

The two suites are compared with ``suite_a == suite_b`` — Pydantic
v2 models define ``__eq__`` as structural equality, which is exactly
the TestCase-identity equality Property 43 specifies.

``max_examples`` is low because each example writes a temp file and
the assertion is pure equality; there is no combinatorial space to
explore beyond "the two modes share the same transform".
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ollama_evaluator.suites import huggingface as hf
from ollama_evaluator.suites.adapter_base import AdapterOptions, HFRef
from ollama_evaluator.suites.gsm8k import Gsm8kAdapter
from ollama_evaluator.suites.hellaswag import HellaSwagAdapter
from ollama_evaluator.suites.humaneval import HumanEvalAdapter
from ollama_evaluator.suites.mmlu import MMluAdapter
from ollama_evaluator.suites.truthfulqa import TruthfulQaAdapter

# Fixture rows: one short, valid row per adapter. A single row is
# enough to exercise the local/remote code paths; the transform itself
# is covered by Property 42's Hypothesis-driven tests.
_MMLU_ROWS: list[dict[str, Any]] = [
    {
        "question": "What is 2 + 2?",
        "choices": ["1", "4", "3", "5"],
        "answer": 1,
        "subject": "math",
    },
    {
        "question": "Who wrote Hamlet?",
        "choices": ["Dickens", "Austen", "Shakespeare", "Joyce"],
        "answer": 2,
        "subject": "literature",
    },
]

_HELLASWAG_ROWS: list[dict[str, Any]] = [
    {
        "ctx": "A man is walking down the street and",
        "endings": ["trips", "sings", "flies", "shouts"],
        "label": "0",
        "ind": 42,
        "activity_label": "Home",
    },
]

_TRUTHFULQA_ROWS: list[dict[str, Any]] = [
    {
        "question": "What is the capital of France?",
        "mc1_targets": {
            "choices": ["Paris", "London", "Berlin"],
            "labels": [1, 0, 0],
        },
        "category": "Geography",
    },
]

_GSM8K_ROWS: list[dict[str, Any]] = [
    {
        "question": "If a bag has 5 apples and 3 are eaten, how many remain?",
        "answer": "5 - 3 = 2\n#### 2",
    },
]

_HUMANEVAL_ROWS: list[dict[str, Any]] = [
    {
        "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"",
        "canonical_solution": "    return a + b",
        "test": "assert add(1, 2) == 3",
        "entry_point": "add",
    },
]


def _write_local_cache(
    cache_dir: Path,
    adapter_name: str,
    ref: HFRef,
    rows: list[dict[str, Any]],
) -> None:
    """Persist ``rows`` under the path layout the local reader expects."""
    base = cache_dir / adapter_name
    if ref.config is not None:
        base = base / ref.config
    base.mkdir(parents=True, exist_ok=True)
    split = ref.split or "train"
    jsonl_path = base / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


@pytest.mark.parametrize(
    "adapter_factory, rows",
    [
        pytest.param(MMluAdapter, _MMLU_ROWS, id="mmlu"),
        pytest.param(HellaSwagAdapter, _HELLASWAG_ROWS, id="hellaswag"),
        pytest.param(TruthfulQaAdapter, _TRUTHFULQA_ROWS, id="truthfulqa"),
        pytest.param(Gsm8kAdapter, _GSM8K_ROWS, id="gsm8k"),
        pytest.param(HumanEvalAdapter, _HUMANEVAL_ROWS, id="humaneval"),
    ],
)
def test_adapter_local_and_remote_produce_equal_suites(
    adapter_factory: Callable[[], Any],
    rows: list[dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Validates: Requirement 17.10**

    For every adapter, ``materialise("local", ...)`` and
    ``materialise("remote", ...)`` return equal :class:`EvaluationSuite`
    objects when both are fed the same source rows. Equality is the
    Pydantic v2 structural ``__eq__``, which implies equal ``name``,
    equal ordered ``TestCase.id`` list, and equal per-field contents.
    """
    adapter = adapter_factory()
    opts = AdapterOptions()

    # Stage the local cache.
    _write_local_cache(tmp_path, adapter.ADAPTER_NAME, adapter.DEFAULT_HF_REF, rows)
    local_suite = adapter.materialise("local", opts, tmp_path)

    # Stub out the remote streamer. Patching the private helper keeps
    # the public :func:`stream_rows` dispatcher honest — it must call
    # ``_stream_remote_rows`` for ``remote`` mode.
    def fake_stream(ref: HFRef) -> Iterator[dict[str, Any]]:
        assert ref == adapter.DEFAULT_HF_REF
        yield from rows

    monkeypatch.setattr(hf, "_stream_remote_rows", fake_stream)
    remote_suite = adapter.materialise("remote", opts, None)

    # Structural equality asserts every Property-43 sub-requirement at once:
    # same name, same ordered ids, same per-field TestCase payloads.
    assert local_suite == remote_suite


def test_huggingface_loader_local_and_remote_produce_equal_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**Validates: Requirement 17.10**

    Covers the generic :func:`materialise_hf` entry point too. The
    fixture rows use a dotted + bracketed path (``answers.text[0]``)
    so the field-map resolver is exercised in both modes.
    """
    rows = [
        {
            "question": "q1",
            "answers": {"text": ["a1", "alt1"]},
            "category": "cat-a",
        },
        {
            "question": "q2",
            "answers": {"text": ["a2"]},
            "category": "cat-b",
        },
    ]
    ref = HFRef(repo_id="demo/qa", config="plain_text", split="validation")
    spec = hf.HFSuiteSpec(
        name="demo",
        hf_ref=ref,
        field_map=hf.HFFieldMap(
            prompt="question",
            expected_output="answers.text[0]",
            tags_from=["category"],
        ),
        metrics=[hf.MetricConfig(name="exact-match")],
    )

    # Local: write JSONL into the slugified cache path.
    base = tmp_path / ref.repo_id.replace("/", "__") / ref.config
    base.mkdir(parents=True, exist_ok=True)
    (base / "validation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    local_suite = hf.materialise_hf(spec, mode="local", cache_dir=tmp_path)

    # Remote: stub the private streamer to yield the same rows.
    def fake_stream(stream_ref: HFRef) -> Iterator[dict[str, Any]]:
        assert stream_ref == ref
        yield from rows

    monkeypatch.setattr(hf, "_stream_remote_rows", fake_stream)
    remote_suite = hf.materialise_hf(spec, mode="remote", cache_dir=None)

    assert local_suite == remote_suite
