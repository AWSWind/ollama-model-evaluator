"""Materialise Tier-3 public-benchmark suites on .224.

Each suite is authored as a direct `rows → EvaluationSuite` transformation so
the resulting YAML is indistinguishable from a hand-authored suite and shows
up in the UI alongside the others.

Invocation
----------
    cd /home/azurewind/workspaces/AI-Model-Evaluation
    .venv/bin/python scripts/materialise_tier3.py

Output
------
    examples/suites/<suite-name>.yaml  (one file per suite)

Every suite carries a human-readable ``description`` so the UI can show a
one-line tooltip. Descriptions are intentionally short (one sentence) so
they fit in the New Run dropdown without truncation.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset

from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)
from ollama_evaluator.suites.writer import dump_suite


OUTPUT_DIR = Path("examples/suites")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _take(ds_iter, limit: int, seed: int) -> list[dict]:
    """Pull at most ``limit`` rows deterministically from a streaming iterator.

    Reservoir-like: collect `limit * 3` candidates then take a seeded
    shuffle's first `limit` so order is deterministic for a given seed
    without needing the full materialised dataset.
    """
    candidates = list(itertools.islice(ds_iter, max(limit, 1) * 3))
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def _write_suite(suite: EvaluationSuite) -> None:
    safe = suite.name.replace("/", "-")
    out = OUTPUT_DIR / f"{safe}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_suite(suite, "yaml"), encoding="utf-8")
    print(f"[{suite.name}] wrote {out} ({len(suite.test_cases)} cases)")


def _safe_build(name: str, builder: Callable[[], EvaluationSuite]) -> None:
    print(f"[{name}] fetching…")
    try:
        suite = builder()
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
        return
    _write_suite(suite)


# ---------------------------------------------------------------------------
# BigBench-Hard — mixed
# ---------------------------------------------------------------------------


BBH_MIXED_SUBSETS = [
    "logical_deduction_three_objects",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "penguins_in_a_table",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
]


def build_bbh_mixed(per_subset_limit: int = 100, seed: int = 42) -> EvaluationSuite:
    test_cases: list[TestCase] = []
    for subset in BBH_MIXED_SUBSETS:
        ds = load_dataset("lukaemon/bbh", subset, split="test", streaming=True)
        rows = _take(ds, per_subset_limit, seed)
        for i, row in enumerate(rows):
            prompt = (
                f"{row['input']}\n\n"
                "Put your final answer on a new line prefixed with "
                "'Final answer:'. For multiple-choice tasks, use the "
                "option letter in parentheses (e.g. '(A)')."
            )
            target = str(row["target"]).strip()
            # BBH targets are either "(A)"-like or direct text.
            escaped = target.replace("(", r"\(").replace(")", r"\)")
            test_cases.append(
                TestCase(
                    id=f"bbh/{subset}/{i}",
                    prompt=prompt,
                    expected_output=target,
                    tags=["bbh", subset],
                    metrics=[
                        MetricConfig(
                            name="regex-match",
                            params={
                                "pattern": rf"[Ff]inal answer:\s*{escaped}",
                            },
                        )
                    ],
                )
            )
    return EvaluationSuite(
        name="bbh-mixed",
        description=(
            "BigBench-Hard: eight hard-reasoning subsets (logical deduction, "
            "causal judgement, date understanding, disambiguation, formal "
            "fallacies, penguin-table, object tracking, web of lies). Good "
            "stress-test for step-by-step reasoning."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=512),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# ARC (Challenge / Easy)
# ---------------------------------------------------------------------------


def _arc_suite(config: str, suite_name: str, description: str, limit: int, seed: int) -> EvaluationSuite:
    ds = load_dataset("allenai/ai2_arc", config, split="test", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for i, row in enumerate(rows):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = row["answerKey"]
        if answer_key not in labels:
            # Some rows use numeric labels; skip them to keep the metric simple.
            continue
        enumerated = "\n".join(f"{labels[j]}) {texts[j]}" for j in range(len(labels)))
        prompt = (
            f"Question: {row['question']}\n{enumerated}\n"
            f"Answer with the single option letter."
        )
        test_cases.append(
            TestCase(
                id=f"{suite_name}/{row['id']}",
                prompt=prompt,
                expected_output=answer_key,
                tags=[suite_name, "science-mcq"],
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={"pattern": rf"\b{answer_key}\b"},
                    )
                ],
            )
        )
    return EvaluationSuite(
        name=suite_name,
        description=description,
        defaults=GenerationDefaults(temperature=0.0, max_tokens=256),
        test_cases=test_cases,
    )


def build_arc_challenge(limit: int = 300, seed: int = 42) -> EvaluationSuite:
    return _arc_suite(
        "ARC-Challenge",
        "arc-challenge",
        "AI2 ARC-Challenge: harder grade-school science MCQ. Good for "
        "probing commonsense + science reasoning; harder than ARC-Easy.",
        limit,
        seed,
    )


def build_arc_easy(limit: int = 300, seed: int = 42) -> EvaluationSuite:
    return _arc_suite(
        "ARC-Easy",
        "arc-easy",
        "AI2 ARC-Easy: grade-school science MCQ, easier split. Good "
        "baseline for basic science/knowledge.",
        limit,
        seed,
    )


# ---------------------------------------------------------------------------
# PIQA (Physical Interaction QA)
# ---------------------------------------------------------------------------


def build_piqa(limit: int = 300, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("piqa", "plain_text", split="validation", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for i, row in enumerate(rows):
        # label == 0 → sol1, label == 1 → sol2
        expected = "A" if row["label"] == 0 else "B"
        prompt = (
            f"Goal: {row['goal']}\n"
            f"A) {row['sol1']}\n"
            f"B) {row['sol2']}\n"
            f"Which solution (A or B) achieves the goal? Answer with a single letter."
        )
        test_cases.append(
            TestCase(
                id=f"piqa/{i}",
                prompt=prompt,
                expected_output=expected,
                tags=["piqa", "physical-commonsense"],
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={"pattern": rf"\b{expected}\b"},
                    )
                ],
            )
        )
    return EvaluationSuite(
        name="piqa",
        description=(
            "Physical Interaction QA: 'A vs B' commonsense about how "
            "everyday physical tasks work. Probes whether the model has "
            "grounded everyday-world knowledge."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=128),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# WinoGrande
# ---------------------------------------------------------------------------


def build_winogrande(limit: int = 300, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for i, row in enumerate(rows):
        # answer is "1" or "2"
        expected = row["answer"]
        if expected not in ("1", "2"):
            continue
        mapped = "A" if expected == "1" else "B"
        prompt = (
            f"Fill in the blank ('_') with one of the two options.\n"
            f"Sentence: {row['sentence']}\n"
            f"A) {row['option1']}\n"
            f"B) {row['option2']}\n"
            f"Answer with a single letter."
        )
        test_cases.append(
            TestCase(
                id=f"winogrande/{i}",
                prompt=prompt,
                expected_output=mapped,
                tags=["winogrande", "coreference"],
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={"pattern": rf"\b{mapped}\b"},
                    )
                ],
            )
        )
    return EvaluationSuite(
        name="winogrande",
        description=(
            "WinoGrande XL: pronoun resolution benchmark. Tests whether "
            "the model uses world knowledge to disambiguate which noun a "
            "pronoun refers to."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=128),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# C-Eval — Chinese academic
# ---------------------------------------------------------------------------


CEVAL_MIXED_SUBJECTS = [
    "computer_network",
    "high_school_mathematics",
    "chinese_language_and_literature",
    "law",
    "modern_chinese_history",
    "college_economics",
]


def build_ceval_mixed(per_subject_limit: int = 50, seed: int = 42) -> EvaluationSuite:
    test_cases: list[TestCase] = []
    for subject in CEVAL_MIXED_SUBJECTS:
        try:
            ds = load_dataset("ceval/ceval-exam", subject, split="val", streaming=True)
            rows = _take(ds, per_subject_limit, seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  ceval/{subject}: skipped ({exc})")
            continue
        for i, row in enumerate(rows):
            answer = str(row["answer"]).strip()
            if answer not in ("A", "B", "C", "D"):
                continue
            prompt = (
                f"问题:{row['question']}\n"
                f"A) {row['A']}\n"
                f"B) {row['B']}\n"
                f"C) {row['C']}\n"
                f"D) {row['D']}\n"
                f"请只用一个字母(A、B、C 或 D)回答。"
            )
            test_cases.append(
                TestCase(
                    id=f"ceval/{subject}/{i}",
                    prompt=prompt,
                    expected_output=answer,
                    tags=["ceval", subject, "zh"],
                    metrics=[
                        MetricConfig(
                            name="regex-match",
                            params={"pattern": rf"\b{answer}\b"},
                        )
                    ],
                )
            )
    return EvaluationSuite(
        name="ceval-mixed",
        description=(
            "C-Eval: Chinese academic MCQ across six subjects (computer "
            "network, HS math, Chinese literature, law, modern history, "
            "economics). Evaluates Chinese-language knowledge."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=128),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------


def build_math500(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for i, row in enumerate(rows):
        prompt = (
            f"Solve the problem below. Put the final answer in \\boxed{{...}} "
            f"at the end of your response.\n\n"
            f"Problem: {row['problem']}\n\nSolution:"
        )
        test_cases.append(
            TestCase(
                id=f"math500/{row['unique_id']}",
                prompt=prompt,
                expected_output=str(row["answer"]),
                tags=["math500", str(row["subject"]), f"level-{row['level']}"],
                metrics=[
                    # Capture-only: grading mathematical equivalence of LaTeX
                    # boxed answers needs a heavier comparator than our
                    # built-in metrics ship. Expected output is recorded so
                    # an external grader can compute correctness later.
                    MetricConfig(name="response-capture"),
                ],
            )
        )
    return EvaluationSuite(
        name="math-500",
        description=(
            "MATH-500: high-school olympiad problems (Hendrycks MATH, 500-"
            "row eval subset). Much harder than GSM8K. Uses response-"
            "capture — external grader required to score correctness of "
            "LaTeX \\boxed{} answers."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=1024),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------


def build_mbpp(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for row in rows:
        prompt = (
            f"{row['text']}\n\n"
            f"Your function should pass these tests:\n"
            + "\n".join(row["test_list"])
            + "\n\nReturn only the Python code, no commentary."
        )
        test_cases.append(
            TestCase(
                id=f"mbpp/{row['task_id']}",
                prompt=prompt,
                expected_output=row["code"],
                reference_data={
                    "test_list": row["test_list"],
                    "test_setup_code": row.get("test_setup_code", ""),
                },
                tags=["mbpp", "code"],
                metrics=[MetricConfig(name="response-capture")],
            )
        )
    return EvaluationSuite(
        name="mbpp",
        description=(
            "MBPP: 'mostly basic Python problems' — short descriptions "
            "paired with unit tests. Uses response-capture; execution-"
            "based grading (pass@1) requires an external sandbox runner."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=512),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# SQuAD v2
# ---------------------------------------------------------------------------


def build_squad_v2(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("rajpurkar/squad_v2", split="validation", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for i, row in enumerate(rows):
        answers = row["answers"]["text"]
        if not answers:
            # "unanswerable" SQuAD v2 case → expect the sentinel phrase.
            prompt = (
                f"Context: {row['context']}\n\n"
                f"Question: {row['question']}\n"
                f"If the context does not answer the question, reply exactly: "
                f"'unanswerable'. Otherwise answer concisely."
            )
            test_cases.append(
                TestCase(
                    id=f"squad-v2/{row['id']}",
                    prompt=prompt,
                    expected_output="unanswerable",
                    tags=["squad-v2", "unanswerable"],
                    metrics=[
                        MetricConfig(
                            name="contains",
                            params={"substrings": ["unanswerable"], "mode": "any"},
                        )
                    ],
                )
            )
        else:
            # Pass if any acceptable answer substring appears (case-insensitive).
            prompt = (
                f"Context: {row['context']}\n\n"
                f"Question: {row['question']}\n"
                f"Answer with a short extract from the context. If the context "
                f"does not answer, reply 'unanswerable'."
            )
            # Unique, non-empty, stripped accepted answers.
            accepted = sorted({a.strip() for a in answers if a.strip()})
            test_cases.append(
                TestCase(
                    id=f"squad-v2/{row['id']}",
                    prompt=prompt,
                    expected_output=accepted[0],
                    tags=["squad-v2", "extractive-qa"],
                    metrics=[
                        MetricConfig(
                            name="contains",
                            params={"substrings": accepted, "mode": "any"},
                        )
                    ],
                )
            )
    return EvaluationSuite(
        name="squad-v2",
        description=(
            "SQuAD v2: extractive QA over paragraphs, including "
            "'unanswerable' cases. Tests reading comprehension and "
            "knowing when not to answer."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=256),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------


def build_ifeval(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("google/IFEval", split="train", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for row in rows:
        test_cases.append(
            TestCase(
                id=f"ifeval/{row['key']}",
                prompt=row["prompt"],
                tags=["ifeval", "instruction-following"] + list(row.get("instruction_id_list", [])),
                reference_data={
                    "instruction_id_list": list(row.get("instruction_id_list", [])),
                    "kwargs": row.get("kwargs", []),
                },
                metrics=[MetricConfig(name="response-capture")],
            )
        )
    return EvaluationSuite(
        name="ifeval",
        description=(
            "Google IFEval: verifiable instruction-following tasks. "
            "Uses response-capture; full scoring requires the official "
            "IFEval verifier which runs programmatic checks against each "
            "response."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=1024),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# MT-Bench prompts
# ---------------------------------------------------------------------------


def build_mt_bench(limit: int = 80, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for row in rows:
        # ``prompt`` is a 2-element list (turn-1, turn-2). Use turn-1 in v1.
        prompts = row["prompt"] if isinstance(row["prompt"], list) else [row["prompt"]]
        if not prompts:
            continue
        test_cases.append(
            TestCase(
                id=f"mt-bench/{row['prompt_id']}",
                prompt=prompts[0],
                tags=["mt-bench", str(row.get("category", "general"))],
                reference_data={"category": row.get("category"), "reference": row.get("reference")},
                metrics=[
                    MetricConfig(
                        name="llm-as-judge",
                        params={
                            "rubric": (
                                "You are a strict grader. Rate the assistant's "
                                "response on correctness, helpfulness, and writing "
                                "quality. Return a single line 'Score: X/10' with "
                                "10 = excellent and 0 = wrong or unhelpful."
                            ),
                        },
                    )
                ],
            )
        )
    return EvaluationSuite(
        name="mt-bench",
        description=(
            "MT-Bench v1 (turn-1 only): 80 open-ended prompts across "
            "8 categories (writing, roleplay, math, reasoning, coding, "
            "extraction, stem, humanities). Scored by a configurable "
            "judge model — set judge_model on the Run."
        ),
        defaults=GenerationDefaults(temperature=0.3, max_tokens=768),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# PubMedQA
# ---------------------------------------------------------------------------


def build_pubmedqa(limit: int = 300, seed: int = 42) -> EvaluationSuite:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for row in rows:
        contexts = row["context"]["contexts"] if isinstance(row.get("context"), dict) else []
        context_text = "\n".join(contexts) if contexts else ""
        expected = str(row["final_decision"]).strip().lower()
        if expected not in ("yes", "no", "maybe"):
            continue
        prompt = (
            "Read the research abstract and answer the question.\n\n"
            f"Abstract:\n{context_text}\n\n"
            f"Question: {row['question']}\n"
            f"Answer with exactly one word: yes, no, or maybe."
        )
        test_cases.append(
            TestCase(
                id=f"pubmedqa/{row['pubid']}",
                prompt=prompt,
                expected_output=expected,
                tags=["pubmedqa", "biomedical"],
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={
                            "pattern": rf"\b{expected}\b",
                            "flags": "i",
                        },
                    )
                ],
            )
        )
    return EvaluationSuite(
        name="pubmedqa",
        description=(
            "PubMedQA (labeled): biomedical yes/no/maybe QA over research "
            "abstracts. Probes domain knowledge and literal reading of "
            "scientific text."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=128),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _safe_build("bbh-mixed", build_bbh_mixed)
    _safe_build("arc-challenge", build_arc_challenge)
    _safe_build("arc-easy", build_arc_easy)
    _safe_build("piqa", build_piqa)
    _safe_build("winogrande", build_winogrande)
    _safe_build("ceval-mixed", build_ceval_mixed)
    _safe_build("math-500", build_math500)
    _safe_build("mbpp", build_mbpp)
    _safe_build("squad-v2", build_squad_v2)
    _safe_build("ifeval", build_ifeval)
    _safe_build("mt-bench", build_mt_bench)
    _safe_build("pubmedqa", build_pubmedqa)


if __name__ == "__main__":
    main()
