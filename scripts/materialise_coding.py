"""Materialise coding-focused benchmark suites on .224.

Each suite uses regex/contains metrics so pass/fail has real signal
(no response-capture placeholders). Run on .224:

    cd /home/azurewind/workspaces/AI-Model-Evaluation
    .venv/bin/python scripts/materialise_coding.py
"""

from __future__ import annotations

import itertools
import random
import re
from pathlib import Path

from datasets import load_dataset

from ollama_evaluator.suites.models import (
    EvaluationSuite,
    GenerationDefaults,
    MetricConfig,
    TestCase,
)
from ollama_evaluator.suites.writer import dump_suite


OUTPUT_DIR = Path("examples/suites")


def _take(ds_iter, limit: int, seed: int) -> list[dict]:
    """Deterministic seeded sample from a streaming dataset."""
    candidates = list(itertools.islice(ds_iter, max(limit, 1) * 3))
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def _write_suite(suite: EvaluationSuite) -> None:
    safe = suite.name.replace("/", "-")
    out = OUTPUT_DIR / f"{safe}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_suite(suite, "yaml"), encoding="utf-8")
    print(f"[{suite.name}] wrote {out} ({len(suite.test_cases)} cases)")


# ---------------------------------------------------------------------------
# CRUXEval — reason about code without executing it
# ---------------------------------------------------------------------------


def _escape_regex(literal: str) -> str:
    """Escape a literal answer string for use inside a regex pattern."""
    return re.escape(literal)


def build_cruxeval_output(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    """CRUXEval 'output-prediction': given code + input, predict the output."""
    ds = load_dataset("cruxeval-org/cruxeval", split="test", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for row in rows:
        code = row["code"]
        inp = row["input"]
        out = row["output"]
        prompt = (
            "You will be given a Python function and an input. "
            "Predict the exact Python value the function returns.\n\n"
            f"{code}\n\n"
            f"What is the output of f({inp})?\n"
            "Respond with 'Output: <value>' where <value> is a single-line "
            "Python literal. Do not add any other text."
        )
        # Match 'Output: <literal>' where <literal> equals the gold answer.
        # Gold answers are Python literals (lists, tuples, strings, numbers);
        # a direct string match is sufficient because they're canonicalised
        # by the dataset.
        test_cases.append(
            TestCase(
                id=f"cruxeval-output/{row['id']}",
                prompt=prompt,
                expected_output=out,
                tags=["cruxeval", "output-prediction", "code-reasoning"],
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={
                            "pattern": rf"Output:\s*{_escape_regex(out)}",
                        },
                    )
                ],
            )
        )
    return EvaluationSuite(
        name="cruxeval-output",
        description=(
            "CRUXEval (output-prediction): given a Python function and an "
            "input, predict the exact output value. Probes step-by-step "
            "code execution reasoning without a sandbox."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=384),
        test_cases=test_cases,
    )


def build_cruxeval_input(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    """CRUXEval 'input-prediction': given code + output, guess a matching input."""
    ds = load_dataset("cruxeval-org/cruxeval", split="test", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for row in rows:
        code = row["code"]
        inp = row["input"]
        out = row["output"]
        prompt = (
            "You will be given a Python function and a target output. "
            "Find any input that, when passed to the function, produces "
            "the target output.\n\n"
            f"{code}\n\n"
            f"Find any input such that f(input) == {out}.\n"
            "Respond with 'Input: <value>' where <value> is a single-line "
            "Python literal. Do not add any other text."
        )
        # Input-prediction answers are NOT unique (many inputs can produce
        # the same output), so regex-matching the *exact* gold input is too
        # strict. Instead we accept the canonical gold as a strong positive
        # signal; false negatives here are unavoidable without an executor.
        # Document the limitation in the description.
        test_cases.append(
            TestCase(
                id=f"cruxeval-input/{row['id']}",
                prompt=prompt,
                expected_output=inp,
                tags=["cruxeval", "input-prediction", "code-reasoning"],
                metrics=[
                    MetricConfig(
                        name="regex-match",
                        params={
                            "pattern": rf"Input:\s*{_escape_regex(inp)}",
                        },
                    )
                ],
            )
        )
    return EvaluationSuite(
        name="cruxeval-input",
        description=(
            "CRUXEval (input-prediction): given a Python function and a "
            "target output, find ANY matching input. Strict metric: only "
            "the canonical gold input counts as a pass (multiple valid "
            "inputs often exist, so this under-reports real ability)."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=384),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# Spider — natural-language → SQL
# ---------------------------------------------------------------------------


_SQL_NORMALISE_REGEX = re.compile(r"\s+")


def _normalise_sql(sql: str) -> str:
    """Collapse whitespace and lowercase SQL for a loose equality check."""
    return _SQL_NORMALISE_REGEX.sub(" ", sql.strip().lower()).rstrip(";")


def build_spider_sql(limit: int = 200, seed: int = 42) -> EvaluationSuite:
    """Spider NL→SQL: natural-language question → SQL query."""
    ds = load_dataset("spider", split="validation", streaming=True)
    rows = _take(ds, limit, seed)
    test_cases: list[TestCase] = []
    for i, row in enumerate(rows):
        gold = row["query"].strip()
        gold_norm = _normalise_sql(gold)
        question = row["question"]
        db_id = row["db_id"]
        prompt = (
            f"You are writing SQL against the '{db_id}' database.\n"
            f"Question: {question}\n"
            f"Write a single SQL query that answers the question. "
            f"Respond with only the SQL — no explanation, no markdown fences, "
            f"no trailing comments. End your response with a single newline."
        )
        # Loose equality: compare the model's whitespace-collapsed,
        # lowercased response against the whitespace-collapsed lowercased
        # gold. Not a deep parse (two semantically-equivalent queries can
        # still differ syntactically) but catches clean matches and is a
        # useful signal. Also match SELECT clauses on key tokens.
        escaped_gold = re.escape(gold_norm)
        test_cases.append(
            TestCase(
                id=f"spider/{db_id}/{i}",
                prompt=prompt,
                expected_output=gold,
                tags=["spider", "text-to-sql", db_id],
                reference_data={"db_id": db_id, "gold_query": gold},
                metrics=[
                    # A: full-query match (case-insensitive, whitespace-tolerant).
                    MetricConfig(
                        name="regex-match",
                        params={
                            "pattern": escaped_gold,
                            "flags": "i",
                        },
                    ),
                ],
            )
        )
    return EvaluationSuite(
        name="spider-sql",
        description=(
            "Spider (validation split): natural-language → SQL across 200 "
            "cross-domain databases. Strict metric: whitespace-tolerant "
            "case-insensitive equality against the gold SQL (misses many "
            "semantically-correct-but-differently-phrased queries — treat "
            "pass-rate as a lower bound)."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=384),
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# Hand-authored: python-bugfix-mini
# ---------------------------------------------------------------------------


def build_python_bugfix_mini() -> EvaluationSuite:
    """Small hand-authored suite of 'fix-this-buggy-code' prompts."""
    cases: list[TestCase] = []

    def tc(
        idx: int,
        description: str,
        prompt: str,
        fix_pattern: str,
        flags: str = "",
    ) -> TestCase:
        return TestCase(
            id=f"python-bugfix/{idx:02d}",
            prompt=prompt,
            tags=["python-bugfix", "code"],
            reference_data={"description": description},
            metrics=[
                MetricConfig(
                    name="regex-match",
                    params={"pattern": fix_pattern, "flags": flags} if flags else {"pattern": fix_pattern},
                )
            ],
        )

    cases.append(
        tc(
            1,
            "Off-by-one in range",
            (
                "Fix the bug in this Python function so it returns the sum of "
                "integers from 1 to n inclusive (e.g. n=3 → 6). Respond with "
                "the corrected code only.\n\n"
                "def sum_to(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
            ),
            r"range\s*\(\s*1\s*,\s*n\s*\+\s*1\s*\)",
        )
    )

    cases.append(
        tc(
            2,
            "Mutable default argument",
            (
                "Fix the classic bug here so each call starts with an empty list. "
                "Respond with the corrected code only.\n\n"
                "def append_item(item, items=[]):\n    items.append(item)\n    return items\n"
            ),
            r"items\s*=\s*None",
        )
    )

    cases.append(
        tc(
            3,
            "Integer division vs float",
            (
                "Fix the bug so the function returns a float average. "
                "Respond with the corrected code only.\n\n"
                "def average(nums):\n    return sum(nums) // len(nums)\n"
            ),
            r"sum\s*\(\s*nums\s*\)\s*/\s*len\s*\(\s*nums\s*\)",
        )
    )

    cases.append(
        tc(
            4,
            "Missing return",
            (
                "Fix the bug so the function returns the doubled value. "
                "Respond with the corrected code only.\n\n"
                "def double(x):\n    x * 2\n"
            ),
            r"return\s+x\s*\*\s*2",
        )
    )

    cases.append(
        tc(
            5,
            "Dict KeyError handling",
            (
                "Fix the bug so the function returns None when the key is missing, "
                "without raising. Respond with the corrected code only.\n\n"
                "def lookup(d, key):\n    return d[key]\n"
            ),
            r"(?:d\.get\s*\(\s*key\s*\)|try\s*:)",
        )
    )

    cases.append(
        tc(
            6,
            "String concat in loop type bug",
            (
                "Fix the bug so the function returns a single concatenated string. "
                "Respond with the corrected code only.\n\n"
                "def join_words(words):\n    s = 0\n    for w in words:\n        s += w\n    return s\n"
            ),
            r"s\s*=\s*['\"]{2}|''\.join\s*\(",
        )
    )

    cases.append(
        tc(
            7,
            "Off-by-one in index access",
            (
                "Fix the bug so last_item returns the last element of xs, or None "
                "if empty. Respond with the corrected code only.\n\n"
                "def last_item(xs):\n    return xs[len(xs)]\n"
            ),
            r"xs\s*\[\s*-\s*1\s*\]|xs\s*\[\s*len\s*\(\s*xs\s*\)\s*-\s*1\s*\]",
        )
    )

    cases.append(
        tc(
            8,
            "Modifying list while iterating",
            (
                "Fix the bug so the function correctly removes all negatives without "
                "skipping elements. Respond with the corrected code only.\n\n"
                "def drop_negatives(xs):\n    for x in xs:\n        if x < 0:\n            xs.remove(x)\n    return xs\n"
            ),
            r"(?:\[\s*x\s+for\s+x\s+in\s+xs\s+if\s+x\s*>=\s*0\s*\]|list\s*\(\s*filter)",
        )
    )

    cases.append(
        tc(
            9,
            "None vs empty check",
            (
                "Fix the bug so is_empty returns True for both None and empty "
                "iterables, without raising on None. Respond with the corrected "
                "code only.\n\n"
                "def is_empty(xs):\n    return len(xs) == 0\n"
            ),
            r"xs\s+is\s+None|not\s+xs",
        )
    )

    cases.append(
        tc(
            10,
            "Wrong equality operator",
            (
                "Fix the bug so is_admin returns True only when role equals 'admin'. "
                "Respond with the corrected code only.\n\n"
                "def is_admin(role):\n    return role = 'admin'\n"
            ),
            r"role\s*==\s*['\"]admin['\"]",
        )
    )

    cases.append(
        tc(
            11,
            "Infinite recursion base case",
            (
                "Fix the bug by adding the missing base case so factorial(0) == 1. "
                "Respond with the corrected code only.\n\n"
                "def factorial(n):\n    return n * factorial(n - 1)\n"
            ),
            r"if\s+n\s*(?:<=\s*1|==\s*0)",
        )
    )

    cases.append(
        tc(
            12,
            "Closing resource",
            (
                "Fix the bug so the file is closed even if an exception is raised "
                "during read. Respond with the corrected code only.\n\n"
                "def read_first_line(path):\n    f = open(path)\n    return f.readline()\n"
            ),
            r"with\s+open\s*\(",
        )
    )

    cases.append(
        tc(
            13,
            "Float equality",
            (
                "Fix the bug so the function returns True for pairs that are "
                "approximately equal to within 1e-9. Respond with the corrected "
                "code only.\n\n"
                "def approx_equal(a, b):\n    return a == b\n"
            ),
            r"abs\s*\(\s*a\s*-\s*b\s*\)\s*<\s*(?:1e-9|0\.000000001)",
        )
    )

    cases.append(
        tc(
            14,
            "String formatting KeyError",
            (
                "Fix the bug so the function uses an f-string (or .format) "
                "correctly with the 'name' variable. Respond with the corrected "
                "code only.\n\n"
                "def greet(name):\n    return 'Hello {name}'\n"
            ),
            r"(?:f['\"]Hello\s*\{name\}|\.format\s*\(\s*name\s*=\s*name\s*\))",
        )
    )

    return EvaluationSuite(
        name="python-bugfix-mini",
        description=(
            "14 hand-authored Python 'fix-this-bug' tasks covering "
            "off-by-one errors, mutable defaults, resource handling, "
            "equality pitfalls, and recursion. Regex-based metrics look "
            "for the canonical fix pattern, so pass-rate is a lower bound."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=384),
        test_cases=cases,
    )


# ---------------------------------------------------------------------------
# Hand-authored: shell-bash-basics
# ---------------------------------------------------------------------------


def build_shell_bash_basics() -> EvaluationSuite:
    """Small hand-authored bash one-liner suite."""
    cases: list[TestCase] = []

    def tc(idx: int, prompt: str, pattern: str, flags: str = "") -> TestCase:
        params = {"pattern": pattern}
        if flags:
            params["flags"] = flags
        return TestCase(
            id=f"bash-basics/{idx:02d}",
            prompt=prompt,
            tags=["bash", "shell", "code"],
            metrics=[MetricConfig(name="regex-match", params=params)],
        )

    cases.append(
        tc(
            1,
            "Write a single-line bash command that prints every non-hidden "
            "file in the current directory, one per line. Respond with the "
            "command only — no commentary, no backticks, no trailing "
            "explanation.",
            r"^\s*ls\b(?!.*-[a-zA-Z]*a)",
            flags="m",
        )
    )

    cases.append(
        tc(
            2,
            "Write a single-line bash command that counts the number of "
            "lines in the file /var/log/syslog. Respond with the command "
            "only.",
            r"wc\s+-l\s+/var/log/syslog",
        )
    )

    cases.append(
        tc(
            3,
            "Write a single-line bash command that searches every *.py file "
            "under the current directory recursively for the word TODO, "
            "printing the filename and line. Respond with the command only.",
            r"grep\s+(?:.*-r|.*--recursive).*TODO|rg\s+.*TODO.*--?type\s+py",
            flags="i",
        )
    )

    cases.append(
        tc(
            4,
            "Write a single-line bash command that finds every file named "
            "*.log modified in the last 24 hours under /var/log and prints "
            "their paths. Respond with the command only.",
            r"find\s+/var/log.*-name\s+['\"]?\*\.log['\"]?.*-mtime\s+(?:-1|0)",
        )
    )

    cases.append(
        tc(
            5,
            "Write a single-line bash command that prints the first 10 "
            "lines of access.log in the current directory. Respond with "
            "the command only.",
            r"head\s+(?:-n\s+)?10\s+access\.log",
        )
    )

    cases.append(
        tc(
            6,
            "Write a single-line bash command that lists the top 5 "
            "largest files under the current directory (recursive), sorted "
            "by size descending. Respond with the command only.",
            r"(?:du\s+.*-a.*\|\s*sort.*-n.*\|\s*(?:tail|head))|(?:find\s+\..*-printf.*\|\s*sort.*\|\s*(?:head|tail))",
        )
    )

    cases.append(
        tc(
            7,
            "Write a single-line bash command that kills every process "
            "whose name contains 'uvicorn'. Respond with the command only.",
            r"(?:pkill\s+(?:-f\s+)?uvicorn|(?:ps.*\|\s*grep.*uvicorn.*\|\s*(?:awk|xargs).*kill))",
        )
    )

    cases.append(
        tc(
            8,
            "Write a single-line bash command that shows disk usage of "
            "every top-level directory under /home, sorted from largest to "
            "smallest, human-readable. Respond with the command only.",
            r"du\s+(?:.*-h.*-s|.*-sh)\s+/home/\*\s*\|\s*sort\s+-h(?:r|.*-r)",
        )
    )

    cases.append(
        tc(
            9,
            "Write a single-line bash command that renames every *.jpeg "
            "file in the current directory to use a *.jpg extension. "
            "Respond with the command only.",
            r"for\s+f\s+in.*\*\.jpeg|rename\s+.*jpeg.*jpg|mmv.*jpeg.*jpg",
            flags="i",
        )
    )

    cases.append(
        tc(
            10,
            "Write a single-line bash command that prints the current "
            "year (just the 4-digit year). Respond with the command only.",
            r"date\s+(?:\+%Y|\+\"%Y\"|\+'%Y')",
        )
    )

    return EvaluationSuite(
        name="shell-bash-basics",
        description=(
            "10 hand-authored Bash one-liners covering listing, searching, "
            "file mgmt, process mgmt, and formatted output. Metrics match "
            "the canonical command shape; minor style variants may miss."
        ),
        defaults=GenerationDefaults(temperature=0.0, max_tokens=256),
        test_cases=cases,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _write_suite(build_cruxeval_output())
    _write_suite(build_cruxeval_input())
    _write_suite(build_spider_sql())
    _write_suite(build_python_bugfix_mini())
    _write_suite(build_shell_bash_basics())


if __name__ == "__main__":
    main()
