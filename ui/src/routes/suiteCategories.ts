/**
 * Client-side suite → category mapping.
 *
 * Categories are a UI-only concept. We group suites in the picker so
 * users can browse by capability (reasoning / coding / multilingual /
 * safety / etc.) instead of alphabetically. The server does not need
 * to know about categories — a new suite just appears in the
 * ``"Other"`` bucket until we update this file.
 *
 * Ordering of :data:`CATEGORIES` is the display order in the UI.
 * Ordering of suite names within each category is alphabetical at
 * render time.
 *
 * Short summaries
 * ---------------
 * :func:`summaryForSuite` returns a ≤70-character one-liner used as
 * the inline description in the picker. It is intentionally shorter
 * than the backend ``description`` (which is verbose and includes
 * methodology caveats). The backend description still renders in the
 * hover tooltip so users can get the full context when they want it.
 */

export type SuiteCategory =
  | "Reasoning"
  | "Knowledge"
  | "Coding"
  | "Instruction"
  | "Math"
  | "Multilingual"
  | "Long context"
  | "Safety"
  | "Open-ended"
  | "Other";

export const CATEGORIES: SuiteCategory[] = [
  "Reasoning",
  "Knowledge",
  "Coding",
  "Math",
  "Instruction",
  "Multilingual",
  "Long context",
  "Safety",
  "Open-ended",
  "Other",
];

/**
 * Map a suite name to its display category. Returns ``"Other"`` for
 * unknown names so new suites are never hidden from the picker.
 */
export function categoryForSuite(name: string): SuiteCategory {
  return SUITE_CATEGORY[name] ?? "Other";
}

/**
 * Map a suite name to a short (≤70 chars), single-line summary used
 * inline in the suite picker. Falls back to the first sentence of the
 * upstream description when a hand-authored summary is not available,
 * or to the suite name as a last resort.
 */
export function summaryForSuite(
  name: string,
  fallbackDescription?: string | null,
): string {
  const s = SUITE_SUMMARY[name];
  if (s) return s;
  if (fallbackDescription) {
    // First-sentence slice as a cheap fallback for future suites that
    // haven't been catalogued here yet. Stops at ``. `` / ``! `` /
    // ``? `` or at the sentence-ending newline.
    const match = fallbackDescription.match(/^(.+?[.!?])(\s|$)/);
    if (match) return match[1]!.trim();
    return fallbackDescription.slice(0, 70).trim() + "…";
  }
  return name;
}

const SUITE_CATEGORY: Record<string, SuiteCategory> = {
  // Reasoning
  "reasoning-basics": "Reasoning",
  "reasoning-advanced": "Reasoning",
  "bbh-mixed": "Reasoning",
  "hellaswag": "Reasoning",
  "piqa": "Reasoning",
  "winogrande": "Reasoning",
  "cruxeval-input": "Reasoning",
  "cruxeval-output": "Reasoning",

  // Knowledge / factual
  "factual-qa": "Knowledge",
  "mmlu": "Knowledge",
  "ceval-mixed": "Knowledge",
  "truthfulqa-mc1": "Knowledge",
  "arc-challenge": "Knowledge",
  "arc-easy": "Knowledge",
  "pubmedqa": "Knowledge",
  "squad-v2": "Knowledge",

  // Coding
  "code-generation-basics": "Coding",
  "humaneval": "Coding",
  "mbpp": "Coding",
  "python-bugfix-mini": "Coding",
  "shell-bash-basics": "Coding",
  "spider-sql": "Coding",

  // Math
  "math-word-problems": "Math",
  "gsm8k": "Math",
  "math-500": "Math",

  // Instruction
  "instruction-following": "Instruction",
  "json-output": "Instruction",
  "ifeval": "Instruction",

  // Multilingual
  "multilingual-basic": "Multilingual",

  // Long context
  "long-context-probe": "Long context",

  // Safety
  "safety-refusal": "Safety",

  // Open-ended (judge-scored)
  "llm-as-judge-general": "Open-ended",
  "mt-bench": "Open-ended",
};

/**
 * Hand-authored one-liners. Keep to ≤70 characters so they fit on one
 * line in the picker without wrapping. The full backend description
 * still appears in the hover tooltip.
 */
const SUITE_SUMMARY: Record<string, string> = {
  // Reasoning
  "reasoning-basics": "Tiny smoke suite — 3 cases for quick sanity checks.",
  "reasoning-advanced": "Logic puzzles and short multi-step reasoning.",
  "bbh-mixed": "BigBench-Hard: 8 difficult reasoning subsets.",
  "hellaswag": "Commonsense sentence-completion.",
  "piqa": "Everyday physical-world commonsense (A vs B).",
  "winogrande": "Pronoun resolution with real-world context.",
  "cruxeval-input": "Given code + output, predict the input.",
  "cruxeval-output": "Given code + input, predict the exact output.",

  // Knowledge
  "factual-qa": "Short-answer facts across geography, science, history.",
  "mmlu": "57-subject academic multiple-choice.",
  "ceval-mixed": "Chinese academic MCQ across 6 subjects.",
  "truthfulqa-mc1": "Resists popular misconceptions (MC1).",
  "arc-challenge": "Harder grade-school science MCQ.",
  "arc-easy": "Easier grade-school science MCQ (good baseline).",
  "pubmedqa": "Biomedical yes/no/maybe over research abstracts.",
  "squad-v2": "Extractive reading including unanswerable cases.",

  // Coding
  "code-generation-basics": "Python one-liners and small functions.",
  "humaneval": "Python function-writing (capture-only — needs sandbox).",
  "mbpp": "Mostly Basic Python Problems (capture-only).",
  "python-bugfix-mini": "14 hand-authored Python 'fix-this-bug' prompts.",
  "shell-bash-basics": "10 hand-authored Bash one-liners.",
  "spider-sql": "Natural-language → SQL across 200 databases.",

  // Math
  "math-word-problems": "Multi-step arithmetic, percentages, algebra.",
  "gsm8k": "Grade-school math word problems with numeric answers.",
  "math-500": "Competition math (capture-only; needs grader).",

  // Instruction
  "instruction-following": "Obeys format, length, and constraint rules.",
  "json-output": "Structured JSON matching a given schema.",
  "ifeval": "Verifiable instruction tasks (capture-only).",

  // Multilingual
  "multilingual-basic": "Same question in EN / FR / ES / DE / ZH / JA / KO.",

  // Long context
  "long-context-probe": "Needle-in-haystack retrieval from long passages.",

  // Safety
  "safety-refusal": "Appropriately refuses clearly harmful requests.",

  // Open-ended
  "llm-as-judge-general": "Open-ended answers graded by a judge model.",
  "mt-bench": "MT-Bench turn-1 prompts, judge-scored.",
};
