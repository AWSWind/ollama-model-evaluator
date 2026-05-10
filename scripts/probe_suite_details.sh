#!/bin/bash
for s in code-generation-basics factual-qa gsm8k hellaswag humaneval instruction-following json-output llm-as-judge-general long-context-probe math-word-problems mmlu multilingual-basic reasoning-advanced reasoning-basics safety-refusal truthfulqa-mc1; do
  encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$s', safe=''))")
  cnt=$(curl -sS "http://127.0.0.1:8765/api/suites/$encoded" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("test_cases", [])))' 2>/dev/null)
  printf "%-28s %s cases\n" "$s" "$cnt"
done
