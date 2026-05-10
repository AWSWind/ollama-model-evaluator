#!/bin/bash
cd /home/azurewind/workspaces/AI-Model-Evaluation || exit 1

echo "=== ui/dist ==="
ls -la ui/dist/assets/

echo
echo "=== summaries stats ==="
curl -sS http://127.0.0.1:8765/api/suites/summaries > /tmp/summaries.json
python3 <<'PY'
import json
d = json.load(open("/tmp/summaries.json"))
print(f"suites total: {len(d)}")
print(f"total cases : {sum(x['test_case_count'] for x in d)}")
print(f"with descr. : {sum(1 for x in d if x.get('description'))}")
print()
print("Per-suite breakdown:")
for s in d:
    desc_fragment = (s.get("description") or "—")[:60].replace("\n", " ")
    print(f"  {s['name']:<28} {s['test_case_count']:>4} cases   {desc_fragment}")
PY
rm -f /tmp/summaries.json
