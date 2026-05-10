"""Read the most recent Run_Report JSON under examples/runs/ and summarise it.

Used from the Windows control host via ``ssh ... python3 scripts/remote_show_report.py``
so the output can be harvested without fighting PowerShell/ssh/bash escaping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS_DIR = Path("examples/runs")


def main() -> int:
    if not RUNS_DIR.is_dir():
        print(f"ERR: {RUNS_DIR} missing")
        return 1
    reports = sorted(RUNS_DIR.glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        print("ERR: no report.json found")
        return 1
    path = reports[-1]
    print(f"== Latest report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"run_id      : {payload['run_id']}")
    print(f"status      : {payload['status']}")
    print(f"started_at  : {payload['started_at']}")
    print(f"ended_at    : {payload['ended_at']}")
    print(f"models      : {[m['name'] for m in payload['models']]}")
    print(f"suites      : {payload['config']['run']['suites']}")
    print()
    print("== Executions:")
    for r in payload["results"]:
        resp = (r["response"] or "").replace("\n", " ")
        if len(resp) > 140:
            resp = resp[:140] + "..."
        print(
            f"  [{r['status']}] {r['test_case_id']} "
            f"total={r['performance']['total_ms']:.0f}ms "
            f"ttft={r['performance']['ttft_ms']}ms "
            f"resp_tokens={r['performance']['response_tokens']} "
            f"tps={r['performance']['tokens_per_second']}"
        )
        print(f"           response: {resp!r}")
        for m in r["metrics"]:
            err = f" error={m.get('error')!r}" if m.get("error") else ""
            print(
                f"           metric[{m['name']}] score={m['score']} "
                f"passed={m['passed']}{err}"
            )
    print()
    print("== Aggregates:")
    for a in payload["aggregates"]:
        print(
            f"  model={a['model']} passed={a['passed']} failed={a['failed']} "
            f"error={a['errored']} timeout={a['timed_out']} "
            f"mean_total_ms={a['mean_total_ms']:.1f} "
            f"mean_tps={a['mean_tokens_per_second']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
