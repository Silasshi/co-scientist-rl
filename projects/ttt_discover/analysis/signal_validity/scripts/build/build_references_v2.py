#!/usr/bin/env python3
"""Build references_v2.jsonl from shared/papers/by_topic/*/*/{research_goal.txt,reference_plan.txt}."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
PAPERS = PROJECT_ROOT / "shared" / "papers" / "by_topic"
OUT = Path(__file__).parent / "references_v2.jsonl"

refs = []
for plan_file in PAPERS.rglob("reference_plan.txt"):
    paper_dir = plan_file.parent
    goal_file = paper_dir / "research_goal.txt"
    if not goal_file.exists():
        print(f"SKIP (no goal): {paper_dir.name}")
        continue
    topic = paper_dir.parent.name
    shortname = paper_dir.name
    refs.append({
        "source": f"paper_{topic}",
        "source_id": shortname,
        "subdomain": topic,
        "goal": goal_file.read_text().strip(),
        "reference_solution": plan_file.read_text().strip(),
    })

refs.sort(key=lambda r: (r["subdomain"], r["source_id"]))
with open(OUT, "w") as f:
    for r in refs:
        f.write(json.dumps(r) + "\n")
print(f"Wrote {len(refs)} references to {OUT}")

# Summary
from collections import Counter
c = Counter(r["subdomain"] for r in refs)
for topic, n in sorted(c.items()):
    print(f"  {topic}: {n}")
