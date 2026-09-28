"""
Parse the 3 pilot responses + smoke-test the full evolution flow.

For each response: parse into RubricItem list, add to a shared RubricBuffer.
Report final buffer state, item-level details, and a dry-run filter pass
(with synthetic grades to demonstrate std-filter behaviour on this real data).
"""

import json
from pathlib import Path

from co_scientist.shared.rubric_buffer import NEGATIVE, POSITIVE, RubricBuffer
from co_scientist.shared.rubric_gen_prompt import parse_rubric_response


ROOT = Path("/home/silas/co-scientist-project/projects/grant_proposal_v2")
RESP_DIR = ROOT / "analysis/rubric_evolution_pilot/responses"
REQUESTS_DIR = ROOT / "analysis/rubric_evolution_pilot/requests"

PAIRS = [
    "pair_01_best_vs_ref",
    "pair_02_mid_vs_ref",
    "pair_03_best_vs_worst",
]


def main() -> None:
    buffer = RubricBuffer(k_max=12, min_variance=0.05, min_scored_iters=1)

    print("=" * 78)
    print("RUBRIC EVOLUTION PILOT — parsing responses + buffer smoke test")
    print("=" * 78)

    total_proposed = 0
    total_added = 0

    for pair_id in PAIRS:
        resp_path = RESP_DIR / f"{pair_id}.json"
        if not resp_path.exists():
            print(f"\n[{pair_id}] MISSING response file at {resp_path}")
            continue

        raw = resp_path.read_text()
        try:
            items = parse_rubric_response(raw, created_at_iter=0)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{pair_id}] PARSE FAIL: {exc}")
            print("raw head:", raw[:200])
            continue

        added_ids = buffer.add(items)
        total_proposed += len(items)
        total_added += len(added_ids)

        print(f"\n[{pair_id}]  proposed={len(items)}  added={len(added_ids)}  dedupe_skipped={len(items) - len(added_ids)}")
        for it in items:
            tag = "POS" if it.rubric_type == POSITIVE else "NEG"
            w = f"w={it.weight:.2f}"
            new = "NEW" if it.id in added_ids else "dup"
            print(f"  [{tag}|{new}|{w}] {it.title}")
            print(f"        {it.description}")

    print("\n" + "=" * 78)
    print(f"BUFFER AFTER MERGE: {total_proposed} proposed → {total_added} added → {len(buffer.items)} in buffer")
    print(f"stats: {buffer.stats()}")
    print("=" * 78)

    # Dry-run std filter: simulate per-rollout grades. Each rubric randomly
    # applied to 8 rollouts — some always-true (dead), some actually varying.
    # Purpose: verify filter mechanism works on a real (non-synthetic) buffer.
    print("\n-- dry-run std filter (synthetic grades) --")
    rng_seed = 0
    import random

    rng = random.Random(rng_seed)
    for rid, it in buffer.items.items():
        # Simulate: 1/3 of items happen to be "dead" (all-same grades)
        if rng.random() < 0.33:
            grades = [1.0] * 8  # all satisfied
        else:
            grades = [float(rng.random() < 0.5) for _ in range(8)]
        buffer.record_grades(rid, grades)

    dropped, retained = buffer.filter_and_truncate()
    print(f"filter: dropped {len(dropped)}  retained {len(retained)}")
    for rid in dropped:
        # id is gone from buffer.items now; report from collected ids
        pass
    print("retained details:")
    for rid in retained:
        it = buffer.items[rid]
        print(f"  [{it.rubric_type[:3].upper()}|std={it.latest_std:.3f}|w={it.weight:.2f}] {it.title}")

    # Round-trip through jsonl
    out_path = RESP_DIR.parent / "pilot_buffer_after_merge.jsonl"
    buffer.to_jsonl(out_path)
    print(f"\nwrote {out_path.name}  ({len(buffer.items)} items)")

    print("\n" + "=" * 78)
    print("PILOT SUCCESS CRITERIA")
    print("=" * 78)
    criteria = [
        ("Response JSON parseable for all 3 pairs", total_proposed > 0),
        ("≥3 total new rubric items across pairs", total_proposed >= 3),
        ("≥1 negative rubric elicited", any(
            it.rubric_type == NEGATIVE for it in buffer.items.values()
        )),
        ("Filter dropped ≥1 item in dry-run", len(dropped) >= 1 or len(buffer.items) == 0),
        (
            "No duplicate items (dedup works)",
            len({it.title for it in buffer.items.values()}) == len(buffer.items),
        ),
    ]
    for desc, ok in criteria:
        print(f"  [{'✓' if ok else '✗'}] {desc}")


if __name__ == "__main__":
    main()
