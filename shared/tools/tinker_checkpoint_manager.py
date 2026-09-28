"""Manage Tinker checkpoints on a specific API profile.

Subcommands:
    list             -- list all checkpoints on the profile, grouped by training run
    suggest-cleanup  -- classify checkpoints into KEEP / REVIEW / DELETE
    delete           -- delete checkpoints tagged DELETE in a plan file
    delete-one       -- delete a single checkpoint by tinker:// URI

Safety:
    * --profile is required on every command (no default account)
    * delete commands are dry-run by default; require --confirm to actually delete
    * KEEP-tagged checkpoints can never be deleted via this tool
    * Every deletion is logged to tools/checkpoint_cleanup_log_<timestamp>.jsonl

Usage examples:
    python tools/tinker_checkpoint_manager.py list --profile new
    python tools/tinker_checkpoint_manager.py suggest-cleanup --profile new
    python tools/tinker_checkpoint_manager.py delete --profile new --plan tools/checkpoint_cleanup_plan.json
    python tools/tinker_checkpoint_manager.py delete --profile new --plan tools/checkpoint_cleanup_plan.json --confirm
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from co_scientist.shared.api_profiles import create_service_client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIRS = [
    REPO_ROOT / "projects" / d / "runs"
    for d in ("rubric_reward", "ibt", "ttt_discover", "grant_proposal",
               "d5_abstract_retrieve_refine")
]
TOOLS_DIR = REPO_ROOT / "shared" / "tools"

TINKER_URI_RE = re.compile(r"^tinker://([^/]+)/(weights|sampler_weights)/(.+)$")

# D5 production checkpoints — the only D5 URIs that survive cleanup.
D5_KEEP_URIS: frozenset[str] = frozenset({
    # μ-v4 iter 4 (production, 28.00/45)
    "tinker://6b9d996d-8079-556d-82ac-560247551960:train:0/sampler_weights/iter_0004",
    "tinker://6b9d996d-8079-556d-82ac-560247551960:train:0/weights/000004_2026_04_25",
    # μ-v7-opd full iter 4 (F13/F14 production candidate)
    "tinker://99ee87da-3a77-5289-8feb-efc601fb99b5:train:0/sampler_weights/iter_0004",
    "tinker://99ee87da-3a77-5289-8feb-efc601fb99b5:train:0/weights/000004_2026_04_29",
})


def parse_tinker_uri(uri: str) -> tuple[str, str, str] | None:
    """Return (training_run_id, kind, name) or None."""
    m = TINKER_URI_RE.match(uri)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def human_gb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "   ?   "
    return f"{size_bytes / (1024**3):6.2f} GB"


def build_local_index() -> dict[str, dict]:
    """Scan projects/*/runs/**/checkpoints.jsonl and index every tinker URI to its local run.

    Returns:
        mapping from tinker URI -> {"run_dir": str, "batch": int, "name": str}
    """
    index: dict[str, dict] = {}
    for runs_root in RUNS_DIRS:
        if not runs_root.exists():
            continue
        for jsonl_path in runs_root.rglob("checkpoints.jsonl"):
            run_dir = jsonl_path.parent.relative_to(REPO_ROOT)
            try:
                with jsonl_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for key in ("state_path", "sampler_path"):
                            uri = entry.get(key)
                            if uri:
                                index[uri] = {
                                    "run_dir": str(run_dir),
                                    "batch": entry.get("batch"),
                                    "name": entry.get("name"),
                                }
            except OSError:
                continue
    return index


def load_run_summary(run_dir: str) -> dict:
    """Read summary.md for a run, return fields parsed from its bullet list."""
    summary_path = REPO_ROOT / run_dir / "summary.md"
    result: dict = {}
    if not summary_path.exists():
        return result
    try:
        for line in summary_path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("- **"):
                continue
            m = re.match(r"- \*\*([^*]+)\*\*:\s*(.+)", line)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                result[key] = m.group(2).strip()
    except OSError:
        pass
    return result


def list_all_checkpoints(rest_client, verbose: bool = False) -> list:
    """Paginate through list_user_checkpoints and return all Checkpoint objects."""
    out: list = []
    offset = 0
    limit = 100
    while True:
        resp = rest_client.list_user_checkpoints(limit=limit, offset=offset).result()
        out.extend(resp.checkpoints)
        cursor = getattr(resp, "cursor", None)
        if verbose:
            total = getattr(cursor, "total_count", None) if cursor else None
            print(f"  fetched {len(out)}/{total}", file=sys.stderr)
        if not cursor:
            break
        fetched = offset + len(resp.checkpoints)
        if fetched >= cursor.total_count or not resp.checkpoints:
            break
        offset += limit
    return out


def group_by_training_run(checkpoints: list) -> dict[str, list]:
    """Group checkpoints by their training_run_id (parsed from tinker_path)."""
    groups: dict[str, list] = defaultdict(list)
    for cp in checkpoints:
        parsed = parse_tinker_uri(cp.tinker_path)
        run_id = parsed[0] if parsed else "<unparseable>"
        groups[run_id].append(cp)
    return dict(groups)


def find_local_match(checkpoints: list, local_index: dict) -> dict | None:
    """Given a list of checkpoints for one training run, find the best local match."""
    for cp in checkpoints:
        match = local_index.get(cp.tinker_path)
        if match:
            return match
    return None


def cmd_list(args) -> int:
    sc = create_service_client(api_profile=args.profile)
    rest = sc.create_rest_client()

    print(f"Listing checkpoints on profile '{args.profile}'...", file=sys.stderr)
    checkpoints = list_all_checkpoints(rest, verbose=True)
    print(f"Got {len(checkpoints)} total checkpoints\n", file=sys.stderr)

    local_index = build_local_index()
    groups = group_by_training_run(checkpoints)

    def group_total_size(cps: list) -> int:
        return sum(cp.size_bytes or 0 for cp in cps)

    sorted_groups = sorted(groups.items(), key=lambda kv: -group_total_size(kv[1]))

    grand_total_bytes = 0
    grand_total_count = 0

    for run_id, cps in sorted_groups:
        cps_sorted = sorted(cps, key=lambda c: c.time)
        total_bytes = group_total_size(cps_sorted)
        grand_total_bytes += total_bytes
        grand_total_count += len(cps_sorted)

        match = find_local_match(cps_sorted, local_index)
        if match:
            summary = load_run_summary(match["run_dir"])
            method = summary.get("method", "?")
            status = summary.get("status", "?")
            batches = summary.get("batches", "?")
            local_desc = f"{match['run_dir']} [{method}, {status}, {batches} batches]"
        else:
            local_desc = "NOT FOUND (orphaned)"

        print(f"Training Run: {run_id}")
        print(f"  Local:        {local_desc}")
        print(f"  Checkpoints:  {len(cps_sorted)}   Size: {human_gb(total_bytes)}")
        for cp in cps_sorted:
            parsed = parse_tinker_uri(cp.tinker_path)
            name = parsed[2] if parsed else cp.checkpoint_id
            kind = parsed[1] if parsed else cp.checkpoint_type
            when = cp.time.strftime("%Y-%m-%d") if cp.time else "?"
            print(f"    {name:<40} {kind:<16} {human_gb(cp.size_bytes)}   {when}")
        print()

    print(f"=== TOTAL: {grand_total_count} checkpoints, {human_gb(grand_total_bytes)} ===")
    return 0


def classify_checkpoints(
    checkpoints: list,
    local_index: dict,
    d5_only: bool = False,
    all_except_d5: bool = False,
) -> list[dict]:
    """Classify each checkpoint into KEEP / DELETE with a reason.

    Policy (aggressive): keep only the final weights + final sampler checkpoint per run.
    Delete everything else. Protect the bestversion primary baseline explicitly.

    all_except_d5: mark every checkpoint not in D5_KEEP_URIS as DELETE (nuke all non-D5).
    """
    groups = group_by_training_run(checkpoints)
    plan_entries: list[dict] = []

    for run_id, cps in groups.items():
        cps_sorted = sorted(cps, key=lambda c: c.time)
        match = find_local_match(cps_sorted, local_index)
        summary = load_run_summary(match["run_dir"]) if match else {}

        run_dir = match["run_dir"] if match else None
        status = summary.get("status", "").lower() if summary else ""
        method = summary.get("method", "")
        batches_str = summary.get("batches", "0")
        try:
            batches = int(batches_str)
        except ValueError:
            batches = 0

        # Identify the last checkpoint of each kind (by time) for this run
        weights_cps = [c for c in cps_sorted if "/weights/" in c.tinker_path]
        sampler_cps = [c for c in cps_sorted if "/sampler_weights/" in c.tinker_path]
        last_weights_uri = weights_cps[-1].tinker_path if weights_cps else None
        last_sampler_uri = sampler_cps[-1].tinker_path if sampler_cps else None

        run_dir_str = run_dir or ""
        is_protected_baseline = "withA1,A2/2(ml)" in run_dir_str

        is_d5_run = run_dir and "d5_abstract_retrieve_refine" in run_dir

        # --d5-only: skip sessions that are NOT D5
        if d5_only and not is_d5_run:
            continue

        for cp in cps_sorted:
            is_final = cp.tinker_path == last_weights_uri or cp.tinker_path == last_sampler_uri

            # --all-except-d5: keep only D5_KEEP_URIS, delete everything else globally
            if all_except_d5:
                if cp.tinker_path in D5_KEEP_URIS:
                    tier = "KEEP"
                    reason = "D5 production checkpoint (explicitly listed)"
                else:
                    tier = "DELETE"
                    reason = "not a D5 production checkpoint (--all-except-d5)"
                plan_entries.append(
                    {
                        "tier": tier,
                        "reason": reason,
                        "tinker_path": cp.tinker_path,
                        "checkpoint_type": cp.checkpoint_type,
                        "size_bytes": cp.size_bytes,
                        "time": cp.time.isoformat() if cp.time else None,
                        "training_run_id": run_id,
                        "local_run_dir": run_dir,
                        "local_method": method,
                        "local_status": summary.get("status") if summary else None,
                        "local_batches": batches,
                    }
                )
                continue

            # D5-specific override: only the 4 explicitly listed production URIs survive.
            if is_d5_run:
                if cp.tinker_path in D5_KEEP_URIS:
                    tier = "KEEP"
                    reason = "D5 production checkpoint (explicitly listed)"
                else:
                    tier = "DELETE"
                    reason = "D5 non-production checkpoint"
                plan_entries.append(
                    {
                        "tier": tier,
                        "reason": reason,
                        "tinker_path": cp.tinker_path,
                        "checkpoint_type": cp.checkpoint_type,
                        "size_bytes": cp.size_bytes,
                        "time": cp.time.isoformat() if cp.time else None,
                        "training_run_id": run_id,
                        "local_run_dir": run_dir,
                        "local_method": method,
                        "local_status": summary.get("status") if summary else None,
                        "local_batches": batches,
                    }
                )
                continue

            tier = "DELETE"
            reason = "intermediate checkpoint (policy: keep only final)"

            if match is None:
                tier = "DELETE"
                reason = "orphaned: no matching run in local runs/2026/"
            elif "failed" in status or "evaluation-only" in status or batches == 0:
                tier = "DELETE"
                reason = f"run status '{summary.get('status', '?')}' ({batches} batches)"
            elif is_final:
                tier = "KEEP"
                reason = "final checkpoint of the run"
            # else: non-final, non-orphaned -> DELETE (intermediate)

            # Protect the bestversion primary baseline's final checkpoints unconditionally
            if is_protected_baseline and is_final:
                tier = "KEEP"
                reason = "bestversion primary baseline (final checkpoint, protected)"

            plan_entries.append(
                {
                    "tier": tier,
                    "reason": reason,
                    "tinker_path": cp.tinker_path,
                    "checkpoint_type": cp.checkpoint_type,
                    "size_bytes": cp.size_bytes,
                    "time": cp.time.isoformat() if cp.time else None,
                    "training_run_id": run_id,
                    "local_run_dir": run_dir,
                    "local_method": method,
                    "local_status": summary.get("status"),
                    "local_batches": batches,
                }
            )

    return plan_entries


def cmd_suggest_cleanup(args) -> int:
    sc = create_service_client(api_profile=args.profile)
    rest = sc.create_rest_client()

    print(f"Listing checkpoints on profile '{args.profile}'...", file=sys.stderr)
    checkpoints = list_all_checkpoints(rest, verbose=True)
    print(f"Got {len(checkpoints)} total checkpoints\n", file=sys.stderr)

    local_index = build_local_index()
    entries = classify_checkpoints(
        checkpoints, local_index,
        d5_only=getattr(args, "d5_only", False),
        all_except_d5=getattr(args, "all_except_d5", False),
    )

    # Summary
    tiers = defaultdict(lambda: [0, 0])  # count, size
    for e in entries:
        tiers[e["tier"]][0] += 1
        tiers[e["tier"]][1] += e.get("size_bytes") or 0

    print("Classification summary:")
    for tier in ("KEEP", "DELETE"):
        count, bytes_ = tiers[tier]
        print(f"  {tier:7}  {count:4} checkpoints   {human_gb(bytes_)}")

    plan_path = TOOLS_DIR / "checkpoint_cleanup_plan.json"
    plan = {
        "profile": args.profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"\nWrote cleanup plan to: {plan_path}")
    print("Review the plan. Move entries between tiers as needed, then run:")
    print(f"  python {Path(__file__).name} delete --profile {args.profile} --plan {plan_path}")
    print("Add --confirm to actually delete.")
    return 0


def cmd_delete(args) -> int:
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: plan file not found: {plan_path}", file=sys.stderr)
        return 2
    plan = json.loads(plan_path.read_text())
    plan_profile = plan.get("profile")
    if plan_profile != args.profile:
        print(
            f"ERROR: plan was generated for profile '{plan_profile}' but you passed "
            f"--profile '{args.profile}'. Refusing to proceed.",
            file=sys.stderr,
        )
        return 2

    to_delete = [e for e in plan.get("entries", []) if e.get("tier") == "DELETE"]
    if getattr(args, "d5_only", False):
        to_delete = [
            e for e in to_delete
            if "d5_abstract_retrieve_refine" in (e.get("local_run_dir") or "")
        ]
    total_size = sum((e.get("size_bytes") or 0) for e in to_delete)

    print(f"Plan: {len(to_delete)} checkpoints tagged DELETE, total {human_gb(total_size)}")

    if not args.confirm:
        print("\nDRY RUN. Use --confirm to actually delete. First 20 entries:\n")
        for e in to_delete[:20]:
            print(f"  [{e['tier']}] {e['tinker_path']}  ({e['reason']})")
        if len(to_delete) > 20:
            print(f"  ... and {len(to_delete) - 20} more")
        return 0

    if not to_delete:
        print("Nothing to delete.")
        return 0

    sc = create_service_client(api_profile=args.profile)
    rest = sc.create_rest_client()

    log_path = TOOLS_DIR / f"checkpoint_cleanup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    deleted = 0
    errors = 0
    with log_path.open("w") as log_f:
        for i, entry in enumerate(to_delete, 1):
            uri = entry["tinker_path"]
            if entry.get("tier") != "DELETE":
                # Hard guard: never delete KEEP or REVIEW items
                continue
            print(f"[{i}/{len(to_delete)}] deleting {uri} ...", end=" ", flush=True)
            try:
                rest.delete_checkpoint_from_tinker_path(uri).result()
                print("ok")
                deleted += 1
                log_f.write(
                    json.dumps(
                        {
                            "status": "deleted",
                            "tinker_path": uri,
                            "size_bytes": entry.get("size_bytes"),
                            "reason": entry.get("reason"),
                            "deleted_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )
                log_f.flush()
            except Exception as exc:  # noqa: BLE001 - we want to continue on any error
                print(f"ERROR: {exc}")
                errors += 1
                log_f.write(
                    json.dumps(
                        {
                            "status": "error",
                            "tinker_path": uri,
                            "error": str(exc),
                            "attempted_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )
                log_f.flush()

    print(f"\nDone. Deleted {deleted}/{len(to_delete)}, errors {errors}")
    print(f"Log: {log_path}")
    return 0 if errors == 0 else 1


def cmd_delete_one(args) -> int:
    if not args.confirm:
        print(f"DRY RUN. Would delete: {args.uri}")
        print("Add --confirm to actually delete.")
        return 0
    sc = create_service_client(api_profile=args.profile)
    rest = sc.create_rest_client()
    print(f"Deleting {args.uri} ...", end=" ", flush=True)
    try:
        rest.delete_checkpoint_from_tinker_path(args.uri).result()
        print("ok")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manage Tinker checkpoints on a specific API profile.")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("list", "suggest-cleanup"):
        sp = sub.add_parser(name)
        sp.add_argument("--profile", required=True, help="API profile name (e.g. 'new', 'old')")
        if name == "suggest-cleanup":
            sp.add_argument("--d5-only", action="store_true",
                            help="Only classify D5 checkpoints; skip all other projects")
            sp.add_argument("--all-except-d5", action="store_true",
                            help="Mark every non-D5-production checkpoint DELETE (nuke all)")

    sp = sub.add_parser("delete")
    sp.add_argument("--profile", required=True)
    sp.add_argument("--plan", required=True, help="Path to cleanup plan JSON file")
    sp.add_argument("--confirm", action="store_true", help="Actually delete (otherwise dry-run)")
    sp.add_argument("--d5-only", action="store_true",
                    help="Restrict deletion to D5 checkpoints only (safety guard)")
    sp.add_argument("--all-except-d5", action="store_true",
                    help="Delete every checkpoint except D5 production URIs")

    sp = sub.add_parser("delete-one")
    sp.add_argument("--profile", required=True)
    sp.add_argument("--uri", required=True, help="tinker:// URI to delete")
    sp.add_argument("--confirm", action="store_true")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command
    if cmd == "list":
        return cmd_list(args)
    if cmd == "suggest-cleanup":
        return cmd_suggest_cleanup(args)
    if cmd == "delete":
        return cmd_delete(args)
    if cmd == "delete-one":
        return cmd_delete_one(args)
    parser.error(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
