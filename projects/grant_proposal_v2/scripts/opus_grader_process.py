"""Helper for the main-session Opus-grader daemon.

Usage:
    python opus_grader_process.py split <request_path>
        → writes <log_path>/grader_workdir/<batch_id>/plan_<idx>_input.json
          with per-plan {system, user, plan_id, signal_ids, hg_ids}

    python opus_grader_process.py aggregate <request_path>
        → reads per-plan responses from grader_workdir and writes the
          final grader_responses/<batch_id>.json atomically, then
          renames request to .done.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


SYSTEM_BASE = (
    "You are Claude Opus acting as a strict substance-focused grader for grant-proposal "
    "research plans. You will be given several INDEPENDENT grading prompts "
    "(10 signal prompts + 4 hard-gate prompts). Each prompt is fully self-contained: "
    "it names the research goal, the plan text, the evaluation dimension, and the exact "
    "XML Output Format it requires. Respond to each prompt EXACTLY as that prompt instructs. "
    "Do not add prose outside the XML envelope that the prompt asks for.\n\n"
    "Calibration: be stricter than a mid-size open-source grader. A '5' means the plan "
    "truly excels on that dimension at expert-grant quality, not just 'mentions the relevant "
    "topics'. A '3' is the median competent plan."
)

SYSTEM_LONG_CRITIQUE = (
    "\n\nWhen a prompt asks for a <critique> block, IGNORE the template's '1-3 sentences' "
    "hint. Instead, write ~300 words of actionable critique per signal, naming SPECIFIC "
    "weaknesses with short verbatim quotes where helpful. Keep <reasoning> concise "
    "(<=3 sentences). Spend length budget on <critique> which is consumed by the revision model."
)

SYSTEM_TAIL = (
    "\n\nAfter completing all prompts, emit ONE final JSON code block with this exact schema:\n"
    "```json\n"
    "{\n"
    '  "plan_id": "<the plan_id>",\n'
    '  "signal_xmls": {"<signal_id_1>": "<evaluation>...</evaluation>", ...},\n'
    '  "hard_gate_xmls": {"gc_target": "<raw output verbatim>", "gc_alt1": "...", "gc_alt2": "...", "cv": "..."}\n'
    "}\n"
    "```\n"
    "Put each prompt's raw output VERBATIM as the corresponding string value. Do not parse "
    "or rewrite the XML — the trainer's existing parser expects the output bytes unchanged."
)


def build_user_message(plan_id: str, plan_text: str, signal_prompts: dict, hg_prompts: dict) -> str:
    parts = [f"plan_id: {plan_id}\n\n=== SIGNAL PROMPTS ===\n"]
    for sid in sorted(signal_prompts.keys()):
        parts.append(f"\n### {sid}\n\n{signal_prompts[sid]}\n")
    parts.append("\n=== HARD-GATE PROMPTS ===\n")
    for hgid in ["gc_target", "gc_alt1", "gc_alt2", "cv"]:
        if hgid in hg_prompts:
            parts.append(f"\n### {hgid}\n\n{hg_prompts[hgid]}\n")
    parts.append(
        "\n---\n\nRespond to each prompt per its own Output Format, then emit the "
        "final {plan_id, signal_xmls, hard_gate_xmls} JSON.\n"
    )
    return "".join(parts)


def cmd_split(request_path: Path) -> None:
    req = json.loads(request_path.read_text())
    batch_id = req["batch_id"]
    long_critique = req.get("grader_long_critique", False)
    system_prompt = SYSTEM_BASE + (SYSTEM_LONG_CRITIQUE if long_critique else "") + SYSTEM_TAIL

    log_path = request_path.parent.parent
    workdir = log_path / "grader_workdir" / batch_id
    workdir.mkdir(parents=True, exist_ok=True)

    for idx, plan in enumerate(req["plans"]):
        pid = plan["plan_id"]
        sp = plan["signal_prompts"]
        hp = plan["hard_gate_prompts"]
        user_msg = build_user_message(pid, plan["text"], sp, hp)
        out = {
            "plan_id": pid,
            "plan_index": idx,
            "batch_id": batch_id,
            "signal_ids": sorted(sp.keys()),
            "hg_ids": sorted(hp.keys()),
            "system": system_prompt,
            "user": user_msg,
        }
        (workdir / f"plan_{idx:02d}_input.json").write_text(json.dumps(out))
    print(f"OK: split {len(req['plans'])} plans to {workdir}/plan_*_input.json")
    print(f"batch_id={batch_id}")
    for idx, plan in enumerate(req["plans"]):
        print(f"  plan_{idx:02d} -> {plan['plan_id']}  ({len(plan['signal_prompts'])} signals + {len(plan['hard_gate_prompts'])} HG)")


def cmd_aggregate(request_path: Path) -> None:
    req = json.loads(request_path.read_text())
    batch_id = req["batch_id"]
    iter_idx = req["iter"]
    log_path = request_path.parent.parent
    workdir = log_path / "grader_workdir" / batch_id
    response_dir = log_path / "grader_responses"
    response_dir.mkdir(parents=True, exist_ok=True)

    judgments: dict[str, dict] = {}
    per_plan_latency: dict[str, float] = {}
    retried = False
    missing: list[str] = []
    for idx, plan in enumerate(req["plans"]):
        pid = plan["plan_id"]
        resp_file = workdir / f"plan_{idx:02d}_response.json"
        if not resp_file.exists():
            missing.append(pid)
            judgments[pid] = {
                "signal_xmls": {sid: f'<evaluation><dim id="{sid}"><score></score></dim></evaluation>' for sid in plan["signal_prompts"]},
                "hard_gate_xmls": {k: '{"score": 0.0}' for k in plan["hard_gate_prompts"]},
            }
            continue
        try:
            r = json.loads(resp_file.read_text())
            if r.get("retried"):
                retried = True
            per_plan_latency[pid] = r.get("latency_sec", 0.0)
            judgments[pid] = {
                "signal_xmls": r.get("signal_xmls", {}),
                "hard_gate_xmls": r.get("hard_gate_xmls", {}),
            }
            # Backfill missing signal keys with empty placeholders
            for sid in plan["signal_prompts"]:
                if sid not in judgments[pid]["signal_xmls"] or not judgments[pid]["signal_xmls"][sid]:
                    judgments[pid]["signal_xmls"][sid] = f'<evaluation><dim id="{sid}"><score></score></dim></evaluation>'
            for hgid in plan["hard_gate_prompts"]:
                if hgid not in judgments[pid]["hard_gate_xmls"] or not judgments[pid]["hard_gate_xmls"][hgid]:
                    judgments[pid]["hard_gate_xmls"][hgid] = '{"score": 0.0}'
        except Exception as e:
            missing.append(pid)
            judgments[pid] = {
                "signal_xmls": {sid: f'<evaluation><dim id="{sid}"><score></score></dim></evaluation>' for sid in plan["signal_prompts"]},
                "hard_gate_xmls": {k: '{"score": 0.0}' for k in plan["hard_gate_prompts"]},
            }
            print(f"WARNING: failed to parse {resp_file}: {e}")

    response = {
        "iter": iter_idx,
        "batch_id": batch_id,
        "n_plans": len(req["plans"]),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judgments": judgments,
        "daemon_diagnostics": {
            "per_plan_latency_sec": per_plan_latency,
            "any_subagent_retried": retried,
            "missing_plans": missing,
        },
    }

    out_path = response_dir / f"{batch_id}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(response))
    tmp_path.replace(out_path)

    # Mark request processed
    done_path = request_path.with_suffix(".json.done")
    os.rename(request_path, done_path)

    # Log
    log_file = log_path / "opus_grader_daemon.log"
    with log_file.open("a") as f:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.write(
            f"{ts}  batch={batch_id}  n_plans={len(req['plans'])}  "
            f"missing={len(missing)}  retried={retried}\n"
        )

    print(f"OK: aggregated -> {out_path}")
    print(f"  missing={missing}  retried={retried}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python opus_grader_process.py {split|aggregate} <request_path>")
        sys.exit(1)
    cmd, req = sys.argv[1], Path(sys.argv[2])
    if cmd == "split":
        cmd_split(req)
    elif cmd == "aggregate":
        cmd_aggregate(req)
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)
