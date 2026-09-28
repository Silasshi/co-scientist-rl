"""D5 Phase 3 Smoke A — critique correlation between Qwen3-30B-self vs Opus.

Pre-validation for κ-self before committing to full κ_self smoke. Tests:
- Challenge 1: self-review collapse (Qwen3 critiquing Qwen3-output, systematic bias)
- Challenge 2: privileged info comprehension (does 30B reader catch source paper details)

Substrate: 8 τ_v4 plans (varied quality, already audited).

Two phases:
  1. SUBMIT (this script no-arg): generate Qwen3 critiques via Tinker + write Opus
     critic request files for downstream subagent dispatch
  2. JUDGE (this script analyze=true after Opus critic responses come back +
     after Opus-as-judge responses come back): aggregate format compliance,
     win-rates, specificity proxies

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.smoke_a_critique_correlation_v1
    # external: dispatch 8 Opus critic subagents + 8 Opus-as-judge subagents
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.smoke_a_critique_correlation_v1 analyze=true
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d5_abstract_retrieve_refine.kappa_prompts_v1 import (
    build_self_review_critic_prompt,
    extract_distillation,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    out_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_smoke_a_critique_corr"
    tau_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_28_tau_v4"
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    source_paper_path: str = "projects/d5_abstract_retrieve_refine/data/source_paper/v2.md"

    model: str = "Qwen/Qwen3-30B-A3B"
    critic_max_tokens: int = 1500
    temperature: float = 1.0
    top_p: float = 0.95

    analyze: bool = False


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_tau_plans(run_dir: Path) -> list[dict]:
    bp = run_dir / "buffer.jsonl"
    plans = []
    for line in bp.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        plans.append({
            "plan_id": f"tau_v4_{d['sample_idx']}",
            "text": extract_distillation(d.get("plan_text", "")),
            "sample_idx": d["sample_idx"],
        })
    return plans


def submit_phase(config: Config) -> None:
    """Phase 1 — generate Qwen3 critiques via Tinker; write Opus critic + judge request files."""
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from co_scientist.shared.api_profiles import create_service_client

    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qwen_critiques").mkdir(exist_ok=True)
    (out_dir / "opus_critic_requests").mkdir(exist_ok=True)

    goal = _resolve(config.goal_path).read_text().strip()
    source_paper = _resolve(config.source_paper_path).read_text().strip()
    plans = _load_tau_plans(_resolve(config.tau_v4_run))
    logger.info("Loaded %d τ_v4 plans (avg %d chars)", len(plans), sum(len(p['text']) for p in plans)//len(plans))

    # Tinker setup for Qwen3 critic
    tokenizer = get_tokenizer(config.model)
    renderer_name = model_info.get_recommended_renderer_name(config.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    service_client = create_service_client(api_profile=config.api_profile, base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model)
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.critic_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    qwen_critiques = []
    for p in plans:
        critic_prompt = build_self_review_critic_prompt(
            goal=goal, distillation_text=p["text"],
            source_paper=source_paper, round_idx=1,
        )
        critic_input = renderer.build_generation_prompt(
            [{"role": "user", "content": critic_prompt}]
        )
        t0 = time.time()
        result = sampling_client.sample(
            prompt=critic_input, num_samples=1, sampling_params=sampling_params,
        ).result()
        critic_raw = tokenizer.decode(result.sequences[0].tokens)
        wall = time.time() - t0
        rec = {
            "plan_id": p["plan_id"],
            "qwen_critique_raw": critic_raw,
            "wall_sec": wall,
            "stop_reason": result.sequences[0].stop_reason,
            "n_tokens": len(result.sequences[0].tokens),
        }
        qwen_critiques.append(rec)
        # save individual file too for easy inspection
        (out_dir / "qwen_critiques" / f"{p['plan_id']}.txt").write_text(critic_raw)
        logger.info("plan %s: Qwen critique %d chars in %.1fs", p["plan_id"], len(critic_raw), wall)

    (out_dir / "qwen_critiques.jsonl").write_text(
        "\n".join(json.dumps(r) for r in qwen_critiques) + "\n"
    )

    # Write Opus critic request files (one per plan, dispatched by 8 parallel subagents)
    for p, qwen_rec in zip(plans, qwen_critiques):
        critic_prompt = build_self_review_critic_prompt(
            goal=goal, distillation_text=p["text"],
            source_paper=source_paper, round_idx=1,
        )
        req = {
            "plan_id": p["plan_id"],
            "rendered_critic_prompt": critic_prompt,
            "instructions": (
                "Call Opus 4.7 with the rendered_critic_prompt verbatim. The"
                " prompt is the full self-review critic format; you (Opus) are"
                " acting as the privileged-info reviewer with source paper"
                " access. Return the <distill_critique>...</distill_critique>"
                " XML block verbatim. DO NOT modify scores/structure."
            ),
        }
        (out_dir / "opus_critic_requests" / f"{p['plan_id']}.json").write_text(json.dumps(req, indent=2))

    logger.info("Wrote %d Opus critic request files. Now dispatch 8 subagents.", len(plans))
    logger.info("After Opus critic responses arrive, write Opus-as-judge request files via separate step.")


def _parse_xml(text: str) -> dict | None:
    """Best-effort parse of <distill_critique> XML block."""
    m = re.search(r"<distill_critique>(.*?)</distill_critique>", text, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    fields = {}
    for tag in ["missing_critical", "noise", "faithfulness", "improvement_directive"]:
        fm = re.search(f"<{tag}>(.*?)</{tag}>", body, re.DOTALL)
        if fm:
            fields[tag] = fm.group(1).strip()
        else:
            fields[tag] = None
    return fields


def analyze_phase(config: Config) -> None:
    """Phase 2 — aggregate Qwen + Opus critiques + Opus-as-judge verdicts."""
    out_dir = _resolve(config.out_path)
    qwen_records = [json.loads(l) for l in (out_dir / "qwen_critiques.jsonl").read_text().splitlines() if l.strip()]
    qwen_by_plan = {r["plan_id"]: r for r in qwen_records}

    opus_critic_dir = out_dir / "opus_critic_responses"
    judge_dir = out_dir / "opus_judge_responses"

    rows = []
    for plan_id, qrec in qwen_by_plan.items():
        opus_resp_path = opus_critic_dir / f"{plan_id}.json"
        judge_resp_path = judge_dir / f"{plan_id}.json"
        opus_critique_raw = ""
        if opus_resp_path.exists():
            opus_critique_raw = json.loads(opus_resp_path.read_text()).get("critique_xml", "") or ""
        judge_verdict = ""
        if judge_resp_path.exists():
            jd = json.loads(judge_resp_path.read_text())
            judge_verdict = jd.get("winner", "") + " — " + jd.get("rationale", "")[:200]

        qwen_xml = _parse_xml(qrec["qwen_critique_raw"])
        opus_xml = _parse_xml(opus_critique_raw)
        rows.append({
            "plan_id": plan_id,
            "qwen_chars": len(qrec["qwen_critique_raw"]),
            "qwen_xml_parsed": qwen_xml is not None,
            "qwen_missing_critical": qwen_xml.get("missing_critical", "")[:300] if qwen_xml else "",
            "opus_chars": len(opus_critique_raw),
            "opus_xml_parsed": opus_xml is not None,
            "opus_missing_critical": opus_xml.get("missing_critical", "")[:300] if opus_xml else "",
            "judge_verdict": judge_verdict,
        })

    md = ["# Smoke A — Critique correlation Qwen3-self vs Opus\n\n"]
    md.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md.append(f"*N = {len(rows)} τ_v4 plans*\n\n")

    n_qwen_parse = sum(1 for r in rows if r["qwen_xml_parsed"])
    n_opus_parse = sum(1 for r in rows if r["opus_xml_parsed"])
    n_judge_done = sum(1 for r in rows if r["judge_verdict"])
    md.append("## Format compliance\n\n")
    md.append(f"- Qwen3-self XML parses: **{n_qwen_parse}/{len(rows)}** ({100*n_qwen_parse/len(rows):.0f}%)\n")
    md.append(f"- Opus XML parses: **{n_opus_parse}/{len(rows)}** ({100*n_opus_parse/len(rows):.0f}%)\n")
    md.append(f"- Opus-as-judge verdicts: **{n_judge_done}/{len(rows)}**\n\n")

    md.append("## Length\n\n")
    if n_qwen_parse:
        avg_q = sum(r["qwen_chars"] for r in rows) / len(rows)
        md.append(f"- Qwen3 critique avg: **{avg_q:.0f} chars** ({avg_q/4:.0f} tokens)\n")
    if n_opus_parse:
        avg_o = sum(r["opus_chars"] for r in rows if r["opus_chars"]) / max(1, n_opus_parse)
        md.append(f"- Opus critique avg: **{avg_o:.0f} chars** ({avg_o/4:.0f} tokens)\n")

    md.append("\n## Per-plan side-by-side missing_critical\n\n")
    for r in rows:
        md.append(f"### {r['plan_id']}\n")
        md.append(f"**Qwen3** ({r['qwen_chars']} chars{', parsed' if r['qwen_xml_parsed'] else ', UNPARSED'}):\n")
        md.append(f"> {r['qwen_missing_critical']}\n\n")
        md.append(f"**Opus** ({r['opus_chars']} chars{', parsed' if r['opus_xml_parsed'] else ', UNPARSED'}):\n")
        md.append(f"> {r['opus_missing_critical']}\n\n")
        if r["judge_verdict"]:
            md.append(f"**Opus-as-judge**: {r['judge_verdict']}\n\n")
        md.append("---\n\n")

    if n_judge_done:
        verdicts = [r["judge_verdict"].split(" — ")[0] for r in rows if r["judge_verdict"]]
        opus_wins = sum(1 for v in verdicts if v.upper() in ("B", "OPUS"))
        qwen_wins = sum(1 for v in verdicts if v.upper() in ("A", "QWEN", "QWEN3"))
        ties = sum(1 for v in verdicts if v.upper() in ("TIE",))
        md.append(f"## Opus-as-judge aggregate\n\n")
        md.append(f"- Opus critique preferred: **{opus_wins}/{len(verdicts)}**\n")
        md.append(f"- Qwen critique preferred: **{qwen_wins}/{len(verdicts)}**\n")
        md.append(f"- Tie: **{ties}/{len(verdicts)}**\n")

    out_md = out_dir / "smoke_a_summary.md"
    out_md.write_text("".join(md))
    logger.info("Wrote: %s", out_md)
    print(out_md.read_text())


def main(config: Config):
    if config.analyze:
        analyze_phase(config)
    else:
        submit_phase(config)


if __name__ == "__main__":
    main(chz.entrypoint(Config))
