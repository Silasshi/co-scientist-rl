"""D5 F4 attribution — offline gradient-mass analysis on existing μ-v4 buffer.

Stage A of F4 Falsifier Probe v2 (per ML-scientist subagent feedback,
plan in /home/silas/.claude/plans/snug-dreaming-yao.md).

Question: at content-token positions in μ-v4 iter-2's generated plans, what
fraction of the total |advantage| mass actually lands? F4's claim is that
content tokens are dominated by stylistic tokens (~100× more numerous, higher
absolute logprob diff). If content_grad_mass_frac < 1% → F4 strongly
supported (gradient barely touches content) → proceed Stage B (paired probe).
If > 5% → F4 weakened (gradient does reach content; learning failure has
another cause) → reconsult before D3a.

This is a cheap test ($3 Tinker compute, ~30 min wall, NO Opus calls).
Replays existing buffer plans through saved μ-v4 iter-2 sampler weights;
does not generate new plans, does not train, does not call critic.

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.f4_attribution_offline \
        out_path=projects/d5_abstract_retrieve_refine/runs/2026_04_29_f4_attribution_offline
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import numpy as np
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
    build_student_prompt,
    build_teacher_prompt,
    parse_critique_xml,
)
from co_scientist.d5_abstract_retrieve_refine.train_mu_v4 import is_content_token

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


REPO_ROOT = Path(__file__).resolve().parents[3]


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    # Source μ-v4 run we're attributing on
    mu_v4_run: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_27_mu_v4"
    target_iter: int = 2  # which iter's plans to analyze (default 2 — peak-vicinity)

    # Iter-2 sampler weights = state at start of iter 2 = end of iter 1.
    # See checkpoints.jsonl row {"iter": 2, "sampler_path": "..."}.
    iter_sampler_path: str = (
        "tinker://6b9d996d-8079-556d-82ac-560247551960:train:0/sampler_weights/iter_0002"
    )

    # Inputs (must match what μ-v4 trained on)
    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    oracle_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"

    # SDPO clamp params (must match μ-v4 production)
    sdpo_scale: float = 1.0
    sdpo_clip_advantage: float = 5.0

    # Output
    out_path: str = (
        "projects/d5_abstract_retrieve_refine/runs/2026_04_29_f4_attribution_offline"
    )

    # Decision thresholds (per plan; subagent recommendation)
    decision_strong_support_thresh: float = 0.01  # < this → F4 strongly supported
    decision_support_thresh: float = 0.05         # < this → F4 supported
    decision_marginal_thresh: float = 0.10        # < this → marginal; ≥ this → F4 weakened


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _load_iter_critique(mu_v4_run: Path, source_iter: int) -> str:
    """Load the critique that fed iter `source_iter`'s teacher prompt.

    For iter 2 this is the critique from iter 1 (the prior iter's critic_response).
    """
    if source_iter == 0:
        # iter 0 used cold-start critique; we need to import COLD_START_CRITIQUE_NONE
        from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
            COLD_START_CRITIQUE_NONE,
        )
        return COLD_START_CRITIQUE_NONE
    prev_iter = source_iter - 1
    crit_path = mu_v4_run / "critic_responses" / f"iter_{prev_iter:03d}.json"
    if not crit_path.exists():
        raise FileNotFoundError(f"No critic_response for iter {prev_iter}: {crit_path}")
    resp = json.loads(crit_path.read_text())
    judg = resp.get("judgments", {})
    if not judg:
        raise RuntimeError(f"No judgments in {crit_path}")
    pid, j = next(iter(judg.items()))
    return parse_critique_xml(j.get("critique_xml", ""))


def _load_iter_plans(mu_v4_run: Path, target_iter: int) -> list[dict]:
    """Load buffer.jsonl rows where iter == target_iter."""
    buf_path = mu_v4_run / "buffer.jsonl"
    if not buf_path.exists():
        raise FileNotFoundError(f"buffer.jsonl missing: {buf_path}")
    plans = []
    with open(buf_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("iter") == target_iter:
                plans.append(rec)
    return plans


def main(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal = _resolve(config.goal_path).read_text().strip()
    oracle = _resolve(config.oracle_path).read_text().strip()
    mu_v4_run = _resolve(config.mu_v4_run)

    prev_critique = _load_iter_critique(mu_v4_run, config.target_iter)
    plans = _load_iter_plans(mu_v4_run, config.target_iter)
    logger.info(
        "Loaded iter=%d data: prev_critique len=%d, n_plans=%d",
        config.target_iter, len(prev_critique), len(plans),
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Build teacher and student prompts (same as the μ-v4 iter trained on)
    teacher_text = build_teacher_prompt(goal, oracle, prev_critique)
    student_text = build_student_prompt(goal, oracle)
    teacher_input = renderer.build_generation_prompt(
        [{"role": "user", "content": teacher_text}]
    )
    student_input = renderer.build_generation_prompt(
        [{"role": "user", "content": student_text}]
    )
    teacher_tokens = teacher_input.to_ints()
    student_tokens = student_input.to_ints()
    logger.info(
        "Prompts built: teacher=%d tokens, student=%d tokens",
        len(teacher_tokens), len(student_tokens),
    )

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    sampling_client = service_client.create_sampling_client(
        model_path=config.iter_sampler_path,
    )
    logger.info("Sampling client ready @ %s", config.iter_sampler_path)

    # ----- Attribution loop -----
    per_plan: list[dict] = []
    for k, p in enumerate(plans):
        raw_text = p.get("raw_tokens_text") or ""
        if not raw_text:
            logger.warning("plan_idx=%s has no raw_tokens_text — skipping", p.get("plan_idx"))
            continue
        try:
            gen_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        except TypeError:
            gen_tokens = tokenizer.encode(raw_text)
        if len(gen_tokens) < 50:
            logger.warning("plan_idx=%s short (%d tokens) — skipping",
                           p.get("plan_idx"), len(gen_tokens))
            continue

        t_lp_full = sampling_client.compute_logprobs(
            types.ModelInput.from_ints(tokens=teacher_tokens + gen_tokens)
        ).result()
        s_lp_full = sampling_client.compute_logprobs(
            types.ModelInput.from_ints(tokens=student_tokens + gen_tokens)
        ).result()
        t_lp = list(t_lp_full[len(teacher_tokens): len(teacher_tokens) + len(gen_tokens)])
        s_lp = list(s_lp_full[len(student_tokens): len(student_tokens) + len(gen_tokens)])
        if len(t_lp) != len(gen_tokens) or len(s_lp) != len(gen_tokens):
            logger.warning("plan_idx=%s length mismatch (t=%d, s=%d, gen=%d) — skipping",
                           p.get("plan_idx"), len(t_lp), len(s_lp), len(gen_tokens))
            continue

        # SDPO advantage formula (matches train_mu_v4.py L427-432 of v6 / equiv L384-389 of pre-probe v4)
        advs = []
        is_content = []
        for tid, t, s in zip(gen_tokens, t_lp, s_lp):
            a = float(t - s) * config.sdpo_scale
            a = max(-config.sdpo_clip_advantage, min(config.sdpo_clip_advantage, a))
            advs.append(a)
            is_content.append(is_content_token(tid, tokenizer))

        abs_advs = [abs(a) for a in advs]
        signed_advs_arr = np.array(advs, dtype=float)
        n_total = len(advs)
        n_content = int(sum(is_content))
        content_density = n_content / n_total if n_total else 0.0
        total_mass = float(sum(abs_advs))
        content_mass = float(sum(a for a, c in zip(abs_advs, is_content) if c))
        content_grad_mass_frac = content_mass / total_mass if total_mass > 0 else 0.0
        mean_abs_content = (
            float(np.mean([a for a, c in zip(abs_advs, is_content) if c]))
            if n_content else 0.0
        )
        mean_abs_stylistic = (
            float(np.mean([a for a, c in zip(abs_advs, is_content) if not c]))
            if n_total - n_content else 0.0
        )
        ratio = (mean_abs_content / mean_abs_stylistic) if mean_abs_stylistic > 0 else float("inf")

        rec = {
            "plan_idx": p.get("plan_idx"),
            "source_iter": p.get("iter"),
            "n_tokens": n_total,
            "n_content": n_content,
            "content_density": content_density,
            "total_abs_adv_mass": total_mass,
            "content_abs_adv_mass": content_mass,
            "content_grad_mass_frac": content_grad_mass_frac,
            "mean_abs_adv_content": mean_abs_content,
            "mean_abs_adv_stylistic": mean_abs_stylistic,
            "ratio_content_vs_stylistic": ratio,
            "mean_signed_adv": float(signed_advs_arr.mean()),
            "frac_pos_adv": float((signed_advs_arr > 0).mean()),
        }
        per_plan.append(rec)
        logger.info(
            "plan %d/%d (idx=%s): n=%d, n_content=%d (%.1f%%), "
            "content_grad_mass=%.3f, ratio=%.2f",
            k + 1, len(plans), p.get("plan_idx"),
            n_total, n_content, 100 * content_density,
            content_grad_mass_frac, ratio,
        )

    if not per_plan:
        logger.error("No plans successfully analyzed. Aborting.")
        return

    # ----- Aggregate -----
    arr = lambda key: np.array([r[key] for r in per_plan if not np.isinf(r[key])], dtype=float)
    agg = {
        "n_plans_analyzed": len(per_plan),
        "mean_n_tokens": float(np.mean([r["n_tokens"] for r in per_plan])),
        "mean_content_density": float(np.mean([r["content_density"] for r in per_plan])),
        "mean_content_grad_mass_frac": float(np.mean([r["content_grad_mass_frac"] for r in per_plan])),
        "median_content_grad_mass_frac": float(np.median([r["content_grad_mass_frac"] for r in per_plan])),
        "min_content_grad_mass_frac": float(np.min([r["content_grad_mass_frac"] for r in per_plan])),
        "max_content_grad_mass_frac": float(np.max([r["content_grad_mass_frac"] for r in per_plan])),
        "mean_ratio_content_vs_stylistic": float(np.mean(arr("ratio_content_vs_stylistic"))),
        "mean_abs_adv_content": float(np.mean([r["mean_abs_adv_content"] for r in per_plan])),
        "mean_abs_adv_stylistic": float(np.mean([r["mean_abs_adv_stylistic"] for r in per_plan])),
    }

    # Decision
    frac = agg["mean_content_grad_mass_frac"]
    if frac < config.decision_strong_support_thresh:
        verdict = "F4_STRONGLY_SUPPORTED"
        action = "Proceed to Stage B (paired probe). Gradient barely touches content tokens."
    elif frac < config.decision_support_thresh:
        verdict = "F4_SUPPORTED"
        action = "Proceed to Stage B (paired probe). Gradient mass on content is small but nonzero."
    elif frac < config.decision_marginal_thresh:
        verdict = "F4_MARGINAL"
        action = "Reconsult ML scientist subagent before committing to D3a; may need to adjust mask granularity."
    else:
        verdict = "F4_WEAKENED"
        action = "Don't pursue D3a — gradient is reaching content tokens; learning failure has another cause. Expand D2 instead."

    summary = {
        "config": {
            "mu_v4_run": str(config.mu_v4_run),
            "target_iter": config.target_iter,
            "iter_sampler_path": config.iter_sampler_path,
        },
        "aggregate": agg,
        "decision": {
            "verdict": verdict,
            "mean_content_grad_mass_frac": frac,
            "thresholds": {
                "strong_support_<": config.decision_strong_support_thresh,
                "support_<": config.decision_support_thresh,
                "marginal_<": config.decision_marginal_thresh,
            },
            "action": action,
        },
        "per_plan": per_plan,
    }

    out_file = out_dir / "f4_attribution_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_file)

    print("\n" + "=" * 70)
    print("F4 OFFLINE ATTRIBUTION — RESULTS")
    print("=" * 70)
    print(f"Plans analyzed:            {agg['n_plans_analyzed']}")
    print(f"Mean tokens/plan:          {agg['mean_n_tokens']:.0f}")
    print(f"Mean content density:      {100*agg['mean_content_density']:.2f}%")
    print(f"Mean |adv| at content:     {agg['mean_abs_adv_content']:.4f}")
    print(f"Mean |adv| at stylistic:   {agg['mean_abs_adv_stylistic']:.4f}")
    print(f"Ratio content/stylistic:   {agg['mean_ratio_content_vs_stylistic']:.2f}")
    print(f"")
    print(f"PRIMARY METRIC:")
    print(f"  Mean content_grad_mass_frac: {100*frac:.2f}%")
    print(f"  Median:                      {100*agg['median_content_grad_mass_frac']:.2f}%")
    print(f"  Range [min, max]:            [{100*agg['min_content_grad_mass_frac']:.2f}%, {100*agg['max_content_grad_mass_frac']:.2f}%]")
    print(f"")
    print(f"DECISION: {verdict}")
    print(f"  ACTION: {action}")
    print("=" * 70)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
