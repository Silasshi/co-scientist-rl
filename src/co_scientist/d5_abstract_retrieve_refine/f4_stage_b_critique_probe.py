"""D5 F4 Stage B — critique-conditioned EVAL probe.

Tests the conditional-context gating hypothesis (after Stage A falsified F4's
gradient-flooding mechanism at content_grad_mass_frac = 15.83% on μ-v4 iter=2
buffer plans, far above F4's predicted ~0.2%).

Question: μ-v4 iter-4 (production) student model was scored 0/8 PUCT in EVAL
under standard `p(plan | goal, oracle)` sampling. Does it generate PUCT when
the EVAL prefix includes a templated pseudo-critique that mentions
"action-selection mechanism" WITHOUT naming PUCT? If yes → conditional-context
gating CONFIRMED: training internalized PUCT into a conditional sub-tree the
unconditional sampler doesn't reach.

Architecture (NO TRAINING):
  Arm A (vanilla):    student_input = goal + oracle              ──> sample 8 plans
  Arm B (pseudo-crit): teacher_input = goal + oracle + pseudo_crit ──> sample 8 plans
  Both via μ-v4 iter-4 sampler weights (production checkpoint).

Outcome: multi-regex content-token presence (PUCT|Q(s,a)|J_β|V(s)|β=N|...).
PRIMARY: PUCT-presence count A vs B.

Decision (pre-registered):
- B≥3/8 PUCT AND A≤1/8 PUCT → CONDITIONAL_GATING_CONFIRMED → Stage C trains
  μ-v7 with critique-prefix replay
- B≤1/8 PUCT AND A≤1/8 PUCT → STUDENT_NEVER_LEARNED → deeper issue, reconsult
- A≥3/8 PUCT → unexpected (training plus inference both produce PUCT
  unconditionally?); reconsult
- mixed → reconsult

Cost: ~$0.50 Tinker sampling + $0 Opus (regex only). Wall ~5-15 min.
Audit /45 deferred (separate command via daemon if desired).

Usage:
    source shared/tools/use_api_profile.sh new
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.f4_stage_b_critique_probe
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from co_scientist.shared.api_profiles import create_service_client
from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v2 import (
    build_student_prompt,
    build_teacher_prompt,
    extract_solution,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


REPO_ROOT = Path(__file__).resolve().parents[3]


# Pre-registered content-token regexes. PUCT-related (PRIMARY) listed first.
PUCT_REGEX = r"PUCT|Q\(\s*s\s*,?\s*a\s*\)|UCB|c\s*\*?\s*P\(s.*?√"
ENTROPIC_REGEX = r"J_?\s*β|J_?\s*beta|J\s*\(\s*β\s*\)|exp\(\s*β"
VALUE_REGEX = r"\bV\(\s*s\s*\)|value\s+function"
HPARAM_REGEX = r"\bβ\s*=\s*[\d.]+|γ\s*=\s*[\d.]+|LoRA\s+rank\s*=?\s*\d+|c\s*=\s*[\d.]+"
SELECTION_REGEX = r"selection\s+rule|select\s+initial\s+state|tree\s+search|MCTS"

CONTENT_REGEXES = {
    "PUCT": PUCT_REGEX,
    "entropic_J_beta": ENTROPIC_REGEX,
    "value_function": VALUE_REGEX,
    "concrete_hparam": HPARAM_REGEX,
    "selection_mention": SELECTION_REGEX,
}

# Pseudo-critique used in Arm B prefix.
# IMPORTANT: does NOT mention PUCT, MCTS, or specific algorithm — only points
# the model toward "action-selection mechanism". If μ-v4 training internalized
# PUCT into a conditional branch keyed on selection-cue critiques, this should
# be enough to surface it; if it requires literal PUCT in the critique, that
# would be a different (weaker) result.
PSEUDO_CRITIQUE_GENERIC = (
    "<critique>"
    "<idea_alignment>The plan addresses the core discovery setting.</idea_alignment>"
    "<missing_components>Prior attempts under-specified the action-selection "
    "mechanism that picks which initial state to expand from the buffer. "
    "The mechanism should be specified formally with an explicit selection "
    "rule and an entropic objective formulation, including concrete "
    "hyperparameter values for the exploration coefficient.</missing_components>"
    "<incorrect_assumptions>None flagged.</incorrect_assumptions>"
    "<feasibility>Feasible within the stated compute budget.</feasibility>"
    "<improvement_directive>Add a formal action-selection rule and an "
    "entropic objective with concrete hyperparameters.</improvement_directive>"
    "</critique>"
)


@chz.chz
class Config:
    api_profile: str | None = "new"
    base_url: str | None = None

    # μ-v4 iter-4 sampler (production checkpoint per F2)
    iter4_sampler_path: str = (
        "tinker://6b9d996d-8079-556d-82ac-560247551960:train:0/sampler_weights/iter_0004"
    )

    goal_path: str = "projects/d5_abstract_retrieve_refine/dataset/research_goal.txt"
    oracle_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md"
    )

    model_name: str = "Qwen/Qwen3-30B-A3B"

    # Sampling (matches Phase 0.5c fix: temp=1.0 + top_p, NO seed)
    n_plans: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95

    # Output
    out_path: str = (
        "projects/d5_abstract_retrieve_refine/runs/2026_04_29_f4_stage_b_critique_probe"
    )


def _resolve(p: str) -> Path:
    return (REPO_ROOT / p).resolve()


def _decode_plan(seq, tokenizer, renderer) -> str:
    parsed = renderer.parse_response(seq.tokens)
    content = parsed[0].get("content", "") if parsed else ""
    return extract_solution(content) if content else tokenizer.decode(seq.tokens)


def _content_hits(plan: str) -> dict[str, bool]:
    return {name: bool(re.search(rx, plan, re.IGNORECASE)) for name, rx in CONTENT_REGEXES.items()}


def main(config: Config) -> None:
    out_dir = _resolve(config.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal = _resolve(config.goal_path).read_text().strip()
    oracle = _resolve(config.oracle_path).read_text().strip()

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    student_text = build_student_prompt(goal, oracle)
    critique_text = build_teacher_prompt(goal, oracle, PSEUDO_CRITIQUE_GENERIC)

    student_input = renderer.build_generation_prompt(
        [{"role": "user", "content": student_text}]
    )
    critique_input = renderer.build_generation_prompt(
        [{"role": "user", "content": critique_text}]
    )
    logger.info("Arm A prompt: %d tokens", len(student_input.to_ints()))
    logger.info("Arm B prompt: %d tokens", len(critique_input.to_ints()))

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )
    sampling_client = service_client.create_sampling_client(
        model_path=config.iter4_sampler_path,
    )
    logger.info("Sampling client ready @ %s", config.iter4_sampler_path)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )

    # Launch both arms in parallel (server-side concurrency)
    armA_future = sampling_client.sample(
        prompt=student_input, num_samples=config.n_plans, sampling_params=sampling_params,
    )
    armB_future = sampling_client.sample(
        prompt=critique_input, num_samples=config.n_plans, sampling_params=sampling_params,
    )
    armA_result = armA_future.result()
    armB_result = armB_future.result()
    logger.info("Both arms sampled (n=%d each)", config.n_plans)

    armA_plans = [_decode_plan(s, tokenizer, renderer) for s in armA_result.sequences]
    armB_plans = [_decode_plan(s, tokenizer, renderer) for s in armB_result.sequences]

    armA_hits = [_content_hits(p) for p in armA_plans]
    armB_hits = [_content_hits(p) for p in armB_plans]

    def _agg(hits_list: list[dict[str, bool]]) -> dict[str, int]:
        out = {}
        for name in CONTENT_REGEXES.keys():
            out[name] = sum(1 for h in hits_list if h[name])
        return out

    armA_counts = _agg(armA_hits)
    armB_counts = _agg(armB_hits)

    armA_density = float(np.mean([np.mean(list(h.values())) for h in armA_hits]))
    armB_density = float(np.mean([np.mean(list(h.values())) for h in armB_hits]))

    armA_puct = armA_counts["PUCT"]
    armB_puct = armB_counts["PUCT"]

    # Pre-registered decision rule
    if armB_puct >= 3 and armA_puct <= 1:
        verdict = "CONDITIONAL_GATING_CONFIRMED"
        action = (
            "F4 reframe confirmed: μ-v4 internalized PUCT into a conditional "
            "branch keyed on selection-cue critiques. Pivot D2 to critique-"
            "prefix replay variant. Consider Stage C: train_mu_v7_critique_prefix "
            "(EVAL-time pseudo-critique conditioning)."
        )
    elif armB_puct <= 1 and armA_puct <= 1:
        verdict = "STUDENT_NEVER_LEARNED_PUCT"
        action = (
            "Both arms ≤1 PUCT — student model never internalized PUCT in any "
            "context. Conditional-context gating ALSO fails. Deeper issue "
            "(LoRA capacity / training signal misalignment / etc). Reconsult ML "
            "scientist; D2 unlikely to help on this goal."
        )
    elif armA_puct >= 3:
        verdict = "PUCT_NOT_CONDITIONAL"
        action = (
            "Arm A produces PUCT unconditionally — contradicts μ-v4 production "
            "audit (0/8 PUCT). Sampling noise or wrong sampler path? Investigate."
        )
    else:
        verdict = "MIXED"
        action = (
            "Marginal result. Review per-plan content for partial PUCT mentions "
            "and reconsult ML scientist."
        )

    summary = {
        "config": {
            "iter4_sampler_path": config.iter4_sampler_path,
            "n_plans": config.n_plans,
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
        "pseudo_critique": PSEUDO_CRITIQUE_GENERIC,
        "armA": {
            "label": "vanilla student (goal+oracle, no critique)",
            "regex_counts": armA_counts,
            "mean_density": armA_density,
            "n_plans": len(armA_plans),
            "per_plan_hits": armA_hits,
        },
        "armB": {
            "label": "critique-conditioned student (goal+oracle+pseudo_critique)",
            "regex_counts": armB_counts,
            "mean_density": armB_density,
            "n_plans": len(armB_plans),
            "per_plan_hits": armB_hits,
        },
        "decision": {
            "verdict": verdict,
            "armA_PUCT": armA_puct,
            "armB_PUCT": armB_puct,
            "armA_density": armA_density,
            "armB_density": armB_density,
            "delta_density": armB_density - armA_density,
            "action": action,
        },
    }

    summary_path = out_dir / "stage_b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plans_path = out_dir / "eval_rollouts.jsonl"
    with open(plans_path, "w") as f:
        for k, p in enumerate(armA_plans):
            f.write(json.dumps({"arm": "A", "plan_idx": k, "plan_text": p}) + "\n")
        for k, p in enumerate(armB_plans):
            f.write(json.dumps({"arm": "B", "plan_idx": k, "plan_text": p}) + "\n")

    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s (16 plans)", plans_path)

    print("\n" + "=" * 72)
    print("F4 STAGE B — CRITIQUE-CONDITIONED EVAL PROBE — RESULTS")
    print("=" * 72)
    print(f"Sampler: {config.iter4_sampler_path}")
    print(f"n_plans/arm: {config.n_plans}, temp: {config.temperature}, top_p: {config.top_p}")
    print()
    print(f"Arm A (vanilla, no critique):")
    for name, c in armA_counts.items():
        print(f"  {name:24s} {c}/{config.n_plans}")
    print(f"  mean density:            {100*armA_density:.1f}%")
    print()
    print(f"Arm B (with pseudo-critique cue):")
    for name, c in armB_counts.items():
        print(f"  {name:24s} {c}/{config.n_plans}")
    print(f"  mean density:            {100*armB_density:.1f}%")
    print()
    print(f"Delta density (B − A):     {100*(armB_density - armA_density):+.1f}%")
    print(f"PUCT delta (B − A):        {armB_puct - armA_puct:+d}/{config.n_plans}")
    print()
    print(f"DECISION: {verdict}")
    print(f"  ACTION: {action}")
    print("=" * 72)


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
