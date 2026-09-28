#!/bin/bash
# Launch B4_v8 + MAIN_v8_C3 + MAIN_v8_C4 on FoundOpt v8 goal dir.
# See .claude/plans/quirky-herding-emerson.md for experiment design.
#
# Two variable changes vs C2/C3/C4 original:
#   1. goal_dir = 01_foundopt_v8 (v8 rubric, 10 signals, G3/G5 removed)
#   2. grader_model_name = openai/gpt-oss-120b (vs Qwen3-30B)
#
# All other config matches C2/C3/C4 exactly (scores_only, no train_on_fresh,
# no novelty bonus, UCB, etc.). grader_repeats=3 (up from 2) for tighter median.

set -e
cd /home/silas/co-scientist-project
source shared/tools/use_api_profile.sh new

RUNS_DIR=projects/grant_proposal_v2/runs
mkdir -p "$RUNS_DIR"

COMMON=(
  config.goal_dir=projects/grant_proposal/dataset/goals/01_foundopt_v8
  config.grader_model_name=openai/gpt-oss-120b
  config.n_fresh=4
  config.n_revise=4
  config.n_revision_candidates=1
  config.grader_repeats=3
  config.grader_temperature=0.0
  config.min_grader_confidence=0.0
  config.scores_only=true
  config.strip_critiques=false
  config.emit_signal_critique=true
  config.include_cot_scaffolding=true
  config.use_ucb=true
  config.cold_start_iters=1
  config.train_on_fresh=false
  config.novelty_weight=0.0
  config.use_retrieval=false
  config.delta_scale=5.0
  config.delta_threshold=0.001
  config.lora_rank=32
  config.learning_rate=4e-5
  config.clip_eps=0.2
  config.max_word_count=2000
  config.save_every=5
  config.seed=0
  config.grader_max_tokens=8192
)

# --- B4_v8 (C2 style): no RL, critique-revise only, upper-bound baseline ---
python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" \
  config.log_path="$RUNS_DIR/2026_04_22_v8_foundopt_B4" \
  config.n_iterations=25 \
  config.skip_rl_update=true \
  > "$RUNS_DIR/2026_04_22_v8_foundopt_B4.log" 2>&1 &
B4_PID=$!

# --- MAIN_v8_C3 (SDPO): per-token self-teacher advantage ---
python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" \
  config.log_path="$RUNS_DIR/2026_04_22_v8_foundopt_MAIN_C3" \
  config.n_iterations=35 \
  config.skip_rl_update=false \
  config.sdpo=true \
  config.sdpo_scale=1.0 \
  config.sdpo_clip_advantage=5.0 \
  config.revision_rl_mode=per_signal \
  > "$RUNS_DIR/2026_04_22_v8_foundopt_MAIN_C3.log" 2>&1 &
C3_PID=$!

# --- MAIN_v8_C4 (aggregate): scalar agg_delta × delta_scale per revision ---
python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" \
  config.log_path="$RUNS_DIR/2026_04_22_v8_foundopt_MAIN_C4" \
  config.n_iterations=35 \
  config.skip_rl_update=false \
  config.sdpo=false \
  config.revision_rl_mode=aggregate \
  > "$RUNS_DIR/2026_04_22_v8_foundopt_MAIN_C4.log" 2>&1 &
C4_PID=$!

echo "=================================================="
echo "Launched v8 FoundOpt experiments:"
echo "  B4_v8            PID=$B4_PID  → $RUNS_DIR/2026_04_22_v8_foundopt_B4/"
echo "  MAIN_v8_C3 (SDPO) PID=$C3_PID  → $RUNS_DIR/2026_04_22_v8_foundopt_MAIN_C3/"
echo "  MAIN_v8_C4 (agg) PID=$C4_PID  → $RUNS_DIR/2026_04_22_v8_foundopt_MAIN_C4/"
echo "=================================================="
echo "Monitor with: bash shared/tools/monitor_v2.sh"
echo "Mid-run depth_audit at iter 5/10/25 (B4) or 5/10/35 (C3/C4):"
echo "  python projects/grant_proposal_v2/scripts/mid_run_depth_audit.py <run_dir> --iter N"
echo "=================================================="

wait
echo "All 3 runs complete."
