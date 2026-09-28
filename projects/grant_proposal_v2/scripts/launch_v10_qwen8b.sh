#!/bin/bash
# Launch v10_C2 + v10_C3 (SDPO) + v10_C4 (aggregate) on FoundOpt with Qwen3-8B.
# Policy = grader = Qwen3-8B (hybrid dense). Tests MoE-vs-dense architecture
# hypothesis against v7 C2/C3/C4 baseline (Qwen3-30B-A3B).
#
# v10 rubric: v8 base + G4/G10 from v9 (with length caps) + relaxed v9 G11/G12/G13.
# See src/co_scientist/shared/grant_rubric_v10.py and
#     .claude/plans/quirky-herding-emerson.md (approved 2026-04-23).

set -e
cd /home/silas/co-scientist-project
source shared/tools/use_api_profile.sh new

RUNS_DIR=projects/grant_proposal_v2/runs
mkdir -p "$RUNS_DIR"

COMMON=(
  config.goal_dir=projects/grant_proposal/dataset/goals/01_foundopt_v10
  config.model_name=Qwen/Qwen3-8B
  config.grader_model_name=Qwen/Qwen3-8B
  config.n_fresh=4
  config.n_revise=4
  config.n_revision_candidates=1
  config.grader_repeats=2
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

# --- v10_C2: B4 (no RL) baseline ---
python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" \
  config.log_path="$RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_C2" \
  config.n_iterations=25 \
  config.skip_rl_update=true \
  > "$RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_C2.log" 2>&1 &
C2_PID=$!

# --- v10_C3: SDPO (per-token self-teacher advantage) ---
python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" \
  config.log_path="$RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_C3" \
  config.n_iterations=35 \
  config.skip_rl_update=false \
  config.sdpo=true \
  config.sdpo_scale=1.0 \
  config.sdpo_clip_advantage=5.0 \
  config.revision_rl_mode=per_signal \
  > "$RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_C3.log" 2>&1 &
C3_PID=$!

# --- v10_C4: aggregate REINFORCE ---
python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" \
  config.log_path="$RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_C4" \
  config.n_iterations=35 \
  config.skip_rl_update=false \
  config.sdpo=false \
  config.revision_rl_mode=aggregate \
  > "$RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_C4.log" 2>&1 &
C4_PID=$!

echo "=============================================================="
echo "Launched v10 × Qwen3-8B C* rerun:"
echo "  v10_C2 (B4)       PID=$C2_PID  25 iters"
echo "  v10_C3 (SDPO)     PID=$C3_PID  35 iters"
echo "  v10_C4 (aggregate) PID=$C4_PID  35 iters"
echo "=============================================================="
echo "Log paths:"
echo "  $RUNS_DIR/2026_04_23_v10_qwen8b_foundopt_{C2,C3,C4}/"
echo ""
echo "Monitor via mid_run_depth_audit.py at iter 5/10/25(C2)/35(C3,C4)"
echo "=============================================================="

wait
echo "All 3 runs complete."
