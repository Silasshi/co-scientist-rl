#!/bin/bash
# Launch Opus-as-grader × V10 rubric ablation on FoundOpt with Qwen3-8B policy.
#
# Design: plan at ~/.claude/plans/polymorphic-sprouting-eagle.md
# 4 conditions × 2 seeds = 8 runs, all with grader_backend=opus_subagent and
# grader_long_critique=true. Trainer is train_cr_v7.py (matches prior v10×Qwen-8B
# C* series — single-variable swap: grader only).
#
# IMPORTANT: each run requires a matching Opus-grader daemon subagent spawned
# from the Claude Code main session, polling <log_path>/grader_requests/.
# Daemon spec: projects/grant_proposal_v2/scripts/opus_grader_subagent.md.
#
# Usage:
#   bash projects/grant_proposal_v2/scripts/launch_opus_c_series.sh <condition> <seed>
# where condition ∈ {C2, C3, C4, HER} and seed ∈ {0, 1}.
#
# The script prints the log_path it's writing and backgrounds the run;
# capture the PID and the log_path for monitoring.

set -e
cd /home/silas/co-scientist-project
source shared/tools/use_api_profile.sh new

CONDITION="${1:?usage: $0 <C2|C3|C4|HER> <seed>}"
SEED="${2:?usage: $0 <C2|C3|C4|HER> <seed>}"
RUNS_DIR="projects/grant_proposal_v2/runs"
TODAY="$(date +%Y_%m_%d)"
RUN_NAME="${TODAY}_opus_v10_qwen8b_${CONDITION,,}_seed${SEED}"
LOG_PATH="${RUNS_DIR}/${RUN_NAME}"
mkdir -p "${LOG_PATH}" "${LOG_PATH}/grader_requests" "${LOG_PATH}/grader_responses"

COMMON=(
  config.goal_dir=projects/grant_proposal/dataset/goals/01_foundopt_v10
  config.model_name=Qwen/Qwen3-8B
  config.grader_model_name=Qwen/Qwen3-8B
  config.grader_backend=opus_subagent
  config.grader_long_critique=true
  config.grader_batch_timeout_sec=1800.0
  config.n_fresh=4
  config.n_revise=4
  config.n_revision_candidates=1
  config.grader_repeats=1
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
  config.grader_max_tokens=8192
  "config.log_path=${LOG_PATH}"
  "config.seed=${SEED}"
)

case "${CONDITION}" in
  C2)
    # B4: no RL update
    EXTRA=(
      config.n_iterations=25
      config.skip_rl_update=true
    )
    ;;
  C3)
    # SDPO: self-teacher per-token advantage
    EXTRA=(
      config.n_iterations=25
      config.skip_rl_update=false
      config.sdpo=true
      config.sdpo_scale=1.0
      config.sdpo_clip_advantage=5.0
      config.revision_rl_mode=per_signal
    )
    ;;
  C4)
    # Aggregate REINFORCE (PPO-style uniform advantage)
    EXTRA=(
      config.n_iterations=25
      config.skip_rl_update=false
      config.sdpo=false
      config.revision_rl_mode=aggregate
    )
    ;;
  HER)
    # Per-signal HER REINFORCE (default MAIN mode)
    EXTRA=(
      config.n_iterations=25
      config.skip_rl_update=false
      config.sdpo=false
      config.revision_rl_mode=per_signal
    )
    ;;
  *)
    echo "Unknown condition: ${CONDITION}" >&2
    exit 1
    ;;
esac

echo "=============================================================="
echo "Launching Opus-as-grader run:"
echo "  condition=${CONDITION}  seed=${SEED}"
echo "  log_path=${LOG_PATH}"
echo "  daemon inbox: ${LOG_PATH}/grader_requests/"
echo "  daemon outbox: ${LOG_PATH}/grader_responses/"
echo "=============================================================="

python src/co_scientist/grant_proposal/train_cr_v7.py \
  "${COMMON[@]}" "${EXTRA[@]}" \
  > "${LOG_PATH}/train.log" 2>&1 &
TRAINER_PID=$!

echo "trainer pid=${TRAINER_PID}"
echo "log:  tail -f ${LOG_PATH}/train.log"
echo ""
echo "Remember to spawn the Opus grader daemon for this log_path:"
echo "  (paste projects/grant_proposal_v2/scripts/opus_grader_subagent.md"
echo "   into a background Claude Code Agent, with LOG_PATH=${LOG_PATH})"
