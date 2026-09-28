#!/bin/bash
# ============================================================================
# CR-v9 Fresh-RL + BoN=1 Experiments (2026-04-18)
#
# Tests whether training RL on fresh plans (exploration) creates a gap
# over no-RL baseline. BoN=1 removes B4's inference-time optimization.
#
# Three conditions:
#   MAIN_fresh_bon1  — RL + train_on_fresh + BoN=1 (max RL advantage)
#   B4_bon1          — no RL + BoN=1 (baseline without RL/BoN)
#   MAIN_fresh_bon2  — RL + train_on_fresh + BoN=2 (isolate BoN effect)
# ============================================================================

set -euo pipefail

source shared/tools/use_api_profile.sh new

TRAINER="src/co_scientist/ttt_discover/train_cr_v7.py"
OPUS_AGENT="src/co_scientist/ttt_discover/opus_eval_agent.py"
GOAL_PATH="projects/ttt_discover/analysis/sanity_check/research_goal.txt"
RUN_DIR="projects/ttt_discover/runs/2026_04_18_v9_fresh_bon1"

COMMON_ARGS="config.n_iterations=25 config.n_fresh=4 config.n_revise=4 config.grader_repeats=2 config.disabled_signals=S4_significance config.grader_model_alt=Qwen/Qwen3-235B-A22B-Instruct-2507 config.emit_signal_critique=true config.seed=0"

launch_run() {
    local name=$1
    shift
    local log_path="${RUN_DIR}/${name}"
    mkdir -p "$log_path"

    echo "=== Launching $name ==="
    PYTHONPATH=src nohup python "$TRAINER" \
        config.log_path="$log_path" \
        $COMMON_ARGS \
        "$@" \
        > "${log_path}/stdout.log" 2>&1 &
    echo "  Trainer PID: $!"
    echo ""
}

# --- MAIN_fresh_bon1: RL + train_on_fresh + BoN=1 ---
launch_run "MAIN_fresh_bon1" \
    config.skip_rl_update=false \
    config.train_on_fresh=true \
    config.n_revision_candidates=1

# --- B4_bon1: no RL + BoN=1 ---
launch_run "B4_bon1" \
    config.skip_rl_update=true \
    config.train_on_fresh=false \
    config.n_revision_candidates=1

# --- MAIN_fresh_bon2: RL + train_on_fresh + BoN=2 ---
launch_run "MAIN_fresh_bon2" \
    config.skip_rl_update=false \
    config.train_on_fresh=true \
    config.n_revision_candidates=2

echo "All 3 runs launched. Monitor with:"
echo "  tail -f ${RUN_DIR}/MAIN_fresh_bon1/stdout.log"
echo "  tail -f ${RUN_DIR}/B4_bon1/stdout.log"
echo "  tail -f ${RUN_DIR}/MAIN_fresh_bon2/stdout.log"
