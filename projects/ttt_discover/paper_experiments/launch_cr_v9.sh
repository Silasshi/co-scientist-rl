#!/bin/bash
# ============================================================================
# CR-v9 Signal Hardening Experiments (2026-04-18)
#
# Three conditions:
#   MAIN_v9     — full RL (per-signal REINFORCE) + v9 signals
#   B4_v9       — no RL (skip_rl_update) + v9 signals
#   B4_stripped  — no RL + v9 signals + revision prompt shows only aggregate
#
# Each run: 25 iters, seed 0, Qwen3-30B policy+grader, 235B alt grader
# Opus eval agent spawned alongside each run.
# ============================================================================

set -euo pipefail

source shared/tools/use_api_profile.sh new

TRAINER="src/co_scientist/ttt_discover/train_cr_v7.py"
OPUS_AGENT="src/co_scientist/ttt_discover/opus_eval_agent.py"
GOAL_PATH="projects/ttt_discover/analysis/sanity_check/research_goal.txt"
RUN_DIR="projects/ttt_discover/runs/2026_04_18_v9_signal_hardening"

# chz uses key=value syntax, not --key value
COMMON_ARGS="config.n_iterations=25 config.n_fresh=4 config.n_revise=4 config.n_revision_candidates=2 config.grader_repeats=2 config.disabled_signals=S4_significance config.grader_model_alt=Qwen/Qwen3-235B-A22B-Instruct-2507 config.emit_signal_critique=true config.seed=0"

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
    local trainer_pid=$!
    echo "  Trainer PID: $trainer_pid"

    # Spawn Opus eval agent
    PYTHONPATH=src nohup python "$OPUS_AGENT" \
        --buffer_path "${log_path}/buffer.jsonl" \
        --eval_every 5 \
        --goal_path "$GOAL_PATH" \
        --max_wait_hours 4 \
        > "${log_path}/opus_eval_stdout.log" 2>&1 &
    local opus_pid=$!
    echo "  Opus eval PID: $opus_pid"
    echo ""
}

# --- MAIN_v9: full RL ---
launch_run "MAIN_v9" \
    config.skip_rl_update=false \
    config.strip_critiques=false

# --- B4_v9: no RL, with per-signal critiques ---
launch_run "B4_v9" \
    config.skip_rl_update=true \
    config.strip_critiques=false

# --- B4_stripped: no RL, aggregate-only feedback ---
launch_run "B4_stripped" \
    config.skip_rl_update=true \
    config.strip_critiques=true

echo "All 3 runs launched. Monitor with:"
echo "  tail -f ${RUN_DIR}/MAIN_v9/stdout.log"
echo "  tail -f ${RUN_DIR}/B4_v9/stdout.log"
echo "  tail -f ${RUN_DIR}/B4_stripped/stdout.log"
echo ""
echo "Opus eval logs at:"
echo "  ${RUN_DIR}/*/opus_eval_log.jsonl"
