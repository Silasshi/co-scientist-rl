#!/bin/bash
# v8 Batch 1+2+3 Orchestrator — sequential 12-cell run
# Filed 2026-04-28 by Claude. Auto-fires from cron at 06:40.
#
# Design: this script launches each of 12 cells one at a time in foreground
# (so the next cell only starts after the prior trainer exits). Each cell's
# Python process blocks on critic.wait() per iter. The Claude Code session
# that started this script is responsible for being the in-process reviewer
# (poll critic_requests/, dispatch Agent subagents, write critic_responses/
# atomically). When trainer logs "v8-d5sdpo run complete", this script moves
# to the next cell.
#
# Critical invariants:
# - max_tokens=8192 is the new default in train_mu_v8_d5sdpo.py Config (do
#   NOT pass max_tokens on CLI; use the source default).
# - n_iter=16 for all cells (matches v7-opd-full for cliff comparison).
# - audit_drop_threshold=10.0 (bumped from 5.0 because audit /45 has higher
#   variance than /20; trainer's early-stop reads "total" agnostic to scale).
# - All cells write to projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_<cell>/
# - On any cell trainer error → log + continue to next cell (don't bail out).
#
# Usage (called by cron-fired Claude Code session):
#   bash projects/d5_abstract_retrieve_refine/scripts/run_v8_batch1_orchestrator.sh
#
# State files written:
#   - <run_dir>/_RUNNING.flag       — present while cell trainer alive
#   - <run_dir>/_DONE.flag          — written when cell trainer exits cleanly
#   - <run_dir>/_FAILED.flag        — written on non-zero exit
#   - projects/d5_abstract_retrieve_refine/runs/_v8_orchestrator_state.json
#     — global state: current_cell_idx, started_at, completed_cells[]

set -u  # unset vars are errors
set -o pipefail

REPO_ROOT="/home/silas/co-scientist-project"
cd "$REPO_ROOT"

source shared/tools/use_api_profile.sh new

RUNS_DIR="$REPO_ROOT/projects/d5_abstract_retrieve_refine/runs"
STATE_FILE="$RUNS_DIR/_v8_phase2_orchestrator_state.json"

# Phase 2 attribution grid — 9 remaining cells.
# Phase 1 already completed: base / G / Gplus / A (A had no useful audit; treat as not-run).
# Priority order: E (α-attribution at G+) → Cplus (PPO-attribution) → Fplus (mask-attribution)
# → A (mask-only) → Bplus / B / C / D / F to fill factorial.
# audit_drop_threshold tightened from 10.0 → 7.0 per base KILLED_README:72 recommendation.
declare -a CELLS=(
    "A       solution_only_mask=True  trust_region_alpha=0.0  loss_fn_name=importance_sampling"
    "C       solution_only_mask=True  trust_region_alpha=0.05 loss_fn_name=importance_sampling"
)

START_TS=$(date -Iseconds)
echo "[orchestrator] start at $START_TS, ${#CELLS[@]} cells queued"

mkdir -p "$RUNS_DIR"
cat > "$STATE_FILE" <<EOF
{
  "started_at": "$START_TS",
  "n_cells": ${#CELLS[@]},
  "current_cell_idx": 0,
  "completed_cells": []
}
EOF

for i in "${!CELLS[@]}"; do
    spec="${CELLS[$i]}"
    cell_name=$(echo "$spec" | awk '{print $1}')
    cell_flags=$(echo "$spec" | cut -d' ' -f2-)
    run_dir="$RUNS_DIR/2026_04_29_mu_v8_d5sdpo_${cell_name}"

    echo ""
    echo "================================================================"
    echo "[orchestrator] cell $((i+1))/${#CELLS[@]}: ${cell_name}"
    echo "[orchestrator] flags: ${cell_flags}"
    echo "[orchestrator] run_dir: ${run_dir}"
    echo "[orchestrator] $(date -Iseconds)"
    echo "================================================================"

    mkdir -p "$run_dir"
    touch "$run_dir/_RUNNING.flag"

    # Launch trainer FOREGROUND. Trainer will block on critic.wait per iter.
    # The Claude Code session running this script is responsible for polling
    # critic_requests/ and dispatching subagents; trainer unblocks when
    # critic_responses/iter_NNN.json appears.
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.train_mu_v8_d5sdpo \
        log_path="projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_${cell_name}" \
        n_iter=16 \
        eval_every=1 \
        save_every=1 \
        audit_drop_threshold=7.0 \
        $cell_flags \
        > "$run_dir/launch.log" 2>&1
    rc=$?

    rm -f "$run_dir/_RUNNING.flag"
    if [ $rc -eq 0 ]; then
        touch "$run_dir/_DONE.flag"
        echo "[orchestrator] cell ${cell_name} exited 0 at $(date -Iseconds)"
    else
        touch "$run_dir/_FAILED.flag"
        echo "[orchestrator] cell ${cell_name} FAILED rc=$rc at $(date -Iseconds)"
        echo "[orchestrator] continuing to next cell despite failure"
    fi

    # Update state file
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['current_cell_idx'] = $((i+1))
state['completed_cells'].append({
    'idx': $i,
    'name': '${cell_name}',
    'rc': $rc,
    'finished_at': '$(date -Iseconds)',
})
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
done

END_TS=$(date -Iseconds)
echo ""
echo "================================================================"
echo "[orchestrator] ALL CELLS COMPLETE at $END_TS"
echo "[orchestrator] Started: $START_TS"
echo "[orchestrator] State file: $STATE_FILE"
echo "================================================================"

touch "$RUNS_DIR/_v8_phase2_ALL_DONE.flag"
