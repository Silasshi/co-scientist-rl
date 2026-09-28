#!/usr/bin/env bash
# Launch CR-v7 per-signal context REINFORCE paper runs.
#
# Changes vs launch_all_v81_minimal.sh:
#   - revision_mode="whole_plan_per_signal" (CR-v7)
#   - locus_based_edit=False, paragraph_level_edit=False (legacy paths off)
#   - All other defaults match v6_minimal baseline:
#     max_tokens=4096, disabled_signals="S4_significance", n_iterations=25
#
# Run dir: projects/ttt_discover/runs/2026_04_30b_paper_experiments_v81/*_v7_per_signal/
#
# Usage: bash projects/ttt_discover/paper_experiments/launch_cr_v7.sh [phase]
#   phase=smoke   3-iter sanity pilot (single run, ~12 min) — RUN THIS FIRST
#   phase=1       MAIN_v7 + B4_v7 (2 paper runs, ~2h concurrent)
#   phase=all     same as phase=1 (does not include smoke)

set -euo pipefail

TRAINER="python src/co_scientist/ttt_discover/train_cr_v7.py"
RUNS_DIR="projects/ttt_discover/runs/2026_04_30b_paper_experiments_v81"
PHASE="${1:-all}"
SKIP_S4="S4_significance"

launch() {
    local name="$1"
    shift
    local log_path="${RUNS_DIR}/${name}"
    mkdir -p "${log_path}/train"
    echo "[$(date +%H:%M:%S)] Launching ${name}..."
    nohup ${TRAINER} config.log_path="${log_path}" "$@" \
        > "${log_path}/train/stdout.log" 2>&1 &
    echo "  PID=$! -> ${log_path}"
}

# ── Phase smoke: 3-iter sanity pilot ────────────────────────────────────────
if [[ "$PHASE" == "smoke" ]]; then
    echo "=== Phase smoke: SMOKE_v7_3iter ==="
    launch "SMOKE_v7_3iter" \
        config.n_iterations=3 \
        config.n_fresh=2 \
        config.n_revise=2 \
        config.disabled_signals="${SKIP_S4}"
    exit 0
fi

# ── Phase 1: MAIN_v7 + B4_v7 paper runs ─────────────────────────────────────
if [[ "$PHASE" == "1" || "$PHASE" == "all" ]]; then
    echo "=== Phase 1: MAIN_v7_per_signal + B4_v7_per_signal ==="

    # MAIN: full CR-v7, 25 iter, RL enabled
    launch "MAIN_v7_per_signal" \
        config.n_iterations=25 \
        config.max_tokens=4096 \
        config.disabled_signals="${SKIP_S4}"

    # B4: CR-v7 in-context only (skip_rl_update=True) — tests if per-signal
    # critiques alone improve plans vs RL with per-signal gradients.
    launch "B4_v7_per_signal" \
        config.n_iterations=25 \
        config.skip_rl_update=True \
        config.max_tokens=4096 \
        config.disabled_signals="${SKIP_S4}"
fi

echo ""
echo "All requested experiments launched. Monitor with:"
echo "  tail -f ${RUNS_DIR}/*_v7_per_signal/train/stdout.log"
