#!/usr/bin/env bash
# Launch all 13 paper experiments for 30B self-grading CR-v5
# Usage: bash projects/ttt_discover/paper_experiments/launch_all.sh [phase]
#   phase=1  MAIN + B4 + A2 + A8 (critical)
#   phase=2  B1 + B2 + B3 (baselines)
#   phase=3  A1 + A3 + A4 + A5 + A6 + A7 (ablations)
#   phase=all  (default) all phases

set -euo pipefail

TRAINER="python src/co_scientist/ttt_discover/train_critique_revise.py"
RUNS_DIR="projects/ttt_discover/runs/2026_04_30b_paper_experiments"
PHASE="${1:-all}"

launch() {
    local name="$1"
    shift
    local log_path="${RUNS_DIR}/${name}"
    mkdir -p "${log_path}/train"
    echo "[$(date +%H:%M:%S)] Launching ${name}..."
    # chz uses config.xxx prefix for all arguments
    nohup ${TRAINER} config.log_path="${log_path}" "$@" \
        > "${log_path}/train/stdout.log" 2>&1 &
    echo "  PID=$! -> ${log_path}"
}

# ── Phase 1: Critical experiments ───────────────────────────────────────
if [[ "$PHASE" == "1" || "$PHASE" == "all" ]]; then
    echo "=== Phase 1: MAIN + B4 + A2 + A8 ==="

    launch "MAIN"
    # No overrides — all defaults

    launch "B4_no_training" \
        config.skip_rl_update=True

    launch "A2_no_s9_focus" \
        config.disabled_signals="S9_focus"

    launch "A8_no_revision" \
        config.n_revise=0 \
        config.n_fresh=8 \
        config.train_on_fresh=True
fi

# ── Phase 2: Baselines ─────────────────────────────────────────────────
if [[ "$PHASE" == "2" || "$PHASE" == "all" ]]; then
    echo "=== Phase 2: B1 + B2 + B3 ==="

    launch "B1_zero_shot" \
        config.n_iterations=1 \
        config.n_fresh=1 \
        config.n_revise=0 \
        config.skip_rl_update=True

    launch "B2_best_of_n" \
        config.n_iterations=1 \
        config.n_fresh=50 \
        config.n_revise=0 \
        config.skip_rl_update=True

    launch "B3_buffer_ttt_entropic" \
        config.n_revise=0 \
        config.n_fresh=8 \
        config.train_on_fresh=True
fi

# ── Phase 3: Ablations ─────────────────────────────────────────────────
if [[ "$PHASE" == "3" || "$PHASE" == "all" ]]; then
    echo "=== Phase 3: A1 + A3 + A4 + A5 + A6 + A7 ==="

    launch "A1_no_hard_gates" \
        config.skip_hard_gates=True

    launch "A3_no_s1_depth" \
        config.disabled_signals="S1_depth"

    launch "A4_best_of_1" \
        config.n_revision_candidates=1

    launch "A5_no_skip_signal" \
        config.min_revision_attempts=999

    launch "A6_no_ucb" \
        config.use_ucb=False
fi

echo ""
echo "All requested experiments launched. Monitor with:"
echo "  tail -f ${RUNS_DIR}/*/train/stdout.log"
echo "  # or"
echo "  bash shared/tools/monitor_v2.sh"
