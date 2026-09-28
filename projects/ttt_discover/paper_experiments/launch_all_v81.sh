#!/usr/bin/env bash
# Launch v8.1-minimal paper experiments for 30B self-grading CR-v5
#
# Changes vs launch_all.sh:
#   - All runs skip S4 grading (weight=0 after Occam's razor review)
#   - A3 changed from "no S1_depth" to "no S8_scope" (S1 was near-zero weight anyway)
#   - Runs go to 2026_04_30b_paper_experiments_v81/ (new dir, preserves v5 archive)
#
# Usage: bash projects/ttt_discover/paper_experiments/launch_all_v81.sh [phase]
#   phase=1   MAIN + B1 + B4 + A3' + A8 (critical + sanity)
#   phase=2   B2 + B3
#   phase=3   A1 + A2 + A4 + A5 + A6 + A7
#   phase=all (default) all phases

set -euo pipefail

TRAINER="python src/co_scientist/ttt_discover/train_critique_revise.py"
RUNS_DIR="projects/ttt_discover/runs/2026_04_30b_paper_experiments_v81"
PHASE="${1:-all}"
# All v8.1-minimal runs skip S4 (weight=0 in Occam-minimal weights)
SKIP_S4="S4_significance"

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

# ── Phase 1: Critical + sanity (5 parallel) ─────────────────────────────
if [[ "$PHASE" == "1" || "$PHASE" == "all" ]]; then
    echo "=== Phase 1: B1 sanity + MAIN + B4 + A3' + A8 ==="

    # Sanity baseline (fast): single zero-shot sample
    launch "B1_zero_shot" \
        config.n_iterations=1 \
        config.n_fresh=1 \
        config.n_revise=0 \
        config.skip_rl_update=True \
        config.disabled_signals="${SKIP_S4}"

    launch "MAIN" \
        config.disabled_signals="${SKIP_S4}"

    launch "B4_no_training" \
        config.skip_rl_update=True \
        config.disabled_signals="${SKIP_S4}"

    # A3' replaces old A3: ablate heaviest signal S8 instead of near-zero S1
    launch "A3p_no_s8_scope" \
        config.disabled_signals="${SKIP_S4},S8_scope"

    launch "A8_no_revision" \
        config.n_revise=0 \
        config.n_fresh=8 \
        config.train_on_fresh=True \
        config.disabled_signals="${SKIP_S4}"
fi

# ── Phase 2: Remaining baselines ─────────────────────────────────────────
if [[ "$PHASE" == "2" || "$PHASE" == "all" ]]; then
    echo "=== Phase 2: B2 + B3 ==="

    launch "B2_best_of_n" \
        config.n_iterations=1 \
        config.n_fresh=50 \
        config.n_revise=0 \
        config.skip_rl_update=True \
        config.disabled_signals="${SKIP_S4}"

    launch "B3_buffer_ttt_entropic" \
        config.n_revise=0 \
        config.n_fresh=8 \
        config.train_on_fresh=True \
        config.disabled_signals="${SKIP_S4}"
fi

# ── Phase 3: Remaining ablations ─────────────────────────────────────────
if [[ "$PHASE" == "3" || "$PHASE" == "all" ]]; then
    echo "=== Phase 3: A1 + A2 + A4 + A5 + A6 + A7 ==="

    launch "A1_no_hard_gates" \
        config.skip_hard_gates=True \
        config.disabled_signals="${SKIP_S4}"

    launch "A2_no_s9_focus" \
        config.disabled_signals="${SKIP_S4},S9_focus"

    launch "A4_best_of_1" \
        config.n_revision_candidates=1 \
        config.disabled_signals="${SKIP_S4}"

    launch "A5_no_skip_signal" \
        config.min_revision_attempts=999 \
        config.disabled_signals="${SKIP_S4}"

    launch "A6_no_ucb" \
        config.use_ucb=False \
        config.disabled_signals="${SKIP_S4}"
fi

echo ""
echo "All requested experiments launched. Monitor with:"
echo "  tail -f ${RUNS_DIR}/*/train/stdout.log"
echo "  # or"
echo "  bash shared/tools/monitor_v2.sh"
