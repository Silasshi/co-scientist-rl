#!/usr/bin/env bash
# Launch v8.1-minimal paper experiments under CR-v6 + minimal-mode goal/prompt.
#
# Changes vs launch_all_v81.sh:
#   - locus_based_edit=True  (CR-v6 locus-revision pipeline is default)
#   - max_tokens=4096        (default 2048 truncated plans mid-methodology,
#                              causing hard-gate failure on smoke — fixed 2026-04-17)
#   - n_iterations=25        (override default 50; corrupted runs plateaued 8-19)
#   - Active goal is Goel-style minimal scenario (loaded from
#     analysis/sanity_check/research_goal.txt; no launcher override needed)
#   - Stitching bug in apply_locus_revisions fixed 2026-04-17 via
#     _spans_complete_block validation: partial-span quotes now reject
#     (fall back to no-op revision) instead of producing duplicated sections.
#
# Run dir: projects/ttt_discover/runs/2026_04_30b_paper_experiments_v81/*_v6_minimal/
#
# Usage: bash projects/ttt_discover/paper_experiments/launch_all_v81_minimal.sh [phase]
#   phase=1   MAIN_v6_minimal + B4_v6_minimal + B1_v6_minimal (the 3 paper runs)
#   phase=all (default) same as phase=1

set -euo pipefail

TRAINER="python src/co_scientist/ttt_discover/train_critique_revise.py"
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

# ── Phase 1: MAIN + B4 + B1 (the 3 paper runs) ──────────────────────────────
if [[ "$PHASE" == "1" || "$PHASE" == "all" ]]; then
    echo "=== Phase 1: MAIN_v6_minimal + B4_v6_minimal + B1_v6_minimal ==="

    # B1 zero-shot baseline (fast): 4 fresh samples, no revisions, no RL
    launch "B1_v6_minimal" \
        config.n_iterations=1 \
        config.n_fresh=4 \
        config.n_revise=0 \
        config.skip_rl_update=True \
        config.locus_based_edit=True \
        config.max_tokens=4096 \
        config.disabled_signals="${SKIP_S4}"

    # MAIN: full CR-v6 + minimal goal/prompt + stitching-fix, 25 iter
    launch "MAIN_v6_minimal" \
        config.n_iterations=25 \
        config.locus_based_edit=True \
        config.max_tokens=4096 \
        config.disabled_signals="${SKIP_S4}"

    # B4 no-training baseline: pure in-context learning, no gradient updates
    launch "B4_v6_minimal" \
        config.n_iterations=25 \
        config.skip_rl_update=True \
        config.locus_based_edit=True \
        config.max_tokens=4096 \
        config.disabled_signals="${SKIP_S4}"
fi

echo ""
echo "All requested experiments launched. Monitor with:"
echo "  tail -f ${RUNS_DIR}/*_v6_minimal/train/stdout.log"
echo "  # or"
echo "  bash shared/tools/monitor_v2.sh"
