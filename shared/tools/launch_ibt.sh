#!/bin/bash
# Launch IBT training variants.
#
# Usage:
#   bash tools/launch_ibt.sh              # all 4 variants
#   bash tools/launch_ibt.sh single_chain # single-chain only (no OPD)
#   bash tools/launch_ibt.sh mini_grpo    # mini-GRPO only (no OPD)
#   bash tools/launch_ibt.sh opd          # both modes with OPD
#   bash tools/launch_ibt.sh all          # all 4 variants

set -e
cd "$(dirname "$0")/.."

RUN_DIR="runs/2026/4/ibt/1"
SCRIPT="src/co_scientist/trainers/ibt/train_ibt.py"

COMMON_ARGS="num_goals=20 num_turns=5 api_profile=NEW log_path=$(pwd)/$RUN_DIR"

MODE="${1:-all}"

launch() {
    local name="$1"
    local args="$2"
    local log_dir="$RUN_DIR/$name"
    mkdir -p "$log_dir/train"
    echo "=== Launching $name ==="
    nohup python3 "$SCRIPT" $args $COMMON_ARGS \
        > "$log_dir/train.log" 2>&1 &
    echo "  PID: $!  Log: $log_dir/train.log"
}

case "$MODE" in
    single_chain)
        launch "single_chain" "mode=single_chain"
        ;;
    mini_grpo)
        launch "mini_grpo" "mode=mini_grpo group_size=4"
        ;;
    opd)
        launch "single_chain_opd" "mode=single_chain use_opd=True"
        launch "mini_grpo_opd" "mode=mini_grpo group_size=4 use_opd=True"
        ;;
    all)
        launch "single_chain" "mode=single_chain"
        launch "mini_grpo" "mode=mini_grpo group_size=4"
        launch "single_chain_opd" "mode=single_chain use_opd=True"
        launch "mini_grpo_opd" "mode=mini_grpo group_size=4 use_opd=True"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [single_chain|mini_grpo|opd|all]"
        exit 1
        ;;
esac

echo ""
echo "Monitor with: tail -f $RUN_DIR/<variant>/train.log"
