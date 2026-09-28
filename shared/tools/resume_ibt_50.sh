#!/bin/bash
# Monitor running IBT variants and resume each for goals 25-49 when it finishes.
# Run this in background: nohup bash tools/resume_ibt_50.sh &

set -e
cd /home/silas/co-scientist-project

RUN=runs/2026/4/ibt/2
IBT=src/co_scientist/trainers/ibt
COMMON="num_goals=50 start_goal=25 api_profile=NEW log_path=$(pwd)/$RUN"

# Training variants that need their own checkpoint
declare -A TRAINING_VARIANTS
TRAINING_VARIANTS=(
    ["single_chain"]="mode=single_chain"
    ["mini_grpo"]="mode=mini_grpo group_size=4"
    ["single_chain_opd"]="mode=single_chain use_opd=True w_opd=0.2"
    ["mini_grpo_opd"]="mode=mini_grpo group_size=4 use_opd=True w_opd=0.2"
)

# Training-free variants (no checkpoint needed, use base model)
declare -A TF_VARIANTS
TF_VARIANTS=(
    ["baseline_tf_accumulated"]="accumulate_hints=True"
    ["baseline_tf_latest"]="accumulate_hints=False"
)

# Track what we've already resumed
RESUMED=""

while true; do
    ALL_DONE=true

    # Check training variants
    for variant in "${!TRAINING_VARIANTS[@]}"; do
        if echo "$RESUMED" | grep -q "$variant"; then
            continue
        fi
        finished=$(grep -c "Training complete" "$RUN/$variant/train.log" 2>/dev/null || echo 0)
        if [ "$finished" -gt 0 ]; then
            echo "$(date): $variant finished 25 goals. Resuming for 25-49..."
            args="${TRAINING_VARIANTS[$variant]}"
            nohup python3 "$IBT/train_ibt.py" $args $COMMON \
                init_checkpoint="$(pwd)/$RUN/$variant" \
                > "$RUN/$variant/train_resume.log" 2>&1 &
            echo "  PID: $!"
            RESUMED="$RESUMED $variant"
        else
            ALL_DONE=false
        fi
    done

    # Check training-free variants
    for variant in "${!TF_VARIANTS[@]}"; do
        if echo "$RESUMED" | grep -q "$variant"; then
            continue
        fi
        finished=$(grep -c "Done\." "$RUN/$variant/train.log" 2>/dev/null || echo 0)
        if [ "$finished" -gt 0 ]; then
            echo "$(date): $variant finished 25 goals. Resuming for 25-49..."
            args="${TF_VARIANTS[$variant]}"
            nohup python3 "$IBT/baseline_training_free.py" \
                num_goals=50 start_goal=25 num_turns=5 $args api_profile=NEW \
                log_path="$(pwd)/$RUN" \
                > "$RUN/$variant/train_resume.log" 2>&1 &
            echo "  PID: $!"
            RESUMED="$RESUMED $variant"
        else
            ALL_DONE=false
        fi
    done

    if $ALL_DONE; then
        echo "$(date): All variants resumed. Exiting monitor."
        break
    fi

    sleep 60
done
