#!/bin/bash
cd /home/silas/co-scientist-project
RUN=runs/2026/4/ibt/2
IBT=src/co_scientist/trainers/ibt
RESUMED=""

while true; do
    ALL_DONE=true

    for variant in single_chain mini_grpo single_chain_opd mini_grpo_opd; do
        echo "$RESUMED" | grep -q "$variant" && continue
        if grep -q "Training complete" "$RUN/$variant/train.log" 2>/dev/null; then
            echo "$(date): Resuming $variant..."
            case $variant in
                single_chain)     ARGS="mode=single_chain" ;;
                mini_grpo)        ARGS="mode=mini_grpo group_size=4" ;;
                single_chain_opd) ARGS="mode=single_chain use_opd=True w_opd=0.2" ;;
                mini_grpo_opd)    ARGS="mode=mini_grpo group_size=4 use_opd=True w_opd=0.2" ;;
            esac
            nohup python3 "$IBT/train_ibt.py" $ARGS num_goals=50 start_goal=25 num_turns=5 \
                api_profile=NEW log_path="$(pwd)/$RUN" init_checkpoint="$(pwd)/$RUN/$variant" \
                > "$RUN/$variant/train_resume.log" 2>&1 &
            echo "  PID: $!"
            RESUMED="$RESUMED $variant"
        else
            ALL_DONE=false
        fi
    done

    for variant in baseline_tf_accumulated baseline_tf_latest; do
        echo "$RESUMED" | grep -q "$variant" && continue
        if grep -q "Done\." "$RUN/$variant/train.log" 2>/dev/null; then
            echo "$(date): Resuming $variant..."
            case $variant in
                baseline_tf_accumulated) ARGS="accumulate_hints=True" ;;
                baseline_tf_latest)      ARGS="accumulate_hints=False" ;;
            esac
            nohup python3 "$IBT/baseline_training_free.py" $ARGS num_goals=50 start_goal=25 \
                num_turns=5 api_profile=NEW log_path="$(pwd)/$RUN" \
                > "$RUN/$variant/train_resume.log" 2>&1 &
            echo "  PID: $!"
            RESUMED="$RESUMED $variant"
        else
            ALL_DONE=false
        fi
    done

    if $ALL_DONE; then
        echo "$(date): All resumed. Done."
        break
    fi
    sleep 60
done
