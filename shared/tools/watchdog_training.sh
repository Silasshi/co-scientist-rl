#!/bin/bash
# Watchdog: monitors v1 and v2 training, restarts if they die.
# Run with: nohup bash tools/watchdog_training.sh > tools/watchdog.log 2>&1 &

cd /home/silas/co-scientist-project
source tools/use_api_profile.sh OLD

LOG="tools/watchdog.log"

check_and_restart() {
    local name="$1"
    local pattern="$2"
    local cmd="$3"

    if ! pgrep -f "$pattern" > /dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [$name] Process died. Restarting..." >> "$LOG"
        eval "$cmd"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [$name] Restarted (PID: $!)" >> "$LOG"
    fi
}

echo "$(date '+%Y-%m-%d %H:%M:%S') Watchdog started" >> "$LOG"

while true; do
    # Check v1
    check_and_restart "v1" "train_multiturn.py.*run4" \
        "nohup python src/co_scientist/trainers/multiturn/train_multiturn.py config.batch_size=8 config.group_size=2 config.log_path=runs/2026/3/multiturn/run4 >> runs/2026/3/multiturn/run4_console.log 2>&1 &"

    # Check v2
    check_and_restart "v2" "train_multiturn_v2.py.*run1" \
        "nohup python src/co_scientist/trainers/multiturn/train_multiturn_v2.py config.batch_size=8 config.group_size=2 config.log_path=runs/2026/3/multiturn_v2/run1 >> runs/2026/3/multiturn_v2/run1_console.log 2>&1 &"

    # Check curriculum generator
    check_and_restart "curriculum" "generate_curriculum.py.*resume=true" \
        "nohup python src/co_scientist/trainers/multiturn/generate_curriculum.py config.split=train config.resume=true config.concurrency=8 >> data/curriculum/generation_train_parallel.log 2>&1 &"

    # Status every 30 min
    MINUTE=$(date +%M)
    if [ "$MINUTE" = "00" ] || [ "$MINUTE" = "30" ]; then
        V1_BATCHES=$(wc -l < runs/2026/3/multiturn/run4/train/batch_summary.jsonl 2>/dev/null || echo 0)
        V2_BATCHES=$(wc -l < runs/2026/3/multiturn_v2/run1/train/batch_summary.jsonl 2>/dev/null || echo 0)
        CURR=$(wc -l < data/curriculum/ml_train_curriculum.jsonl 2>/dev/null || echo 0)
        echo "$(date '+%Y-%m-%d %H:%M:%S') STATUS: v1=${V1_BATCHES} batches, v2=${V2_BATCHES} batches, curriculum=${CURR} goals" >> "$LOG"
    fi

    sleep 300  # Check every 5 minutes
done
