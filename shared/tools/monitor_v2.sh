#!/bin/bash
# V2 Training Monitor — checks status every 10 minutes
LOG_DIR="/home/silas/co-scientist-project/runs/2026/3/multiturn_v4/2"
SUMMARY="$LOG_DIR/train/batch_summary.jsonl"
STDOUT="$LOG_DIR/stdout.log"

while true; do
    echo "=== $(date) ==="

    # Check if process is alive
    if ! pgrep -f 'train_multiturn_v4.*batch_size=8' > /dev/null; then
        echo "ALERT: Training process NOT running!"
        echo "Last 20 lines of stdout:"
        tail -20 "$STDOUT" 2>/dev/null
        break
    fi

    # Show latest batch metrics
    if [ -f "$SUMMARY" ]; then
        NBATCH=$(wc -l < "$SUMMARY")
        echo "Batches completed: $NBATCH"
        echo "Latest:"
        tail -1 "$SUMMARY" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
b = d.get('batch_idx', '?')
rubric = d.get('rubric/mean', 0)
reward = d.get('reward/mean', 0)
score_mean = d.get('student_score/mean', 0)
score_dist = d.get('student_score/dist', {})
collapse = d.get('student_score/collapse_alert', False)
comp = d.get('conv/completed_rate', 0)
waves = d.get('wave/count', 0)
w_datums = d.get('datums/wave_total', 0)
p_datums = d.get('datums/plan_total', 0)
forced = d.get('forced_plan/count', 0)
t = d.get('time/total', 0)
hints = d.get('hint/total', 0)
print(f'  Batch {b}: rubric={rubric:.3f} reward={reward:.3f} score={score_mean:.2f} comp={comp:.0%}')
print(f'  waves={waves} datums={w_datums}+{p_datums} forced_plan={forced} hints={hints} time={t:.0f}s')
print(f'  score_dist={score_dist} collapse={collapse}')
" 2>/dev/null
    else
        echo "No batch summary yet"
    fi

    echo ""
    sleep 600
done
