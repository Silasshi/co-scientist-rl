#!/bin/bash
# V3 Training Monitor — checks every 2 minutes for new batches
LOG_DIR="/home/silas/co-scientist-project/runs/2026/3/multiturn_v4/3"
SUMMARY="$LOG_DIR/train/batch_summary.jsonl"
SEEN=0

while true; do
    if ! pgrep -f 'train_multiturn_v4.*batch_size=8' > /dev/null; then
        echo "$(date): ALERT — training process stopped"
        tail -5 "$LOG_DIR/stdout.log" 2>/dev/null
        break
    fi

    if [ -f "$SUMMARY" ]; then
        CURRENT=$(wc -l < "$SUMMARY")
        if [ "$CURRENT" -gt "$SEEN" ]; then
            # New batch(es) completed
            tail -n $((CURRENT - SEEN)) "$SUMMARY" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    b = d['batch_idx']
    print(f'=== Batch {b} ===')
    print(f'  rubric={d[\"rubric/mean\"]:.3f} reward={d[\"reward/mean\"]:.3f} prm={d[\"prm_score/mean\"]:.2f}')
    print(f'  prm_dist={d[\"prm_score/dist\"]} collapse={d[\"prm_score/collapse_alert\"]}')
    print(f'  comp={d[\"conv/completed_rate\"]:.0%} disc={d[\"conv/discussion_turns_mean\"]:.1f} rev={d[\"conv/plan_revisions_mean\"]:.1f} turns={d[\"conv/turns_mean\"]:.1f}')
    print(f'  hints={d[\"hint/total\"]} opd={d[\"opd/sets_computed\"]} opd_kl={d[\"opd/mean_kl\"]:.4f} opd_contrib={d.get(\"opd/advantage_contribution\",0):.3f}')
    print(f'  forced={d[\"forced_plan/count\"]} datums={d[\"datums/total\"]} fmt={d[\"format/compliance_rate\"]:.0%} wc={d[\"length/mean\"]:.0f}')
    print(f'  time={d[\"time/total\"]:.0f}s')
"
            SEEN=$CURRENT
        fi
    fi
    sleep 120
done
