#!/bin/bash
LOG_DIR="/home/silas/co-scientist-project/runs/2026/3/multiturn_v4/4"
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
            tail -n $((CURRENT - SEEN)) "$SUMMARY" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    b = d['batch_idx']
    print(f'=== Batch {b} ({d[\"time/total\"]:.0f}s) ===')
    print(f'  rubric={d[\"rubric/mean\"]:.3f} reward={d[\"reward/mean\"]:.3f} prm={d[\"prm_score/mean\"]:.2f}')
    print(f'  prm_dist={d[\"prm_score/dist\"]} fallback={d[\"prm/parse_fallback_count\"]}')
    print(f'  comp={d[\"conv/completed_rate\"]:.0%} disc={d[\"conv/discussion_turns_mean\"]:.1f} rev={d[\"conv/plan_revisions_mean\"]:.1f}')
    print(f'  waves={d[\"wave/count\"]} stuck={d.get(\"conv/stuck_injections\",0)} forced={d[\"forced_plan/count\"]}')
    print(f'  hints={d[\"hint/total\"]} opd={d[\"opd/sets_computed\"]} opd_c={d.get(\"opd/advantage_contribution\",0):.3f}')
    print(f'  datums={d[\"datums/wave_total\"]}+{d[\"datums/plan_total\"]} fmt={d[\"format/compliance_rate\"]:.0%}')
"
            SEEN=$CURRENT
        fi
    fi
    sleep 120
done
