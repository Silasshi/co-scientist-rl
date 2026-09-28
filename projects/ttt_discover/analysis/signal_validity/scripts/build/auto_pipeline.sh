#!/usr/bin/env bash
# Auto-pipeline: wait for perturbations → grade them → analyze
# Legacy v6 pipeline — kept for reference; superseded by per-version scripts
# under scripts/grade/ and scripts/analyze/.
set -e
cd /home/silas/co-scientist-project
source .env

BASE=projects/ttt_discover/analysis/signal_validity
PERTURB_LOG=$BASE/logs/perturbations2.log
PERTURB_FILE=$BASE/data/perturbations/perturbations.jsonl

echo "[pipeline] Step 1: Wait for perturbation generation to finish..."
while ! grep -q "Tokens:" $PERTURB_LOG 2>/dev/null; do
    sleep 60
done
echo "[pipeline] Perturbations done. Saved: $(wc -l < $PERTURB_FILE)"

echo "[pipeline] Step 2: Grade perturbations"
nohup python $BASE/scripts/grade/grade_perturbations_v6.py > $BASE/logs/grade_perturbations.log 2>&1

echo "[pipeline] Step 3: Run correlation analysis"
python $BASE/scripts/analyze/analyze_v6.py > $BASE/reports/v6/FINAL_ANALYSIS_v6.txt

echo "[pipeline] All stages complete. See reports/v6/FINAL_ANALYSIS_v6.txt"
