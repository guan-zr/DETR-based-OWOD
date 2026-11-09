#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/experiments/B0_PROB_OWOBJ
cd /root/autodl-tmp/OWOBJ-main
exec 9>"$ROOT/pipeline.lock"
flock -n 9 || { echo 'B0 pipeline already running'; exit 1; }
echo "$$" > "$ROOT/pipeline.pid"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=disabled MPLBACKEND=Agg
PY="$ROOT/env/bin/python"
trap 'rc=$?; printf "B0 pipeline failed: exit=%s line=%s\n" "$rc" "$LINENO" | tee "$ROOT/reports/pipeline_failure.txt"; "$PY" -m b0.write_reports; exit "$rc"' ERR
"$PY" -u -m b0.train >> "$ROOT/logs/train.log" 2>&1
"$PY" -m b0.write_reports
bash b0/eval_task1.sh >> "$ROOT/logs/official_eval.log" 2>&1
"$PY" -u -m b0.analyze --checkpoint "$ROOT/checkpoints/task1/checkpoint0040.pth" >> "$ROOT/logs/query_analysis.log" 2>&1
"$PY" -m b0.write_reports
echo B0_PIPELINE_COMPLETE
