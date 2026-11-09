#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/experiments/B0_PROB_OWOBJ
cd /root/autodl-tmp/OWOBJ-main
nohup bash b0/pipeline.sh >> "$ROOT/logs/pipeline.log" 2>&1 < /dev/null &
echo "B0 pipeline PID=$! ; log=$ROOT/logs/pipeline.log"
