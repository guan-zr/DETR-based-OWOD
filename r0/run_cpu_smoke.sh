#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/OWOBJ-R0
R0_ROOT=/root/autodl-tmp/experiments/R0
mkdir -p "$R0_ROOT/cpu_smoke/tmp" "$R0_ROOT/cpu_smoke/mpl_cache"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R0_ROOT/cpu_smoke/tmp" MPLCONFIGDIR="$R0_ROOT/cpu_smoke/mpl_cache"
nice -n 19 /root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python -u -m r0.cpu_smoke "$@"
