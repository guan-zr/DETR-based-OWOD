#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/OWOBJ-R1
R1_ARTIFACTS=/root/autodl-tmp/experiments/R1/cpu_smoke
mkdir -p "$R1_ARTIFACTS/tmp" "$R1_ARTIFACTS/mpl_cache"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R1_ARTIFACTS/tmp" MPLCONFIGDIR="$R1_ARTIFACTS/mpl_cache"
nice -n 19 /root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python -m r1.cpu_smoke "$@"
