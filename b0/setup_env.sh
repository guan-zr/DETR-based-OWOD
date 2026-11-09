#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/experiments/B0_PROB_OWOBJ
REPO=/root/autodl-tmp/OWOBJ-main
if [[ ! -x "$ROOT/env/bin/python" ]]; then
    /root/miniconda3/bin/conda create --prefix "$ROOT/env" --clone /root/miniconda3 --offline -y
fi
"$ROOT/env/bin/python" -m pip install --no-cache-dir -r "$REPO/b0/requirements.txt" pypdf==4.3.1
cd "$REPO/models/ops"
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
    "$ROOT/env/bin/python" -m pip install --no-deps --no-build-isolation .
"$ROOT/env/bin/python" -m pip freeze > "$ROOT/reports/pip_freeze.txt"
/root/miniconda3/bin/conda env export --prefix "$ROOT/env" > "$ROOT/reports/conda_environment.yml"
