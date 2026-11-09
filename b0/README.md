# B0: PROB + OWOBJ / M-OWODB / Task 1

Artifacts and both Chinese progress reports live in
`/root/autodl-tmp/experiments/B0_PROB_OWOBJ`.
The original supplied OWOBJ snapshot is the `vendor-owobj` branch. Original
PROB and source datasets remain untouched. No reconstruction experiments exist.

Use the B0 scripts, rather than the upstream multi-task `run.sh`:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/experiments/B0_PROB_OWOBJ/env
cd /root/autodl-tmp/OWOBJ-main
bash b0/launch.sh
```

This runs 41 epochs with the released `sketch` model, physical batch 2,
accumulation 10, effective batch 20. Batch 5 failed the worst-padding memory
test. The official M protocol is **TOWOD, 20 VOC known classes**, not the
19-class OWDETR recipe in the mislabeled upstream training script.

The pipeline resumes its own rolling checkpoint, evaluates through the official
entrypoint, then runs streaming query analysis and updates the reports. A lock
prevents concurrent training pipelines. Logs remain in `logs/`. Rolling state is
saved every 100 optimizer updates and at epoch boundaries; final and pre-LR-drop
checkpoints are retained. Intermediate checkpoints include optimizer, scheduler,
epoch/microbatch position and RNG state; worker augmentation order after restart
is not guaranteed bitwise equal to uninterrupted training.

`b0/features.py` provides read-only hooks. `b0/metrics.py` and `b0/analyze.py`
are offline-only: unknown test labels never enter training. Query statistics use
unique non-difficult GT; official evaluation preserves the original repeated
test IDs and its original denominator. The Gaussian score is for analysis and
soft training targets. Released OWOBJ inference uses classifier sigmoid Top100;
do not multiply the Gaussian score again.

The released model and C++/CUDA source are unmodified. Paper/code discrepancies,
including detached BN statistics in the released KL term, are documented in the
Chinese reports. Smoke metrics are diagnostic and are never full B0 results.
