# B1 Task2 runner

All configuration/output paths are under `/root/autodl-tmp/experiments/B1_PROB_OWOBJ`.

- `audit.py`: source/strict checkpoint load, full split inventory, measured storage budget.
- `replay.py`: official learned selection and merge, training data only; no Task1 updates.
- `train.py`: B0 FP32 single-GPU accumulation; official stage limits and loss. New optimizer for pretrain; B1 optimizer/scheduler/RNG/offset for resume.
- `evaluate.py`: original official main and engine; non-mutating raw query observer; correct Task2 known40.
- `pipeline.py`: serial GPU execution, file lock, background session, real exit codes, resource monitor, completion gates.

New execution compatibility: TOWOD20/20 rather than mislabeled upstream OWDETR19/21, fixed CLI LR, selection flag enabled, B1 writable split view, seed-per-example augmentation for deterministic resume, bounded checkpoint retention. No architecture/loss/inference method changes. Official freeze flag ineffectiveness is preserved, not silently repaired.

Run `configs/launch.sh` with setsid; resume with `configs/resume.sh` with setsid. Do not run these concurrently. Do not use B0 `b0.train`/`b0.pipeline` from this worktree (those are inherited history and contain B0 paths). The B1 coordinator only calls `b1.*` modules, except read-only imports of the common optimizer and offline IoU helpers.
