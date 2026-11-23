# R1代码准备：Query诊断与简单掩码重建

worktree：`/root/autodl-tmp/projects/OWOBJ-R1`，分支`r1/query-mechanism-prepare`。
产物：`/root/autodl-tmp/experiments/R1`。
主项目、R0和依赖环境均只读；本次R1代码未自动提交。

[真实Query接口](../docs/query_interface.md)；[实验与统计协议](../docs/r1_protocol.md)。

## 当前CPU验收

```bash
cd /root/autodl-tmp/projects/OWOBJ-R1
bash r1/run_cpu_smoke.sh --stage common
bash r1/run_cpu_smoke.sh --stage r1-0
bash r1/run_cpu_smoke.sh --stage r1-1
```

三个套件独立，分别是共用接口、R1-0诊断、R1-1重建。
`--stage all`先验证共用接口，然后两个专属套件分别报告，任一专属失败不阻塞另一个。
CPU运行显式隐藏CUDA，单线程、nice19。
端到端合成文件/入口/绘图样例：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/root/autodl-tmp/experiments/R1/cpu_smoke/mpl_cache \
/root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python -m r1.cpu_demo --output /root/autodl-tmp/experiments/R1/cpu_smoke/new_demo
```

## B0全部结束后：共用接口

以下真实命令本轮未运行。真实运行会在B0未完成时拒绝启动。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/experiments/B0_PROB_OWOBJ/env
cd /root/autodl-tmp/projects/OWOBJ-R1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/root/autodl-tmp/experiments/R1/mpl_cache
python -m r1.validate_real --config r1/configs/diagnostic.json
python -u -m r1.extract --config r1/configs/diagnostic.json --output /root/autodl-tmp/experiments/R1/evaluation_cache
```

缓存默认1/3/6层。`--all-layers`支持全层，需先核对磁盘预算。
缓存不含GT，两个实验共用同一份评价缓存；训练不依赖这个缓存。

## R1-0独立运行

```bash
python -m r1.diagnose --config r1/configs/diagnostic.json --cache /root/autodl-tmp/experiments/R1/evaluation_cache --output /root/autodl-tmp/experiments/R1/diagnostic
python -m r1.plot --inputs /root/autodl-tmp/experiments/R1/diagnostic --output /root/autodl-tmp/experiments/R1/diagnostic_figures
```

## R1-1独立运行

可以先于R1-0诊断执行训练。每层一个AE，以下示例mask0.6；替换为0或0.3为独立消融。

```bash
for layer in 1 3 6; do
  python -u -m r1.train --config r1/configs/L${layer}_mask0.6.json
  python -m r1.evaluate --config r1/configs/L${layer}_mask0.6.json --cache /root/autodl-tmp/experiments/R1/evaluation_cache --checkpoint /root/autodl-tmp/experiments/R1/L${layer}_mask0.6/checkpoint.pth --output /root/autodl-tmp/experiments/R1/L${layer}_mask0.6/evaluation
done
python -m r1.plot --inputs /root/autodl-tmp/experiments/R1/L1_mask0.6/evaluation /root/autodl-tmp/experiments/R1/L3_mask0.6/evaluation /root/autodl-tmp/experiments/R1/L6_mask0.6/evaluation --output /root/autodl-tmp/experiments/R1/reconstruction_figures
```

`r1.evaluate`默认在CPU使用缓存评价，不需要加载DETR；完整数据仍等B0完成后再执行。
中断续训：`python -m r1.train --config ... --resume .../checkpoint.pth`。
输出/缓存/图表默认拒绝覆盖。不得把不同mask/归一化/容量/预算的结果混成一次层间比较。

主要文件：features.py（共用提取）；cache.py/extract.py（缓存）；analysis.py/diagnose.py（R1-0）；
reconstruction.py/train.py/evaluate.py（R1-1）；plot.py（读结果绘图）；reuse.py（R0通用工具只读复用）。
