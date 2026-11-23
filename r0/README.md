# R0 空间重建式Objectness：代码准备

当前主评价采用零GT交集背景；[协议修复报告](../../OWOBJ-R1/r1/protocol_repair/修复报告.md)记录原规则对照、CPU测试与数量变化。final_detection在Query去重修复前被明确禁用。

独立worktree：`/root/autodl-tmp/projects/OWOBJ-R0`，分支`r0/spatial-reconstruction-prepare`。
只新增r0目录，未改B0主工作区或环境，也未自动提交R0改动。
实际数据复用`/root/autodl-tmp/datasets/OWOD_B0`，没有复制图片或权重。

接口和统计规则分别见 [feature_interfaces.md](docs/feature_interfaces.md) 与 [protocol.md](docs/protocol.md)。

| 文件 | 功能 |
| --- | --- |
| features.py / types.py | 冻结检测器的只读S1/S2提取、几何与多尺度数据结构 |
| reconstruction.py | 轻量卷积AE、空间随机mask、无类别MSE与masked统计 |
| regions.py | 原图→resize→stride坐标、ROIAlign、支持图加权均值 |
| data.py / labels.py | 无标注训练图片接口、离线GT读取和互斥候选分组 |
| train.py / runtime.py | 后续真实训练、断点、严格权重加载和B0完成门禁 |
| evaluate.py | 流式GT/Query候选评价、4次固定mask、逐尺度与总体结果 |
| metrics.py / visualize.py / summarize.py | AUROC/AP、分布图及CSV离线重新统计 |
| validate_detector.py | B0结束后真实权重预测一致性及新AE真实设备一步更新检查 |
| cpu_smoke.py / tests/test_cpu.py | 当前可执行的CPU合成Smoke |

## 当前可执行

```bash
cd /root/autodl-tmp/projects/OWOBJ-R0
bash r0/run_cpu_smoke.sh
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python -m r0.plan --config r0/configs/R0-S1.json
```

CPU Smoke用单线程、nice19和空CUDA_VISIBLE_DEVICES；输出在`experiments/R0/cpu_smoke/verified`。
训练/真实模型验证/真实评价会在B0完整训练、官方评价和Query分析全部完成且后台流水线退出前拒绝启动。

## 等B0全部完成后执行

以下命令本轮**未执行**。先验证真实检测器接口，再分别执行两个成对实验。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/experiments/B0_PROB_OWOBJ/env
cd /root/autodl-tmp/projects/OWOBJ-R0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/root/autodl-tmp/experiments/R0/mpl_cache
python -m r0.validate_detector --config r0/configs/R0-S1.json
python -u -m r0.train --config r0/configs/R0-S1.json
python -u -m r0.evaluate --config r0/configs/R0-S1.json --checkpoint /root/autodl-tmp/experiments/R0/R0-S1/checkpoint.pth
python -u -m r0.train --config r0/configs/R0-S2.json
python -u -m r0.evaluate --config r0/configs/R0-S2.json --checkpoint /root/autodl-tmp/experiments/R0/R0-S2/checkpoint.pth
```

无掩码消融替换相应config文件。每个实验默认拒绝覆盖已有输出。
中断续训使用同一配置：

```bash
python -m r0.train --config r0/configs/R0-S1.json --resume /root/autodl-tmp/experiments/R0/R0-S1/checkpoint.pth
python -m r0.summarize --records /root/autodl-tmp/experiments/R0/R0-S1/evaluation/regions.csv.gz --output /root/autodl-tmp/experiments/R0/R0-S1/recomputed
```

正式评价图及AUROC/AUPRC只能在上述真实实验后得到；当前CPU合成分数不能用于判断方法是否有效。
