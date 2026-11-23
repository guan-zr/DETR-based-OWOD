# R0 预先固定的最小验证协议

## 方法依据与范围

参考[OW-DAR官方论文](https://ojs.aaai.org/index.php/AAAI/article/view/38182)：
空间随机mask后重建特征，再在候选区域聚合误差。
[官方GitHub](https://github.com/lllhye/OW-DAR)本次检查仅有README和.gitignore，未提供可核对实现。
本实现是用户要求的DETR可行性基线，不声称复现OW-DAR完整系统或其论文指标。
不实现AREA动态权重、Weibull、Faster R-CNN架构或与OWOBJ融合。

## 成对训练配置

| 参数 | R0-S1 / R0-S2共同值 |
| --- | --- |
| 冻结检测器 | B0最终Task1 epoch40权重，strict加载，SHA256一致 |
| 尺度/通道 | 4级，每级256；各级独立但同容量AE |
| AE | Conv256→64→32(stride2)→32；bilinear上采样回原H/W；Conv32→64→256；GELU，无skip/BN |
| 掩码 | 有效空间位置独立Bernoulli，概率0.6，同位置通道共享 |
| 训练目标 | 普通MSE：先按通道均值，再对有效空间均值，最后4尺度等权平均 |
| 日志 | valid MSE、masked-only MSE、有效/掩码位置数；无掩码时masked MSE缺失 |
| 输入 | Task1训练图片；**训练Dataset不读取XML**；resize短边800/长边≤1333及官方归一化 |
| 预算 | 2000 optimizer steps，batch1，AdamW lr1e-4/wd1e-4，seed73 |
| 图像次序 | 同seed的确定性随机排列，S1/S2相同，断点续训保持步数对应关系 |
| 模型选择 | 最后一步；不根据Unknown GT或测试指标选权重/掩码率/容量 |

2000步为本轮准备的固定预算pilot，不声称完整OW-DAR训练复现。
S1/S2的配置只允许name/source/output不同；CPU测试核对其余字段完全一致。
额外提供R0-S1-no-mask/R0-S2-no-mask，训练和评价mask均为0，属于明确消融。

## 离线评价

- 使用`OWOD_B0/ImageSets/TOWOD/owod_all_task_test.txt`；R0描述性统计按唯一图像ID，9119张。
  B0官方检测评估原始10246行含1127额外重复记录的行为不变；R0不修改该文件。
- Known是VOC20，Unknown是同一标注体系剩余60类；difficult排除统计。
  本轮复用已完成B0全量数据审计，并只抽查6个XML，没有再次全量扫描。
- 标注支持80个基准类内的Known/Unknown评价；不穷尽开放世界物体，也没有背景GT。
  未标注物体可能进入与可用GT无交集的候选，必须把Background解释为“相对现有标注的背景”。
- 预测候选默认所有100个最终层Query框，不按分类/物体性过滤；可配置增加官方最终检测框池。
- GT-reference池含已标注GT，以及图像尺寸和固定seed产生的64个背景参考候选，经全部Known/Unknown/difficult GT过滤。
  背景参考框不参与训练，不伪称人工背景标注，也不在不同重建方法间改变候选几何。
- 预测框互斥分组：与全部GT最近IoU≥0.5且GT非difficult→Known/Unknown；与全部可用GT（含Unknown/difficult）的交集面积均为0→Background；其余Ambiguous排除。无效框单列invalid。
  原最大IoU<0.1规则仅在`configs/iou_only/`独立对照中使用，不能解释为可靠背景。
- GT和预测框均先在副本上裁剪到原图边界，再计算IoU与ROI误差，保证标签对应同一区域；CSV记录裁剪后的原图坐标，不改模型输出。
- 评价随机掩码率0.6，每图4次，图像ID哈希+seed固定，同S1/S2一致；先平均误差图，再ROI聚合。
  同时保存条件masked-only区域误差。无掩码消融单次评价。
- 主分数方向预先固定为高重建误差→更像前景，不利用Unknown GT翻转方向或选择超参。
- 每尺度独立统计；overall对同一候选4个均有效的尺度分数等权均值，不把尺度当作独立重复样本池。
- Unknown vs Background：AUROC、AUPRC；Known vs Background：AUROC（并额外提供AUPRC）。
  AUPRC使用按并列分数组合的阶梯积分average precision，不采用梯形插值。
  缺少任一组时输出null与计数，不产生虚构指标。
- 误差分布图、每图元信息、每区域CSV.gz、紧凑标量NPZ、JSON指标均保存。
  只保留标量分布供总体指标/绘图；不将每张Encoder特征完整缓存到磁盘。

## 当前验证边界

CPU已验证：hook约定、S2恢复token顺序、冻结/输出不变、MSE反向、所有掩码模式、
Padding排除、resize/stride/奇数padding/ROIAlign、互斥标签、AUROC/AP并列分数、图表、
无XML的训练Dataset、确定性断点图像次序、S1/S2公平配置、B0未完成时阻止真实运行。

待B0完整流水线结束后：最终权重strict加载、真实CUDA Forward、真实图像S1/S2特征与检测输出逐位一致、
真实重建训练/显存/吞吐、完整标注统计上的指标和可分性结论。
当前合成图表及数值仅用于软件检查，没有科研性能结论。

## 2026-11-09 最小协议修复

主配置使用`background_rule=disjoint_gt`；正例IoU=0.5不变。框边界接触但交集面积为0允许作为背景。
`gt_reference`仍由注释GT和过滤后背景提案构成；变为Ambiguous的随机提案保留在审计计数中，不加入背景。
评价额外输出`grouping_audit.json`：同一候选池上IoU-only与零交集的全部组数量、差值、迁移；不按评分挑选规则。
`python -m r0.audit_grouping`可在显式隐藏CUDA后独立复算GT候选数量，无须加载模型。
原始JSON逐字保存在`configs/legacy_original/`；`configs/iou_only/`是可运行的几何规则独立对照。
主协议和IoU-only可复用同一份完整AE权重；仅允许background_rule、name、output不同，其余参数必须匹配。
保留实际训练config和评价config供复核，不增加训练步数。

`include_final_detections=true`当前会在访问B0前明确报错。最终pair→唯一Query映射/去重修复前不得启用该正式对照。
官方后处理未改动。修复证据见[R0/R1修复报告](../../../OWOBJ-R1/r1/protocol_repair/修复报告.md)。
