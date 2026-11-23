# 分析审计与可用结论

研究问题：DETR 中的重建误差能否区分前景与背景；对部分包含物体信息的 Ambiguous Query，是否也能体现一定的物体性差异。

单位是 Query，来自同一组 9,119 张图中的每图全部 100 个最终 Query。R0 仅第二轮 S2 trained73 overall；R1 仅 formal_20261112 L6 L2 Mask0 seed91 full_error。每种方法一个固定训练种子。本轮为完整评估池的描述性分析，不进行 Query 独立性假设下的显著性检验，也不计算 AUROC、Recall@K 或候选筛选指标。

## 核验

| 组别 | R0 第二轮 | R1 正式 | 与原统计一致 |
| --- | --- | --- | --- |
| known | 140,527 | 140,527 | 是 |
| unknown | 21,382 | 21,382 | 是 |
| background | 287,554 | 287,554 | 是 |
| ambiguous | 462,437 | 462,437 | 是 |
| invalid | 0 | 0 | 是 |

R0/R1 全部裁剪框、分组标签和最大 IoU 逐 Query 完全相等；Gaussian 也逐项完全相等。无重复键、无缺失误差、无有效性剔除。R0 的 IoU 由原候选几何缓存恢复，所有重建误差只取第二轮数组。没有执行训练或前向推理。

## Claim Candidates

- Claim：明确背景误差偏低，Known/Unknown 向较高误差移动。
  - Source evidence：两种方法的 summary_statistics.csv、四类/三组分布图、四类 CDF。
  - Allowed wording：当前固定评估池支持组间分布偏移，背景主要集中在相对低误差区域。
  - Forbidden stronger wording：高误差能无误地识别物体，或所有背景都具有低误差。
  - Uncertainty：组间重叠和背景长尾仍存在；单种子、同图相关 Query。
  - Next check：需要时在独立数据和种子上检验稳健性；本轮不启动新实验。
  - Decision：keep。

- Claim：Ambiguous 存在内部异质性，IoU 子组均值、中位数、P25/P75 均随组别递增。
  - Source evidence：ambiguous_iou_group_stats.csv，IoU 分组曲线与 CDF。
  - Allowed wording：按重叠诊断分组后，高重叠子组的误差分布向前景侧移动，提示潜在物体性诊断价值。
  - Forbidden stronger wording：IoU 是严格物体性标签，或提高 IoU 必然导致任意单个 Query 误差增加。
  - Uncertainty：尺度、类别、难度及同图候选相关性未控制；高 IoU 组全为 difficult。
  - Next check：必要时用固定几何/类别条件检查，当前只报告观察到的关联。
  - Decision：keep，但保持诊断性、关联性表述。

- Claim：Ambiguous 的总体中心位于背景与明确前景之间。
  - Source evidence：三组图和中位数表；R0 0.061067 < 0.132249 < 0.165788；R1 0.191227 < 0.295713 < 0.425246（×10⁻³）。
  - Allowed wording：总体中位数位于两者之间；R0 的中位数位置更靠前景，R1 的中位数位置略靠背景。
  - Forbidden stronger wording：整个 Ambiguous 分布与前景等价，或所有 Ambiguous 都包含可检测物体。
  - Uncertainty：中心接近不代表完整分布距离接近，各组尾部和混合比例不同。
  - Next check：结合子组和 CDF，不能只看峰值。
  - Decision：keep，明确所指是中位数位置。

- Claim：高误差候选能提高最终检测性能。
  - Source evidence：本轮没有检测性能对照。
  - Allowed wording：尚未验证。
  - Forbidden stronger wording：已证明提高 Unknown Discovery、OWOBJ 或最终检测性能，或优于所有物体性模型。
  - Uncertainty：未经下游验证。
  - Next check：另行设计下游实验，本轮不执行。
  - Decision：discard。
