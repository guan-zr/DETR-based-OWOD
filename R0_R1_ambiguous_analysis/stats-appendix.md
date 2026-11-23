# 统计与绘图方法附录

## 范围与单位

完整评估池：9,119 张唯一图像 × 每图 100 个 Query = 911,900。Foreground=Known+Unknown，共 161,909；Ambiguous=462,437；Background=287,554。每个 Query 在所属组内等权，每条密度曲线面积为 1。密度高度不代表某类占总体的比例。图例保留实际样本数。

R0/R1 误差单位不同，不能凭横轴绝对大小或密度峰高比较方法优劣。R0：Encoder S2 四尺度独立 AE 的空间误差经支持归一化 ROI 聚合，再取四尺度原始算术平均；采用第二轮 seed73 trained AE 的 overall。R1：L2 归一化的 L6 特征、Mask0、seed91，full_error 是 256 维预测与目标的均方误差。Mask0 的 masked_error 未定义，本轮未读取或替代 full_error。Gaussian = exp(-(1.3/256)*pred_obj)，高分代表模型物体性分数高，非重建偏差。

## 标签与区间

遵循已完成实验的 disjoint_gt 标签。Background：有效框与全部已提供 GT 无正面积交集；Known/Unknown：匹配最大 IoU≥0.5，且该最大匹配 GT 非 difficult；剩余相交框为 Ambiguous。低 IoU 但相交的候选仍属 Ambiguous。区间为 [0,0.1)、[0.1,0.3)、[0.3,0.5)、[0.5,1]，无边界重复或遗漏。高 IoU Ambiguous 的 5,639 个 Query 位于 1,159 张图，匹配 GT 均为 difficult；本规则未发现另外的 ignore 分支。

## 密度与 CDF

使用 Matplotlib，不导入 seaborn。使用全部数据构造 4,096 个等宽箱，按组除以 n×箱宽，再进行 Gaussian 核卷积，得到直方图分箱近似 KDE。不是对每个点逐项计算的精确 KDE；没有抽样。带宽 h=0.9×min(总体样本标准差, 总体IQR/1.349)×N^(-1/5)，在同一方法的所有组、所有密度图中固定。零端反射；Gaussian 的一端也反射。误差横轴覆盖 0 到实际最大值+4h，Gaussian 覆盖 [0,1]；不截尾、不删异常值、不做对数变换。卷积截断为 4 个带宽。每条曲线数值积分为 1，纵轴上界高于全部曲线峰值。

| 方法 | 共同带宽 h |
| --- | --- |
| R0_S2 | 0.00455521325411 |
| R1_L6 | 1.00187277592e-05 |
| Gaussian | 0.000688055226007 |

CDF 使用全部样本的精确经验累积比例，保留重复值权重，不平滑、不截尾，末端均为 1。CDF 纵轴为 Cumulative Probability，其余分布图纵轴为 Probability Density。所有 PNG 为 360 dpi，3,888×2,124 像素，白底、外置图例，使用经字形核验的 Noto Sans CJK SC。

## 描述统计与效应量

CSV 保留均值、样本标准差（ddof=1）、最小最大值、P01/P05/P25/P50/P75/P95/P99、IQR、Query 数和出现该组的图像数。分位数采用 NumPy 默认线性插值。下表以“组中位数减背景中位数，再除以背景 IQR”描述相对偏移；这不是检验统计量，也不用于方法排名。

| 方法 | 对比 | 中位数差 | 差 / 背景IQR |
| --- | --- | --- | --- |
| R0_S2 | Foreground_minus_Background | 0.104721434 | 2.2986 |
| R0_S2 | Ambiguous_minus_Background | 0.071182413 | 1.5624 |
| R1_L6 | Foreground_minus_Background | 0.000233897248 | 2.2040 |
| R1_L6 | Ambiguous_minus_Background | 0.000104486688 | 0.9846 |
| Gaussian | Foreground_minus_Background | 0.00388368964 | 0.2502 |
| Gaussian | Ambiguous_minus_Background | -0.00330358744 | -0.2128 |

不计算显著性、p 值或置信区间：目标是完整固定评估池的描述，且每图 Query 存在相关性；单种子结果不能代表种子不确定性。不以未校正的逐 Query 检验声称泛化显著差异。没有推断性对比族，因此本轮不存在多重检验校正问题。若后续进行总体推断，应按图像聚类并纳入训练种子。

## 低值区域的辅助描述

以明确前景 P25 为相对低值参照，R0 有 88.50% 的 Background 低于 0.134215，R1 有 86.61% 的 Background 低于 0.000320664。背景 P75 仅作为本轮分布位置参照，不作为部署阈值、候选筛选器或验证后的最佳阈值。完整比例在 distribution_mass_diagnostics.csv。

## 局限

可用 GT 只有已审计的 80 类标注，所谓明确背景是相对于这些标注而言，不能保证画面中不存在未标注物体。框面积、宽高比、类别比例、纹理和 difficult 等均可能影响误差；当前分布不控制这些混杂因素。高 IoU difficult 组不得解释为单纯 IoU 的效应。均值/分位数递增不意味着分布处处不交叉，也不意味着每个 Query 的严格单调关系。
