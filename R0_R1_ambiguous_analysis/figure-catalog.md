# 图目录与逐图解释

共 12 张独立 PNG；6 张核心密度图、2 张 Gaussian 辅助密度图、4 张 CDF。所有图的误差、IoU 和来源见 source_manifest.json；方法见 stats-appendix.md。

## R0_S2_四类分布.png

- 目的：定位 Known、Unknown、Background、Ambiguous 的相对分布。
- 数据来源：R0_S2，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：Background 主体偏低；Known/Unknown 偏右；Ambiguous 中位数介于背景和明确前景之间，同时保留与两侧的重叠。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R0_S2_前景Ambiguous背景三组分布.png

- 目的：简洁表达明确前景、Ambiguous、背景的位置和重叠。
- 数据来源：R0_S2，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：Background 主体偏低；Known/Unknown 偏右；Ambiguous 中位数介于背景和明确前景之间，同时保留与两侧的重叠。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R0_S2_Ambiguous按IoU分组分布.png

- 目的：检验 Ambiguous 内部异质性；观察随重叠变化的分布。
- 数据来源：R0_S2，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：IoU 子组的均值、中位数、P25/P75 均按组别递增；最低 IoU 组更靠近背景，但不与背景等同；高 IoU difficult 组明显偏高。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R0_S2_四类CDF.png

- 目的：核验分布曲线中的位置差异及长尾，避免只比较主峰。
- 数据来源：R0_S2，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：Background 主体偏低；Known/Unknown 偏右；Ambiguous 中位数介于背景和明确前景之间，同时保留与两侧的重叠。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R0_S2_Ambiguous按IoU分组CDF.png

- 目的：核验分布曲线中的位置差异及长尾，避免只比较主峰。
- 数据来源：R0_S2，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：IoU 子组的均值、中位数、P25/P75 均按组别递增；最低 IoU 组更靠近背景，但不与背景等同；高 IoU difficult 组明显偏高。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R1_L6_四类分布.png

- 目的：定位 Known、Unknown、Background、Ambiguous 的相对分布。
- 数据来源：R1_L6，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：Background 主体偏低；Known/Unknown 偏右；Ambiguous 中位数介于背景和明确前景之间，同时保留与两侧的重叠。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R1_L6_前景Ambiguous背景三组分布.png

- 目的：简洁表达明确前景、Ambiguous、背景的位置和重叠。
- 数据来源：R1_L6，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：Background 主体偏低；Known/Unknown 偏右；Ambiguous 中位数介于背景和明确前景之间，同时保留与两侧的重叠。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R1_L6_Ambiguous按IoU分组分布.png

- 目的：检验 Ambiguous 内部异质性；观察随重叠变化的分布。
- 数据来源：R1_L6，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：IoU 子组的均值、中位数、P25/P75 均按组别递增；最低 IoU 组更靠近背景，但不与背景等同；高 IoU difficult 组明显偏高。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R1_L6_四类CDF.png

- 目的：核验分布曲线中的位置差异及长尾，避免只比较主峰。
- 数据来源：R1_L6，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：Background 主体偏低；Known/Unknown 偏右；Ambiguous 中位数介于背景和明确前景之间，同时保留与两侧的重叠。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## R1_L6_Ambiguous按IoU分组CDF.png

- 目的：核验分布曲线中的位置差异及长尾，避免只比较主峰。
- 数据来源：R1_L6，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：IoU 子组的均值、中位数、P25/P75 均按组别递增；最低 IoU 组更靠近背景，但不与背景等同；高 IoU difficult 组明显偏高。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## Gaussian_四类分布.png

- 目的：定位 Known、Unknown、Background、Ambiguous 的相对分布。
- 数据来源：Gaussian，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：四类主峰均在约 0.25 附近且高度重叠；Known 存在高分长尾。Ambiguous 中位数略低于 Background，不支持简单的严格中间位置。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。

## Gaussian_前景Ambiguous背景三组分布.png

- 目的：简洁表达明确前景、Ambiguous、背景的位置和重叠。
- 数据来源：Gaussian，完整固定 Query 池；R0=第二轮 S2 trained73 overall；R1=formal_20261112 L6 L2 Mask0 seed91 full_error；Gaussian=同一 Query 的原始物体性分数。
- 图注：配置和实际 n 已标在图中；密度按组归一化、共用方法内带宽；CDF 为精确全样本经验累积比例；无误差条。
- 观察：四类主峰均在约 0.25 附近且高度重叠；Known 存在高分长尾。Ambiguous 中位数略低于 Background，不支持简单的严格中间位置。
- 含义：描述当前固定评估池中的分布差异；不能宣称因果物体性、无误分类或下游检测收益。
- 检查点：结合分位数表检查中心和尾部；不比较不同单位的密度峰高；IoU 只作为诊断分组；检查 difficult 混杂。
