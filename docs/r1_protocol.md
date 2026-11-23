# R1独立机制验证协议

## 独立路径

共用提取接口正确后，R1-0与R1-1独立开发/测试/运行。

- R1-0：评价图像→冻结Query缓存→GT分组→范数/物体性/覆盖率诊断。
- R1-1训练：Task1训练图片→冻结Query→某一实际层→共享MLP AE；不读XML。
- R1-1评价：同一评价Query缓存+该层AE→离线GT分组→误差区分能力/排名分析。

R1-1不读取R1-0诊断结果、不依赖其绘图或可选线性探测器。
本次未实现可选线性探测器，因此没有Unknown标签监督训练的诊断分支。
范数诊断不能充分判断全部高维信息是否可分，也不能把LayerNorm后的范数接近视为没有物体性信息。

## 无标签Query重建

输入[B,Q,C]，同一层所有B*Q共享一个MLP；各层模型相互独立。
网络：拼接(masked_query, binary_mask)为2C维→128→32→128→C，GELU。
实际C256时每层107040参数。无Transformer、跨层重建、Query去噪或空间上下文模块。

目标detach；普通全维MSE训练，另记录masked-only误差。
未mask的真实零值与被mask的零值由二值mask通道区分。
无掩码时mask通道全0、masked误差缺失，不伪造为零误差。
AE optimizer只包含自身参数；检测器eval/no_grad/全冻结，不改变原损失或预测。

固定初始预算：每层2000步、batch1、seed91、AdamW lr1e-4/wd1e-4。
同一掩码率下1/3/6层配置只允许name/layer/output不同。
提供0、0.3、0.6三种掩码率；训练图像、容量、预算、resize相同。
默认输入l2表示，另支持raw/pre_layernorm；训练重建所选表示，不拟合测试集标准化参数。
固定最后一步权重，不根据Unknown GT选择模型、方向、掩码率或容量。

## 标注与候选集合

数据实际位于`datasets/OWOD_B0`，TOWOD为M-OWODB，Task1 Known=VOC20，Unknown=其余60类。
复用R0已核验的词表、GT读取、互斥分组和AUROC/AP工具，R0文件保持只读并以哈希锁定。
评价按唯一图像，原测试10246行/9119唯一图像；不修改B0的重复项官方检测评估协议。

GT和预测框在副本上裁剪到原图范围后计算分组IoU。
Known/Unknown：与全部GT最近IoU≥0.5且GT非difficult；Background：与全部可用GT（含Unknown/difficult）交集面积均为0；
其余、difficult匹配、无效值归Ambiguous/Ignore或Invalid并保留计数。
不把仅未匹配Known的Query当Background。
80类外物体和漏标可能存在，Background仅相对可用标注，不能视为穷尽可靠背景。

每层统计全部有效Query，以及该层按原Top100 pair规则去重的selected集合。
最终层为真实筛选，中间层为同规则离线模拟，导出selection_scope区分。
GT仅由离线诊断/评价读取；特征缓存不含GT分组，训练直接读训练图片不读评价缓存。

## 指标解释

- 每层三组数量、raw/pre-LN/L2范数、Gaussian概率及unknown分类sigmoid分布。
- Known/Unknown各自对Background的AUROC、阶梯average precision（AUPRC），含正负数量与阳性比例。
- 重建误差高/低两个方向均单独报告，明确为事后离线分析，不选更优方向宣称无监督性能。
- 各层比较AUROC/AP；未经校准的层间MSE不用于排名层优劣。
- 与两种原物体性信号的Pearson/Spearman相关；同一候选集合的排名位移、TopK重合度。
- 固定K=10/50/100，保存背景数/模糊数及有效K，避免只报告AUROC；K大于集合则截断并记录。
- Unknown GT在各自层预测框中的IoU≥0.3/0.5/0.7覆盖率。
- Gaussian/classifier低分数量使用预先固定0.5参考线：仅描述已有候选，非原模型真实过滤阈值。
- “有候选但最终unknown未覆盖”按唯一图像的几何IoU≥0.5判断，允许框重复覆盖GT，
  不等同于官方score取整、重复测试条目、一对一匹配下的U-Recall。
  Gaussian分数未在OWOBJ推理中直接相乘，不能据低Gaussian分数直接推断漏检因果。

## 文件与资源

缓存每图一个NPZ，保存raw/l2/pre-LN、逐层框/logits/分数/选择索引及图像几何信息，pickle关闭。
单次只读取一个特征文件；默认1/3/6层缓存预算12GiB，超限中止并保留incomplete标记，不静默跳过。
典型未压缩三种特征约0.88MiB/图，完整评价缓存约8GiB；可用--all-layers提取全层，但须调整磁盘预算。
诊断/误差评价导出queries.csv.gz、unknown_cases.csv.gz、ranking.jsonl、summary.json。
指标仅累计标量数组，不保留整个数据集的高维Query在内存。
绘图至多显示每组前20000个标量和每层2000个散点；图形明确为确定性显示子集，统计指标使用全部有效样本。

所有真实检测器操作均先检查B0完整训练、评估、Query分析、流水线退出和权重SHA一致；本轮只允许CPU合成。

## 2026-11-09 评价协议修复与来源校验

主配置固定`background_rule=disjoint_gt`，有交集但达不到原正例规则的框归Ambiguous；不按Unknown AUROC选择阈值。
`configs/iou_only/`仅保留原IoU-only分组对照；`configs/legacy_original/`逐字归档原JSON（其中batch2提取配置仅存档，当前入口拒绝运行）。
每份L*_mask*.json的`*_raw.json`是独立raw配置，与l2配置仅representation/name/output不同。
训练预算、AE、掩码与损失均不改变。主协议与IoU-only评价可以复用同一份完整重建权重，其他训练/提取/评分字段必须一致。

R1-1训练与评价缓存提取统一batch=1。两者使用同一ImageOnlyDataset和pack_images：官方确定性resize（800/1333）、
相同RGB/ImageNet归一化、NestedTensor按批最大尺寸补零。batch1的图像画布等于resized_hw，内部特征mask保留原实现。
`config_read`拒绝batch不一致；真实帧读取还核验resized_hw=padded_hw。

诊断和重建评价在输出/读重建权重前都通过FrameReader核验split角色、split文件名、文件字节SHA256、
完整去重图像ID集合（当前正式评价9119）、索引唯一性/数量/文件存在、batch/resize/归一化/层策略/关键源码SHA，
并核验冻结基线文件路径/SHA256及synthetic标志。每帧继续检查图像ID、逐层结构和实际padding。
旧缓存缺来源信息、子集缓存或错误缓存明确报错，不能靠补写manifest将batch2缓存变成batch1缓存；需要重新提取。
完整缓存manifest写入结果summary。特征缓存不含GT标签，raw/l2成对评价共享同一缓存。
评价输出`grouping_audit.json`，逐层all与selected池分别记录同候选几何规则迁移。

B0全流程门禁保留。B0结束后先运行`r1.validate_real`：同图batch1重复，
再与能改变该图padding的batch2比较真实Query（raw/l2/pre-LN）、own-layer框、logits及Gaussian distance，
逐层记录max_abs/RMS及固定rtol=1e-5、atol=1e-6结果。batch2比较是测量，不要求其与batch1相等。
未找到能改变padding的同批图像时明确失败；该检查本轮未运行，合成GroupNorm反例不代替真实结果。

详见[修复报告](../r1/protocol_repair/修复报告.md)。
