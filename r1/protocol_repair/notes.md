# 核查记录

依据 experiments/R1/R0_R1_实验协议核查报告.md。R0 r0/、R1 r1/及两份docs是既有未提交文件，原内容按哈希归档。

- 共用分组在R0 labels，R1通过reuse及依赖哈希门禁使用；主规则零交集、iou_only独立对照。
- GT-reference保持注释GT＋筛选背景，Ambiguous随机提案只进入几何审计。
- 全量583616提案中：背景372966→137507；235459转Ambiguous；正例数不变。48图背景1878→683，复现旧统计。
- R1训练/提取统一batch1、相同resize/normalization/pack_images；真实Query batch诊断保留但本轮不运行。
- 诊断与重建评价均校验split SHA、完整ID集合、提取设置/源码和baseline；测试错误来源均明确报错。
- 原14份JSON及修改前源码逐字归档；raw/l2共9对；IoU-only共14份配置。
- R0 21、R1 23项单元测试及原smoke入口、R1端到端均通过；CUDA未初始化。
- train/reconstruction/features SHA及B0 guard/load_detector/pack_images/sampler AST未变；不改B0，不正式训练，不提交Git。
- final_detection去重待实现，以入口门禁禁止提前启用。其余P1算法限制及真实验证要求见修复报告。
