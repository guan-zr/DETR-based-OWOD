# R0 实验笔记

修复报告位于 R1/r1/protocol_repair/修复报告.md，只读；R0 36 个文件匹配 after_source_sha256.json。B0 独立恢复通过 11 指标且保留旧失败，需只改 R0 门禁。证据和原源码备份：experiments/R0/formal_20261111。

后台流水线已启动：PID30975；训练均固定2000步，S1→S2→共同全量9119图流式评价→统计/报告。最终报告仅完整验收后生成。新增只修改原runtime.py门禁/官方后端，其他既有方法模块未改；新增流水线、成对流式提取、评价和报告脚本。最终状态以pipeline/status.json及每阶段commit为准。

2026-11-12复核：pipeline completed，所有阶段exit0，昨晚23:19:43完成；验收评价所有chunk、analysis所有文件与report SHA通过。主规则overall GT-reference S1 Unknown AUROC.925559/AP.628377，S2 .872645/.496850；All-query S1 .860795/.205401，S2 .892601/.248624。B类11434区域S1 .924917/S2 .858280。S1在三个原图面积层及小于1%相对面积遗漏Unknown层仍有信号；尺寸/形状混杂未完全排除。已完成计划。
