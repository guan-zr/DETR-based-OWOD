# Ambiguous 重建误差分析计划
## 目标
只读取正式 R0 第二轮 S2 trained73 overall B0 Query 与 R1 formal_20261112 L6 L2 Mask0 seed91 full_error，完成分布图、统计表与中文报告。
## 阶段
- [x] 创建分析目录与持久记录
- [x] 核验逐 Query 数据、标签规则、IoU 及原始计数
- [x] 计算描述统计与绘制 Matplotlib 分布曲线
- [x] 检查图像与一致性并生成报告
## 约束
不训练、不改原实验、不删文件、不提交 Git；新增文件仅写入本目录。
## 错误记录
- 默认 sandbox 启动失败（bwrap namespace），通过审查后的执行环境继续。
- rg 不可用，改用 find / Python / grep。
- 图像 QA 发现初版纵轴固定上界 1，已改为按全部曲线最大密度设置上界并增加断言，重新生成全部图。
## 状态
全部完成；12 张 PNG、两张主统计表、中文报告和来源/复现附件通过检查。
