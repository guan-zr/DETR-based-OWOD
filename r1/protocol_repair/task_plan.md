# R0/R1 最小协议修复计划

## 目标与边界
修复可靠背景定义、R1 batch/padding 一致性和缓存来源校验；保留原协议对照。只写 R0/R1 worktree；不干预 B0、不启动 CUDA/正式训练、不改损失或预算、不提交 Git。

## 阶段
- [x] 读取报告及调用路径，记录原文件哈希
- [x] 修改共用分组并输出规则迁移统计
- [x] 统一 R1 batch=1、校验提取配置和缓存来源
- [x] 准备原协议及 raw/l2 对照，处理 final_detection 启用边界
- [x] 运行 CPU 单元测试与 GT 统计复核，写交付报告

## 错误记录
- sandbox namespace 创建失败；受控 require_escalated 命令执行。
- rg 未安装，使用 find/grep/Python。

- 首轮 R1 测试：新增成功路径 fixture 缺 XML；修正为复用 fixture GT loader，生产校验逻辑无改动。

- 初次追加真实验证入口时命令工作目录不符；改为绝对路径，未改动其他文件。

## 当前状态
已完成：R0 21/R1 23 项 CPU 测试、默认 smoke、R1 端到端、48图与全量9119图GT统计、配置及归档复核；真实B0验证单列为后续要求。
