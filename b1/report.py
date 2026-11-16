"""Generate reports only from on-disk evidence; missing stages remain explicitly pending."""
import json, os, subprocess
from pathlib import Path
from b1.common import *

def get(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def generate(final=False):
 A=get(E/'run_state/stage_a_complete.json');B=get(E/'run_state/stage_b_complete.json')
 ea=get(E/'evaluation/stage_a/manifest.json');eb=get(E/'evaluation/stage_b/manifest.json');status=get(E/'run_state/pipeline.json',{})
 orig=get(B0/'evaluations/final/official_metrics.json');baseline_query=get(B0/'query_analysis/final/summary.json',{});baseline_coverage=baseline_query.get('coverage',{}).get('unknown',{}).get('all',{}).get('0.5');storage=get(E/'reports/storage_budget.json',{});environment=get(E/'reports/environment.json',{})
 t1=get(E/'replay/learned_owod_t1_ft.provenance.json');t2=get(E/'replay/learned_owod_t2_ft.provenance.json')
 lines=['# B1_PROB_OWOBJ_Task2复现报告',f'更新时间：{now()}。状态：'+('全部训练与评价完成' if all([A,B,ea,eb]) else '执行中 / 尚未完成，不作为最终结果报告'),'',
 '## 实验与协议',
 'B1分支 b1/prob-owobj-mowodb-t2，独立源码 /root/autodl-tmp/projects/OWOBJ-B1，输出 /root/autodl-tmp/experiments/B1_PROB_OWOBJ。B0/R0/R1/S2与共享数据内容不修改。详见 B1_Task2_协议审计.md、checkpoint_audit.json、data_audit.json。',
 'B0 Task1已训练epoch0–40。继承checkpoint0040.pth，严格加载81维头，旧VOC20索引0–19不变，新增COCO20索引20–39，未来40–79折叠Unknown80。采用TOWOD20/20与B0和官方M评价脚本一致，修正官方训练脚本OWDETR19/21的类别错配。',
 '## 两阶段训练与回放',
 'Stage A：pretrain从epoch40继续epoch41–50共10轮，重新初始化AdamW与StepLR，lr2e-5，obj_loss_coef8e-4，obj_temp1.3，obj_kl_div0.1；传freeze_prob_model；仅监督新类。',
 'Stage B：pretrain从Stage A epoch50继续epoch51–110共60轮，重新初始化AdamW/StepLR，默认lr2e-4，lr_drop40；不传freeze_prob_model；使用Task2合并exemplar，监督累计40类。不是额外从0训练51+111轮。',
 '两阶段FP32 batch2×accum10，有效batch20，复用B0环境、数据和初始权重。训练seed42，TF32均False；每样本增强使用按epoch/index派生seed，保证中断续训与worker预取无关，此为执行层确定性差异。尾部不足有效batch20丢弃，与B0一致。',
 '官方 freeze_prob_model 只切换affine=False的两组BN为eval，train_one_epoch的model.train随后将其还原；B1保留实际行为，未增加参数冻结方法。BN变化及分类器新/旧/未来行梯度与参数变化记录在各阶段epochs.jsonl及smoke报告。',
 'Task1 exemplar由B0最终权重在Task1训练集补建，无测试/未来GT监督；官方置信度两端选择50实例/类、上限850。因原RNG未保存，使用固定seed与原训练增强重建，selection-only不drop_last。Task2选择新类后合并Task1文件，去重shuffle截断1743，Stage B在该集合回放旧新类；Stage A不直接混入旧exemplar，这是官方实际实现。',
 f'Task1 replay：{json.dumps(t1,ensure_ascii=False) if t1 else "待完成"}',f'Task2 replay：{json.dumps(t2,ensure_ascii=False) if t2 else "待完成"}',
 f'Stage A完成记录：{json.dumps(A,ensure_ascii=False) if A else "未完成"}',f'Stage B完成记录：{json.dumps(B,ensure_ascii=False) if B else "未完成"}',
 '## 官方评价',
 '独立评价进程保持B0复核数值设置：batch5，matmulTF32=False、cuDNNTF32=True、benchmark=False、deterministic=False。原main_open_world.main/engine.evaluate/PostProcess/OWEvaluator；全局sigmoid top100，无额外阈值/NMS；VOC07 AP11，IoU>0.5；+1坐标转换，score3位/box1位。WI=源码recall0.8下比例，AOSA实际为A-OSE次数。',
 '|阶段|Previous mAP50 %|Current mAP50 %|Both mAP50 %|Unknown Recall50 %|Unknown AP50 %|WI 比例|A-OSE 次|原始Query覆盖率 %|',
 '|---|---|---|---|---|---|---|---|---|']
 for name,item in [('Stage A',ea),('Stage B',eb)]:
  if item:
   m=item['metrics'];vals=[m.get(k) for k in ['PK_AP50','CK_AP50','K_AP50','U_R50','U_AP50','WI','AOSA']]+[item['query_diagnostic']['coverage_percent']]
   lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' if v is not None else '未定义' for v in vals)+'|')
  else:lines.append('|'+name+'|'+'|'.join(['待运行']*8)+'|')
 lines += ['|论文M-OWODB Task2|58.4|34.4|45.7|23.8|未报告于主表|未填|未填|无|',
 '原始Query覆盖率定义：唯一图像的非difficult Unknown GT，每个GT独立取原始100个query框连续坐标IoU最大值>=0.5，不看分数/标签；允许query复用。与官方U-Recall的重复行分母、贪心匹配和序列化不同，不可混同。',
 '测试集owod_all_task_test共10246行/9119唯一图像，保留官方重复行。每个阶段权重hash、split hash、代码hash、数值后端及完整单位见evaluation/<stage>/manifest.json，逐类AP与query_unknown_gt.csv可复查。',
 '## 对照与解释限制',
 f'B0官方Task1：{json.dumps(orig,ensure_ascii=False)}。其Known为旧20，Unknown为后60；Task2 Known为40、Unknown为后40。不同任务Known mAP不能解释为同一集合变化；可用B1 PK_AP50对照B0 K_AP50检查旧类，但检测竞争集合也已改变。',
 '论文参考来源：https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Open-World_Objectness_Modeling_Unifies_Novel_Object_Detection_CVPR_2025_paper.html ，B0已有sources/owobj_paper.txt表1。单次确定性种子、无多seed统计，不做显著性主张。差异可能来自B0起点、环境版本、单卡累积与BN语义、补建exemplar RNG及上游脚本协议错误，不能据此单独归因某一因素。',
 f'B0原始Query覆盖率（Unknown60，unique非difficult GT）={100*baseline_coverage:.6f}%。Task2将Unknown缩小到40类，数值不可当作同一GT集合的改进。' if baseline_coverage is not None else 'B0 Query覆盖率基线不可用。',
 '## 资源、恢复与退出',
 f'存储预算：{json.dumps(storage,ensure_ascii=False)}',
 '每阶段仅滚动latest+previous，最终权重为latest硬链接；原子临时写入+fsync，每100次更新及epoch末保存，低于7GiB在更新边界保存并退出75，5GiB为写入最低门槛。不会删除B0/R0/R1文件。resume恢复B1本阶段优化器/scheduler/RNG/批偏移；恢复命令configs/resume.sh。',
 f'环境：{json.dumps(environment,ensure_ascii=False)}',f'后台状态：{json.dumps(status,ensure_ascii=False)}',
 '运行日志logs/，PID及分步骤真实退出码run_state/steps.jsonl，训练epoch/批位置run_state/stage_*_status.json，最终GPU与磁盘run_state/final_resources.txt。',
 '## 产物与下一步',
 '只有Stage B和两阶段官方评价全部成功才可关闭实例。S2/R1后续如使用B1权重必须改为TOWOD Task2累计40类、Unknown80及本次相同预处理；现有Task1类别掩码/GT划分/特征缓存不可直接混用。本任务不修改或运行S2/R1。']
 lines += ['## 与论文Task2的数值差异（百分点）']
 for name,item in [('Stage A',ea),('Stage B',eb)]:
  if item:
   m=item['metrics'];lines.append(f"{name}减论文参考：Previous={m['PK_AP50']-58.4:+.6f}，Current={m['CK_AP50']-34.4:+.6f}，Both={m['K_AP50']-45.7:+.6f}，U-Recall={m['U_R50']-23.8:+.6f}百分点。只描述单次复现与报告值差异，不作显著性/因果结论，不据测试结果调参。")
  else:lines.append(name+'尚未完成，不能计算差异。')
 (E/'reports/B1_PROB_OWOBJ_Task2复现报告.md').write_text('\n'.join(lines)+'\n')
 comp=['# B0_B1增量训练对照',f'更新时间：{now()}，B1状态：'+('完成' if all([A,B,ea,eb]) else '执行中，Task2指标待完成'),'',
 '|项目|B0 Task1|B1 Task2|','|---|---|---|','|Known|VOC20索引0–19|旧20不变，新20–39，共40|','|Unknown|其余60类，统一80|其余40类，统一80|','|模型|81维分类头、100query|维度不变，新增有效logit20–39|',
 '|继承|训练epoch0–40|B0 e40→Stage A41–50→Stage B51–110|','|旧类保留|Task1训练|完整模型继承，Stage B回放旧新exemplar|',
 '|防遗忘机制|无跨任务回放|官方阶段A仅新类；阶段B平衡回放；没有新方法|',
 f'B0 Known mAP50={orig["K_AP50"]:.6f}%，U-Recall50={orig["U_R50"]:.6f}%，U-AP50={orig["U_AP50"]:.6f}%，WI={orig["WI"]:.8f}，A-OSE={orig["AOSA"]:.0f}。']
 for n,item in [('B1 Stage A',ea),('B1 Stage B',eb)]:
  comp.append(n+'：'+(json.dumps(item['metrics'],ensure_ascii=False) if item else '待运行，无结果。'))
  if item:
   m=item['metrics'];comp.append(f"{n}旧20类AP相对B0变化{m['PK_AP50']-orig['K_AP50']:+.6f}百分点；U-Recall数值差{m['U_R50']-orig['U_R50']:+.6f}百分点、U-AP差{m['U_AP50']-orig['U_AP50']:+.6f}百分点，但Unknown类集合不同，不能据此声称同一GT上提升/退化。")
 if baseline_coverage is not None:comp.append(f'B0原始query对Unknown60 GT覆盖率IoU>=0.5为{100*baseline_coverage:.6f}%；B1仅对未来40类统计，不能将不同GT集合直接比较。')
 comp += ['B0 Known20与B1 Both Known40不是同一集合，不将两者直接解读为性能升降；B1 Previous20与B0 Known20类名一致，可用于旧类保留诊断。Unknown定义从60类变40类，Recall/AP变化也受目标集合变化影响。',
 'S2/R1后续可将B1 Stage B checkpoint作为Task2起点，但必须同时采用40类掩码与GT映射并重提特征，不能替换原Task1权重后沿用旧缓存。本次没有修改S2/R1。']
 (E/'reports/B0_B1增量训练对照.md').write_text('\n'.join(comp)+'\n')
 if final: assert all([A,B,ea,eb]),'Final report requires both training and evaluation stages'
if __name__=='__main__':generate()
