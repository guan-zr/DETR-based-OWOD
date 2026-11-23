"""Generate the requested report only from complete, verified real experiment artifacts."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from r0.runtime import file_sha256,write_json
ROOT=Path('/root/autodl-tmp/experiments/R0');OUT=ROOT/'reports';EVIDENCE=ROOT/'formal_20261111'

def table(frame,columns=None):
    if columns is not None:frame=frame[columns]
    def text(v):
        if pd.isna(v):return '未定义'
        if isinstance(v,float):return f'{v:.6g}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(text(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def main():
    analysis=ROOT/'analysis';done=json.loads((analysis/'complete.json').read_text())
    if not done.get('complete'):raise RuntimeError('Analysis incomplete')
    for name in ['analysis-report.md','stats-appendix.md','figure-catalog.md','analysis_facts.json']:
        if file_sha256(analysis/name)!=done['artifacts'][name]:raise RuntimeError('Analysis source mismatch')
    ev=json.loads((ROOT/'paired_evaluation/complete.json').read_text())
    if not ev.get('complete') or ev['unique_images']!=9119 or ev['scope']!='full_9119_unique':raise RuntimeError('Cannot report partial evaluation as full')
    if file_sha256(ROOT/'paired_evaluation/complete.json')!=done['evaluation_complete_sha256']:raise RuntimeError('Evaluation changed')
    preflight=json.loads((EVIDENCE/'preflight.json').read_text());gpu=json.loads((EVIDENCE/'gpu_validation.json').read_text());probe=json.loads((EVIDENCE/'resource_probe.json').read_text())
    metrics=pd.read_csv(analysis/'metrics.csv');unknown=pd.read_csv(analysis/'unknown_coverage.csv');common=pd.read_csv(analysis/'common_valid_metrics.csv');dominance=pd.read_csv(analysis/'scale_dominance.csv')
    distributions=pd.read_csv(analysis/'counts_distributions.csv');geometry=pd.read_csv(analysis/'geometry_strata.csv');classes=pd.read_csv(analysis/'unknown_class_size.csv')
    audit=json.loads((analysis/'grouping_audit.json').read_text());context=json.loads((analysis/'coverage_context.json').read_text());ci=json.loads((analysis/'cluster_bootstrap.json').read_text())
    # Verify immutable inputs immediately before a success report is issued.
    protected=json.loads((EVIDENCE/'protected_before.json').read_text());changed={p:dict(before=h,after=file_sha256(p)) for p,h in protected.items() if file_sha256(p)!=h}
    write_json(EVIDENCE/'protected_after.json',dict(passed=not changed,checked_files=len(protected),changed=changed))
    if changed:raise RuntimeError('Protected B0/R1 files changed; refuse final acceptance')
    checkpoints={};training={}
    for source in ['S1','S2']:
        state=json.loads((ROOT/'pipeline/stages'/f'{source}_train.json').read_text());directory=Path(state['artifact_directory']);end=json.loads((directory/'training_complete.json').read_text())
        if not end['complete'] or end['steps']!=2000:raise RuntimeError('Incomplete training')
        logs=[json.loads(l) for l in (directory/'train.jsonl').read_text().splitlines()];canonical={r['step']:r for r in logs}
        actual=[canonical[k]['image_ids'][0] for k in range(1,2001)];expected=(EVIDENCE/'train_image_sequence.txt').read_text().splitlines()
        if actual!=expected:raise RuntimeError('Training sequence drift')
        checkpoints[source]=dict(path=str(directory/'checkpoint.pth'),sha256=file_sha256(directory/'checkpoint.pth'))
        training[source]=dict(steps=2000,seconds_last_invocation=end['seconds'],sequence_sha256=preflight['train_sequence_sha256'],
            logged_rows=len(logs),canonical_steps=len(canonical),first_loss=canonical[1]['loss'],last_loss=canonical[2000]['loss'])
    existing_report=OUT/'R0_S1_S2_正式验证报告.md'
    date=next((l[6:] for l in existing_report.read_text().splitlines() if l.startswith('date: ')),None) if existing_report.exists() else None
    date=date or datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    main_metrics=metrics.loc[(metrics.rule=='disjoint_gt')&(metrics.level=='overall')]
    report=f'''---
type: results-report
date: {date}
experiment_line: R0
round: 0
purpose: spatial-reconstruction-objectness-feasibility
status: complete
source_artifacts:
  - ../analysis/analysis-report.md
  - ../analysis/stats-appendix.md
linked_experiments: []
linked_results: []
---
# R0 S1/S2 正式验证报告

本次固定 seed=73 的 S1、S2 各2000步训练及9,119唯一图评价均已真实完成。本报告为给定候选区域上的离线诊断；未验证自动未知候选生成或端到端发现增益。报告生成时间：{date}（Asia/Shanghai）。

## 实验身份、有效性和协议

研究问题：冻结 PROB+OWOBJ 后，空间重建误差能否作为Known、Unknown共同有效的类无关前景信号，尤其在最终Query未覆盖的Unknown GT区域。

B0 权重：`{ev['baseline']['path']}`；SHA256 `{ev['baseline']['sha256']}`；完整epoch0–40，训练41轮。独立恢复验收通过11项官方指标及完整2,050批/10,246行逐图一致性；R0门禁核验权重、训练逐轮记录、恢复指标、trace manifest/index及split哈希，没有补写旧pipeline成功标记。原官方Known AP50=56.74260330200195%、Unknown AP50=0.5515708923339844%、Unknown R50=18.426243567753%。

设备/软件及明确数值后端：

```json
{json.dumps(gpu['environment'],indent=2,ensure_ascii=False)}
```

训练 split `owod_t1_train`：{preflight['train_rows']}行，{preflight['train_unique']}唯一图；评价 `owod_all_task_test`：10,246原始行，9,119唯一图；集合交集=0。训练数据接口只读图像，不解析XML；Unknown GT只在训练结束后的离线分析使用。完整45,131条per_gt已逐项核验XML索引、Known/Unknown身份及坐标，R0与B0框坐标精确一致：原始连续XYXY，xmin/ymin−1，xmax/ymax保留。

六张结构性真实图像通过原始预测和官方后处理Hook前后逐位一致、S1/S2特征及固定掩码误差重复逐位一致。实际接口：S1为Transformer输入4级256通道，S2为encoder memory按真实空间形状拆分；名义stride8/16/32/64。逐图shape、mask和有效位置数量见前置GPU工件与评价每块images.jsonl。额外成对提取与原S1接口逐位一致；快速ROI向量传输与原评分记录逐条相等。真实batch2的padding排除常量图测试通过；跨batch上下文不要求预测逐位相等。

图像边界、小框、极端长宽比、非有限/退化ROI检查通过。ROIAlign=7×7、sampling_ratio2、aligned=True；按原图→实际resize→名义stride映射；分子/支持联合采样排除padding，最小支持比例0.5，原有极小框规则保留。

正式配置只有source/name/output三项不同，AE/FMR/valid MSE、mask0.6、4次评价重复、AdamW lr1e−4/weight_decay1e−4、梯度裁剪1.0、seed73、batch1、2000步预算及固定采样协议未改变。未启用R1/S3/AREA/final_detection。每尺度独立AE，4尺度同容量；原始MSE取等权均值为预先指定overall。没有按测试GT选择步数、方向或融合权重。

训练完成与图像序列：

```json
{json.dumps(training,indent=2,ensure_ascii=False)}
```

实际顺序哈希两者相同，完整顺序文件在formal_20261111/train_image_sequence.txt；原始训练日志保留重试时的未提交步骤，canonical日志按固定step取最后一次，未把重试当新增训练预算。无全量高维缓存；真实探针为{probe['rows']}行/{probe['compressed_bytes']}字节，峰值分配{probe['environment']['peak_allocated_mib']:.1f}MiB；正式评价峰值{ev['peak_allocated_mib']:.1f}MiB，全部{ev['rows']:,}条评分记录逐块保存。每32图保存可验证的complete manifest，部分块不标记完整。

## 两种候选池的可分性

GT-reference使用注释身份GT＋固定随机背景提案；All-query使用每图全部100个最终Decoder原始框，没有分数/类别预筛。两池分开计算。主规则disjoint_gt的背景与全部可用GT（包括difficult）无正面积交集；IoU-only为旧低IoU对照，不能称可靠背景。两个规则使用同一AE权重和相同评分，不重复训练。GT-reference中转为模糊的随机提案只进审计，不进入主参考池；注释框difficult为Ambiguous。

主规则overall指标（AUROC/AP均为0–1；AP为按分数并列分组的Average Precision，并报告正例比例）：

{table(main_metrics,['method','pool','comparison','positive_count','negative_count','auroc','auprc_average_precision','positive_prevalence'])}

全部每尺度、两规则与合并前景精确表：

{table(metrics,['method','pool','rule','level','comparison','positive_count','negative_count','auroc','auprc_average_precision'])}

各组数量、有效/无效及ROI支持（distribution_record为空的行为计数，True行为对应分布统计；详细分位数见CSV）：

{table(distributions.loc[distributions.distribution_record.isna()],['method','pool','rule','level','group','valid','count','small_feature_count','valid_fraction_mean'])}

## 真实候选分组迁移

```json
{json.dumps(audit,indent=2,ensure_ascii=False)}
```

真实Query Background→Ambiguous={audit['query']['transitions'].get('background->ambiguous',0):,}。随机提案 Background→Ambiguous={audit['gt_reference_proposals']['transitions'].get('background->ambiguous',0):,}；历史235,459只属于固定随机提案，不能替代Query迁移。

## B0漏检Unknown区域

按B0首次出现上下文的原始Query IoU≥0.5分覆盖/未覆盖；额外单列官方未匹配且无候选的B类。全部未覆盖数与11,434个B类不是同一集合：部分官方检出依赖后续重复图像上下文或VOC序列化口径。

```json
{json.dumps(context,indent=2,ensure_ascii=False)}
```

给定GT区域的重建误差信号：

{table(unknown,['method','rule','category','positive_count','negative_count','mean_score','median_score','auroc','auprc_average_precision','small_feature_fraction','valid_fraction_mean'])}

细分Unknown类别与大小、形状结果在`../analysis/unknown_class_size.csv`，各分层均含精确正负例数量；同尺寸/形状层比较使用同层背景，类别分析使用对应规则全部背景。B类按类别主规则诊断：

{table(classes.loc[(classes.rule=='disjoint_gt')&(classes.category=='b0_missed_B_no_candidate')&(classes.stratum=='class_name')],['method','value','positive_count','negative_count','auroc','auprc_average_precision'])}

所有评分GT框由评价提供，因此任何正向信号只能说明给定区域的空间特征可能保留前景信息，不能宣称找回11,434个目标。

## 尺度、几何混杂及共同有效集合

S1/S2误差幅值不可直接作为物体性强弱。每尺度AUROC/AP见完整表；均值贡献分解如下，数值权重较高的尺度可能支配raw mean，但未据此重新加权：

{table(dominance)}

共同有效候选的S1/S2对比：

{table(common)}

目标面积（原图像素：小于32²、32²至96²、≥96²）、相对面积（<1%、1–10%、≥10%）、长宽比（<0.5、0.5–2、>2）、有效支持比例及特征极小框分层见geometry_strata.csv。GT和随机背景的尺寸/形状分布见geometry_distributions.csv。主规则Unknown面积/形状诊断：

{table(geometry.loc[(geometry.rule=='disjoint_gt')&(geometry.comparison=='unknown_vs_background')],['method','pool','stratum','value','positive_count','negative_count','auroc','auprc_average_precision'])}

这些分层用于检查几何解释，没有按测试数据匹配后重训、调整采样或寻找最佳评分规则。即使分层内仍有差异，也不能排除未测量几何/纹理混杂。

## 统计验证与研究假设判断

只有一个训练seed。以下200次图像聚类成对bootstrap的95%区间只描述评价图像抽样不确定性，不支持跨训练seed稳定性或参数选择。没有p值或多重显著性宣称。候选来自同一图像并非独立样本；主要效应为AUROC/AP与S2−S1成对差值。

```json
{json.dumps(ci,indent=2,ensure_ascii=False)}
```
'''
    # Evidence-driven statements, always bounded to this run and primary positive direction.
    report+='\nA. 空间误差是否存在可重复的Unknown/Background可分性：同输入重复提取和评分已验证；跨训练seed重复性尚未验证。各候选池主方向结果如下。\n\n'
    for pool in ['gt_reference','query']:
        report+=f'- {pool}：'
        for source in ['S1','S2']:
            row=main_metrics.loc[(main_metrics.pool==pool)&(main_metrics.method==source)&(main_metrics.comparison=='unknown_vs_background')].iloc[0]
            interval=ci[pool]['estimates'][source]['auroc']['ci95'];report+=f"{source} AUROC={row.auroc:.6f}（共同有效集图像区间{interval[0]:.6f}–{interval[1]:.6f}）；"
        report+='仅限本次固定条件，不能由单seed确认训练层面的可重复性。\n'
    report+='\nB. S1/S2比较：以共同有效候选表和成对差值区间为准；不能按原始MSE幅值排名。'
    for pool in ['gt_reference','query']:
        delta=ci[pool]['estimates']['S2_minus_S1']['auroc'];report+=f" {pool}的S2−S1 AUROC差均值{delta['mean']:.6f}，95%区间{delta['ci95'][0]:.6f}–{delta['ci95'][1]:.6f}。"
    report+='\n\nC. B0最终Query漏检区域：表中b0_raw_uncovered和b0_missed_B_no_candidate给出直接事实，属于已知GT区域诊断；若高误差方向AUROC高于0.5，只支持存在区域排序信号的有限推测。\n\nD. 几何、背景采样、尺度和集合变化都可能解释部分差异。报告提供分层、尺度贡献与共同有效集，但未证明语义因果；可靠背景仅相对现有标注，不能排除未标注物体。\n\nE. 当前证据只足以决定是否值得设计后续无GT候选实验，不足以证明实际发现增益、创新性或跨seed稳定。应先评估本次负面/正面结果是否跨尺度和几何层一致；后续需要预先登记的多seed重复、无需Unknown GT的候选生成、完整推理与官方检测评估。本流水线不自动启动这些后续工作。\n'
    report+='''
## 图表解释、负面结果与边界

![Main separability](../analysis/figures/main_separability.png)

此图比较每尺度与原始均值的排序可分性；基线0.5。若只有某尺度有效，均值不能自动视为统一空间Objectness。两个池不能合并总体，也不能比较不同尺度MSE幅值。

![Unknown coverage](../analysis/figures/unknown_coverage.png)

此图直接检查给定的B0未覆盖及B类GT区域；正向结果支持空间信息的有限线索，低于或接近0.5同样作为负面事实保留。未绘制反方向指标来替换原方法。

![Training loss](../analysis/figures/training_loss.png)

固定预算训练曲线用于核对训练动态。loss下降不是Unknown可分性证据；网络未使用测试评分选择checkpoint。

主要局限：单训练seed、mask ensemble有限重复、有限且非穷尽XML标注、随机背景几何分布、S1/S2表示范围不同、batch1与官方batch5 padding上下文不同、GT-reference非自动候选。可重复的同输入数值不能替代跨训练seed重复。未运行final_detection/P1-01、R1、联合训练或新检测分支。

## 可复现性与验收工件

- 流水线命令与每阶段日志/状态：`../pipeline/commands.json`、`../pipeline/logs/`、`../pipeline/stages/`；独立会话PID在`../pipeline/pipeline.pid`，失败退出状态保留。
- 原始源代码与修复哈希：`../formal_20261111/original_r0/`、`repair_version_check.json`；本次实际源码快照和哈希：`../pipeline/source_snapshot/`、`source_manifest.json`。
- B0验收及GPU探针：`../formal_20261111/preflight.json`、`gpu_validation.json`、`resource_probe.json`，分别含来源哈希、真实shape、后端与资源证据。29项CPU测试日志在cpu_final.log；原21项smoke及合成工件单独标注，不充作真实研究指标。
- 评价块：`../paired_evaluation/complete.json`及chunks/；每块有评分CSV、逐图记录、两规则迁移与SHA256。没有缓存完整空间特征。
- 严格分析：`../analysis/analysis-report.md`、`stats-appendix.md`、`figure-catalog.md`和所有数值CSV；不需重新运行检测器即可复核主表。
- 受保护文件复核：`../formal_20261111/protected_after.json`，B0与R1既有工件未改变。没有访问Windows本地文件，无Obsidian写回。

AE权重及SHA256：

```json
'''+json.dumps(checkpoints,indent=2,ensure_ascii=False)+'\n```\n\n后续操作：保留本次固定方案正负结果与完整工件；对照几何和共同集合解释信号，先补跨seed及无GT候选证据，再决定是否推进重建式空间Objectness。\n'
    OUT.mkdir(exist_ok=True);path=OUT/'R0_S1_S2_正式验证报告.md'
    if path.exists():
        if path.read_text()!=report:raise RuntimeError('Refuse to overwrite existing formal report')
    else:path.write_text(report)
    write_json(OUT/'report_complete.json',dict(complete=True,report=str(path),sha256=file_sha256(path),analysis_complete_sha256=file_sha256(analysis/'complete.json'),generated_at=date))
    print('Report complete:',path,flush=True)
if __name__=='__main__':main()
