"""Evidence-only second-round report. No model/score/direction selection or shutdown."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from r0.round2.common import *

def table(f,cols=None):
    if cols is not None:f=f[cols]
    def value(x):
        if pd.isna(x):return '未定义'
        return f'{x:.6g}' if isinstance(x,float) else str(x).replace('|','/')
    return '| '+' | '.join(map(str,f.columns))+' |\n|'+'|'.join(['---']*len(f.columns))+'|\n'+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))+'\n'

def main():
    out=ROOT/'analysis';manifest_check(out/'complete.json');ev=json.loads((ROOT/'evaluation/complete.json').read_text())
    if not ev['complete'] or ev['scope']!='full_9119_unique' or ev['unique_images']!=9119:raise RuntimeError('Cannot report partial controls')
    metrics=pd.read_csv(out/'all_controls_metrics.csv');paired=pd.read_csv(out/'paired_differences.csv');stability=pd.read_csv(out/'seed_stability.csv');correlation=pd.read_csv(out/'query_correlations.csv');ranking=pd.read_csv(out/'query_rank_changes.csv');retention=pd.read_csv(ROOT/'prepared/matching_retention.csv')
    meta=pd.read_csv(ROOT/'prepared/metadata.csv.gz',dtype={'image_id':str});matches=np.load(ROOT/'prepared/geometry_matches.npz');geom=[];unmatched=[]
    area=(meta.x2-meta.x1)*(meta.y2-meta.y1);relative=area/(meta.original_h*meta.original_w);aspect=(meta.x2-meta.x1)/(meta.y2-meta.y1);vectors=np.log(np.stack([area,relative,aspect],1))
    for row in retention.to_dict('records'):
        key=row['pool']+'__'+row['comparison'];pairs=matches[key+'__pairs'];missing=matches[key+'__unmatched'];bg=np.flatnonzero((meta.source==row['pool'])&(meta['group']=='background'));pos=np.r_[pairs[:,0],missing]
        for i,name in enumerate(['log_pixel_area','log_relative_area','log_aspect']):
            a,b=vectors[pos,i],vectors[bg,i];aa,bb=vectors[pairs[:,0],i],vectors[pairs[:,1],i]
            def smd(x,y):
                den=np.sqrt((np.var(x)+np.var(y))/2);return float((np.mean(x)-np.mean(y))/den) if len(x) and len(y) and den>0 else None
            if len(pairs) and np.max(np.abs(aa-bb))>np.log(1.25)+1e-10:raise RuntimeError('Matching caliper violated')
            geom.append(dict(pool=row['pool'],comparison=row['comparison'],quantity=name,original_standardized_mean_difference=smd(a,b),matched_standardized_mean_difference=smd(aa,bb),original_KS=float(ks_2samp(a,b).statistic) if len(a) and len(b) else None,matched_KS=float(ks_2samp(aa,bb).statistic) if len(aa) and len(bb) else None,max_matched_log_difference=float(np.max(np.abs(aa-bb))) if len(pairs) else None))
        f=meta.iloc[missing].copy();f['global_candidate']=missing;f['comparison']=row['comparison'];f['reason']='No unused existing background inside all three fixed geometry calipers';unmatched.append(f)
    pd.DataFrame(geom).to_csv(out/'matching_geometry_audit.csv',index=False);pd.concat(unmatched).to_csv(out/'unmatched_objects.csv.gz',index=False,compression='gzip')
    loss=pd.read_csv(ROOT/'loss_probe/same_training_images_loss.csv');loss_summary=loss.groupby(['source','method','level']).mse.agg(['mean','std','count']).reset_index();loss_summary.to_csv(out/'training_side_loss_summary.csv',index=False)
    # Preserve all old output files and all old B0/R1 artifacts.
    changed=[]
    for p,h in json.loads((ROOT/'protected_before.json').read_text()).items():
        if file_sha256(p)!=h:changed.append(p)
    write_json(ROOT/'protected_after.json',dict(passed=not changed,checked_files=len(json.loads((ROOT/'protected_before.json').read_text())),changed=changed))
    if changed:raise RuntimeError('First-round/B0/R1 protected artifact changed')
    figures=out/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for si,source in enumerate(SOURCES):
        for pi,pool in enumerate(['gt_reference','query']):
            f=metrics.loc[(metrics.source==source)&(metrics.pool==pool)&(metrics.comparison=='unknown')&(metrics['mode']=='original')&(metrics.level=='overall')]
            methods=['trained73','trained74','trained75','random73','random74','random75','l2','energy','channel_variance','mean_error'];f=f.set_index('method').reindex(methods);ax=axes[si,pi]
            y=f.auroc.to_numpy();lower=np.maximum(0,y-f.auroc_ci_low.to_numpy());upper=np.maximum(0,f.auroc_ci_high.to_numpy()-y)
            ax.errorbar(range(len(methods)),y,yerr=np.stack([lower,upper]),fmt='o',capsize=2);ax.axhline(.5,color='gray',ls='--');ax.set(title=source+' / '+pool,ylim=(0,1),xticks=range(len(methods)),xticklabels=methods,ylabel='Unknown/background AUROC');ax.tick_params(axis='x',rotation=65)
    fig.savefig(figures/'all_controls.png',dpi=180,bbox_inches='tight');fig.savefig(figures/'all_controls.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,source in zip(axes,SOURCES):
        f=metrics.loc[(metrics.source==source)&(metrics.pool=='gt_reference')&(metrics.comparison=='b0_missed_B_no_candidate')&(metrics.level=='overall')&metrics.method.isin(['trained73','energy','mean_error','random73'])]
        for method in ['trained73','energy','mean_error','random73']:
            part=f.loc[f.method==method].set_index('mode').reindex(['original','matched_positives_original_background','geometry_matched']);ax.plot(range(3),part.auroc,marker='o',label=method)
        ax.set(title=source+' / B0 missed B GT',xticks=range(3),xticklabels=['original','retained positives','matched pairs'],ylim=(0,1),ylabel='AUROC');ax.axhline(.5,color='gray',ls='--');ax.legend(fontsize=8)
    fig.savefig(figures/'missed_geometry.png',dpi=180,bbox_inches='tight');fig.savefig(figures/'missed_geometry.svg',bbox_inches='tight');plt.close(fig)
    primary=metrics.loc[(metrics.level=='overall')&(metrics['mode']=='original')];matched_metrics=metrics.loc[(metrics.level=='overall')&(metrics['mode']=='geometry_matched')]
    effects=paired.loc[(paired.level=='overall')&(paired['mode']=='original')]
    matched_effects=paired.loc[(paired.level=='overall')&(paired['mode']=='geometry_matched')]
    columns=['source','pool','comparison','method','positive_count','negative_count','auroc','auroc_ci_low','auroc_ci_high','auprc_average_precision','auprc_average_precision_ci_low','auprc_average_precision_ci_high']
    effect_columns=['source','pool','comparison','method_left','method_right','auroc_difference','auroc_difference_ci_low','auroc_difference_ci_high','auprc_average_precision_difference','auprc_average_precision_difference_ci_low','auprc_average_precision_difference_ci_high']
    # The complete strict analysis bundle precedes the internal report.
    (out/'stats-appendix.md').write_text('''# Round2 statistical appendix\n\n所有方向、分数和几何规则在protocol.json事先固定。每方法/尺度/候选池/分组报告精确AUROC、并列分组AP、正负例数和200次图像聚类bootstrap百分位95%区间；每次对9119唯一图有放回抽样，同一抽样权重在所有方法共享，方法差值由同一重复相减计算。匹配对允许跨图但无候选复用，每个对象和背景按自身来源图赋权，不把一对或每个候选视为独立抽样单位。某子集缺任一类时指标及区间未定义。区间描述评价图像抽样，不替代3个训练seed的均值/样本标准差。三个seed过少，不作训练总体显著性或正态性断言。多对照和分层作为预先规定的描述性诊断，无多重显著性获胜/最优分数选择。主要效应量为同候选AUROC/AP差值；AP依赖正例率。\n''')
    (out/'figure-catalog.md').write_text('''# Figure catalog\n\n- figures/all_controls.png/.svg：所有预定对照在同候选池的Unknown AUROC，误差条为图像聚类95%区间；核对训练、随机、能量等谁提供额外值，不能只看训练模型绝对数值。\n- figures/missed_geometry.png/.svg：B类GT原始、保留正例/原背景、匹配对背景的比较，区分正例保留影响与背景变化；匹配后结果只适用于保留的共同支持对象。\n\n图必须与精确CSV、保留率和单seed值共同解释。正向图不能推出自动Unknown Discovery增益。\n''')
    (out/'analysis-report.md').write_text('''# R0 round2 evidence analysis\n\n比较问题：空间重建学习相对简单统计、未训练AE和B0已有Query分数是否增加信息；单一固定候选集合、固定方向和原ROI口径。完整真实9119图才生成本文件。\n\nClaim candidates：\n- 当前B0特征的区域判别性：依据metrics.csv；仅能陈述本次条件和区间。\n- 重建学习额外价值：依据paired_differences.csv、seed_stability.csv及匹配背景表；必须对照能量/均值/随机AE，不以绝对AUROC或loss下降替代。\n- 对B0 Objectness的补充排序：依据相同前向Query、query_correlations.csv、条件分层和rank_changes；不声称已学习或验证融合增益。\n- 无GT Unknown Discovery：本轮无自动候选生成和官方检测增益验证，不允许据前三类证据直接宣称成立。\n\n所有详细事实和样本量在CSV中。单seed获胜不得升格为跨seed确定结论；匹配未覆盖对象保留在unmatched_objects.csv.gz。\n''')
    source=source_hashes();write_json(ROOT/'final_source_sha256.json',source)
    generated=stamp();report=f'''# R0 第二轮有效性验证报告

生成于{generated}（Asia/Shanghai）。全部S1/S2三训练seed、三随机初始化和简单统计对照已在9,119唯一图完成；seed73复用原权重，未重训。所有原实验/B0/R1保护工件SHA复核通过。未自动关闭AutoDL实例。

## 实验定义与来源

原B0 checkpoint SHA：`{json.loads((ROOT/'preflight.json').read_text())['baseline_sha256']}`。原Known/Unknown类表、原图连续XYXY坐标、resize、batch1和有效支持归一化ROIAlign保持不变。主背景disjoint_gt；GT-reference与All-query分开统计。每尺度及固定原始四尺度均值全部保留，不按测试选择尺度或方向。

L2：先算每位置sqrt(sum_c f²)，再ROI；平方能量：每位置mean_c f²；通道方差：每位置mean_c(f−mean_c f)²；零预测MSE严格等于平方能量，作为等价别名而非独立证据；均值重建使用相同2000张训练图的有效位置加权通道均值μ，评分mean_c(f−μ)²。所有评分固定高值为前景，未展示取反后挑优。

AE结构/FMR/valid MSE/mask0.6/AdamW/梯度裁剪/2000步预算与第一轮一致。新增seed74、75仅改变初始化及训练掩码；训练顺序仍为第一轮seed73的同2000个图像ID，评价所有AE共用第一轮seed73的每图每尺度4次掩码。随机AE三初始化73/74/75与对应训练初始化一致；seed73原首步loss重现验证通过。训练不读XML。训练均值也不使用标签。

Query Gaussian=exp(−1.3/256×pred_obj)，classifier unknown=sigmoid最后unknown logit。两者与所有空间评分来自同一次batch1前向，严格核对每图Query0..99和原round1裁剪框；记录原始未裁剪框。GT区域不构造Query Gaussian代理；B0未覆盖GT的诊断只在GT-reference定义，不为不存在的合格Query虚构候选。全部图重算seed73验证与原评分的最大绝对差={ev['seed73_max_abs_drift']}，原评分与权重均保持不动。

测速与实际计算记录：gpu_probe.json、training/complete.json、evaluation/complete.json、pipeline/stages及日志。原始训练loss日志、训练图像序列、mean向量、初始化/最终权重、源码快照和哈希均保留。

## A. 简单统计是否已解释重建判别性

必须比较trained73与l2、energy/zero、channel_variance、mean_error的同候选差值和区间；下面是所有原始背景overall指标，包含Known、Unknown、合并前景及B0未覆盖/B类GT；所有每尺度值见analysis/all_controls_metrics.csv。

{table(primary,columns)}

固定原始候选上的训练模型差值：

{table(effects.loc[effects.method_right.isin(['l2','energy','channel_variance','mean_error'])],effect_columns)}

'''
    # Explicit numerical verdicts for each predeclared contrast, including negative results.
    report+='对Unknown及B类的trained73减简单对照AUROC事实：\n\n'
    for r in effects.loc[effects.comparison.isin(['unknown','b0_missed_B_no_candidate'])&effects.method_right.isin(['l2','energy','channel_variance','mean_error'])].to_dict('records'):
        lo,hi=r['auroc_difference_ci_low'],r['auroc_difference_ci_high'];direction='本次图像区间支持正差' if pd.notna(lo) and lo>0 else ('本次图像区间支持负差' if pd.notna(hi) and hi<0 else '区间跨0或未定义')
        report+=f"- {r['source']}/{r['pool']}/{r['comparison']}：trained73−{r['method_right']}={r['auroc_difference']:.6f}，95%区间[{lo:.6f},{hi:.6f}]；{direction}。\n" if pd.notna(lo) else f"- {r['source']}/{r['pool']}：样本不足，未定义。\n"
    report+=f'''
## B. 训练AE相对随机AE和均值重建

全部随机初始化均报告；不以单个random73替代随机机制稳定性。seed73/74/75的训练后−对应随机初始化差值及mean_error对照见paired_differences.csv。以下原背景overall随机对照：

{table(effects.loc[effects.method_right.str.startswith('random')],effect_columns)}

相同64张训练图、相同评价掩码的before/after loss，仅诊断重建拟合，不作为Objectness收益证据：

{table(loss_summary.loc[loss_summary.level=='overall'])}

## C. 面积、相对面积与长宽比匹配

匹配只使用现有disjoint背景候选及几何，不使用评分。三维ln(area)、ln(relative area)、ln(aspect)每一维差≤ln1.25，固定SHA顺序最近邻、不放回；可跨图。每个comparison独立匹配。未匹配对象不会进入匹配指标，但全部记录且报告保留率；这限制了结论的适用范围，不将其隐去。保留正例配全部原背景的额外表用于分离正例保留偏差与背景分布变化。

{table(retention)}

{table(pd.DataFrame(geom))}

匹配候选overall全部对照：

{table(matched_metrics,columns)}

匹配后trained73减简单统计/随机AE：

{table(matched_effects.loc[matched_effects.method_right.isin(['energy','mean_error','random73','l2','channel_variance'])],effect_columns)}

较小目标、B0未覆盖及B类GT的固定匹配大小分层见all_controls_metrics.csv中geometry_matched_small/medium/large；全部不能匹配对象ID、框、类别及失败原因见unmatched_objects.csv.gz。不得把匹配保留子集的结论推广到全部未匹配目标。

## D. 三seed稳定性

各seed单次值在上述指标表；下表给出全部固定原背景overall的均值与样本标准差（ddof1）。三seed不是大量随机实验；图像bootstrap不替代训练seed不确定性。

{table(stability.loc[(stability.level=='overall')&(stability['mode']=='original')])}

## E. B0已有Objectness之外的排序信息

仅使用相同前向、相同QueryID和框；All-query原背景B0与AE指标已列在主表，成对差值也见paired_differences.csv。以下是预先定义的Gaussian底部25%与10个无标签分数分位层的Unknown/Background比较，阈值只由全部Query分数生成；没有训练融合或校准。

{table(metrics.loc[(metrics.level=='overall')&(metrics['mode']=='gaussian_bottom25')],columns)}

所有十个分位层值及区间在all_controls_metrics.csv。若层内AUROC/AP仍有信息，结合不完全单调相关可支持“补充排序信息”的有限判断；不能直接推出已验证融合收益。

{table(correlation.loc[correlation.method.str.startswith('trained')])}

{table(ranking.loc[ranking.method.str.startswith('trained')])}

Spearman/Kendall若接近1表示排序近似单调；不同排序本身不保证更有效。必须与简单统计、随机AE和层内标签区分对照共同解释。上述相关性是描述性事实，没有以其选分数或学融合权重。

## F. 是否继续无Unknown GT空间候选研究

本轮分别提供四层证据：当前B0特征的区域判别能力；重建学习相对简单统计的额外收益；相对B0分数的补充排序；实际Unknown Discovery提升。前三者由对应表分别判断，不能自动推出第四个。若训练−能量/均值/随机AE差值为负或跨0，应保留该负面/不确定事实，不增加复杂模块掩盖，也不以训练loss下降替代结论。

目前仍缺无需Unknown GT的候选生成及完整官方推理/检测增益验证；因此本轮最多支持是否值得进行下一项预先登记实验，不证明实际发现提升。任何进一步研究应先以本报告所有控制、seed和匹配保留率为依据，不能挑最好一项重新包装当前方法。

## 图表及统计边界

![All fixed controls](analysis/figures/all_controls.png)

所有预定训练/随机seed及简单统计的Unknown AUROC，误差条为图像聚类95%区间。重点比较训练相对简单/未训练控制，不能只看绝对可分性。

![Missed B geometry](analysis/figures/missed_geometry.png)

此图分离全部B类、保留B类及匹配背景，展示共同支持选择影响；有利或不利变化都保留。

{(out/'stats-appendix.md').read_text()}

GT只用于离线分组和几何诊断。匹配仅相对现有标注可靠背景，仍可能有未标注物体。单seed方向不被取反，所有seed和尺度均报告，无测试集参数选择、无额外模块或R1操作。

## 完整工件与退出状态

- protocol.json与analysis_plan.json：先验定义、方向、固定匹配、掩码与统计计划。
- prepared/metadata.csv.gz、geometry_matches.npz、matching_retention.csv：同候选索引、匹配对和未匹配ID。
- training/与initializations/：四个新最终AE、训练bundle/optimizer恢复状态、三个原始随机初始化、训练侧均值、原始loss日志。
- evaluation/：每块same-forward评分NPZ、raw Query框、有效性、原模型身份与SHA；完整9,119唯一图commit。
- analysis/all_controls_metrics.csv：所有方法/seed/尺度/池/匹配/条件分层的数值、样本数与95%区间。
- analysis/paired_differences.csv、seed_stability.csv：成对差值区间及所有seed，不挑最佳。
- analysis/query_correlations.csv、query_rank_changes.csv：同Query相关性与排序变化；GT-reference不强行使用Query分数。
- analysis/analysis-report.md、stats-appendix.md、figure-catalog.md：严格分析工件和图解释。
- source_snapshot/、source_manifest.json、final_source_sha256.json：实际源码；protected_before/after.json：第一轮/B0/R1工件不变。
- pipeline/status.json、stages/、logs/、commands.json：每阶段原始日志、退出码、可核实完成状态；不包含关机动作。

下一步：先审查本轮所有预定对照和负面结果；只有当重建学习的额外价值在seed及几何共同支持下仍有依据时，才值得预先登记无GT候选发现实验。当前流水线不会自行添加模块、学习测试融合或关闭实例。
'''
    p=ROOT/'R0_第二轮有效性验证报告.md'
    if p.exists():raise RuntimeError('Refuse to overwrite existing report')
    p.write_text(report)
    write_json(ROOT/'report_complete.json',dict(complete=True,report=str(p),sha256=file_sha256(p),generated_at=generated,sha256_weights={s:{str(k):file_sha256(train_path(s,k)) for k in SEEDS} for s in SOURCES},sha256_initializations={str(k):file_sha256(init_path(k)) for k in SEEDS},protected_artifacts_unchanged=True))
if __name__=='__main__':main()
