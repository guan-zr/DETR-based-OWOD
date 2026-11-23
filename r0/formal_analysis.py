"""Complete-artifact-only analysis. Fixed directions and strata; single training seed, clustered image uncertainty."""
import argparse,csv,gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from r0.runtime import write_json,file_sha256
from r0.metrics import binary_metrics

ROOT=Path('/root/autodl-tmp/experiments/R0')
LEVELS=['s0_stride8','s1_stride16','s2_stride32','s3_stride64','overall']

def metric_frame(frame,group='group',positive='unknown'):
    if positive=='foreground':chosen=frame[group].isin(['known','unknown','background']);y=frame.loc[chosen,group]!='background'
    else:chosen=frame[group].isin([positive,'background']);y=frame.loc[chosen,group]==positive
    return binary_metrics(y.astype(int).to_numpy(),frame.loc[chosen,'score'].to_numpy())

def bootstrap_pair(frame,ids,repeats=200):
    """Exact weighted tie-aware AUROC/AP, resampling images jointly for S1/S2."""
    nimages=len(ids);idmap={s:i for i,s in enumerate(ids)}
    image=np.asarray([idmap[s] for s in frame['image_id']],dtype=np.int32);label=(frame['group']=='unknown').to_numpy()
    prepared={}
    for source in ['S1','S2']:
        scores=frame['score_'+source].to_numpy();order=np.argsort(-scores,kind='stable');ordered=scores[order]
        ends=np.r_[np.flatnonzero(np.diff(ordered)),len(ordered)-1] if len(ordered) else np.array([],dtype=int)
        prepared[source]=(order,ends)
    rng=np.random.default_rng(73);values={s:[] for s in ['S1','S2']}
    for b in range(repeats):
        counts=np.bincount(rng.integers(0,nimages,nimages),minlength=nimages);weight=counts[image]
        for source,(order,ends) in prepared.items():
            y=label[order];w=weight[order];tp=np.cumsum(w*y)[ends];fp=np.cumsum(w*(~y))[ends]
            if not len(tp) or tp[-1]==0 or fp[-1]==0:values[source].append((np.nan,np.nan));continue
            tpr=tp/tp[-1];fpr=fp/fp[-1]
            precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)
            auc=np.trapz(np.r_[0,tpr],np.r_[0,fpr]);ap=np.sum(np.diff(np.r_[0,tpr])*precision)
            values[source].append((auc,ap))
    arrays={s:np.asarray(v) for s,v in values.items()}
    result={}
    for source,v in [*arrays.items(),('S2_minus_S1',arrays['S2']-arrays['S1'])]:
        result[source]={metric:dict(mean=float(np.nanmean(v[:,i])),ci95=np.nanquantile(v[:,i],[.025,.975]).tolist()) for i,metric in enumerate(['auroc','auprc_average_precision'])}
    return dict(repeats=repeats,seed=73,unit='unique image, paired resampling',training_seeds=1,interval_scope='evaluation image sampling; not training-seed uncertainty',estimates=result)

def analyze(evaluation=None,output=None):
    evaluation=Path(evaluation or ROOT/'paired_evaluation');out=Path(output or ROOT/'analysis');out.mkdir(parents=True,exist_ok=True)
    complete=json.loads((evaluation/'complete.json').read_text())
    if not complete.get('complete') or complete['scope']!='full_9119_unique' or complete['unique_images']!=9119:raise RuntimeError('Analysis requires complete 9119-image real evaluation')
    source_sha=file_sha256(evaluation/'complete.json')
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        if done['evaluation_complete_sha256']!=source_sha:raise RuntimeError('Analysis source changed')
        for p,h in done['artifacts'].items():
            if file_sha256(out/p)!=h:raise RuntimeError('Analysis artifact changed')
        return
    frames=[];audit=defaultdict(lambda:defaultdict(Counter));count_rows=0
    # Scalar-only frame; no high-dimensional features retained.
    use=['image_id','source','candidate_id','origin','feature_source','group_iou_only','group_disjoint_gt','include_disjoint_gt',
        'level','score','valid','valid_fraction','small_feature_roi','x1','y1','x2','y2','original_h','original_w','class_name','gt_index','b0_max_iou','batch1_max_iou','batch1_clipped_max_iou','b0_miss_type']
    for chunk in complete['chunks']:
        directory=Path(chunk['directory'])
        for filename,digest in chunk['sha256'].items():
            if file_sha256(directory/filename)!=digest:raise RuntimeError('Damaged evaluation chunk')
        df=pd.read_csv(directory/'regions.csv.gz',usecols=use,dtype={'image_id':str,'class_name':str,'b0_miss_type':str})
        if len(df)!=chunk['rows']:raise RuntimeError('Row count mismatch')
        frames.append(df);count_rows+=len(df)
        for pool,items in json.loads((directory/'grouping_audit.json').read_text()).items():
            for key,counts in items.items():audit[pool][key].update(counts)
    if count_rows!=complete['rows']:raise RuntimeError('Total row mismatch')
    data=pd.concat(frames,ignore_index=True);del frames
    for c in ['source','origin','feature_source','group_iou_only','group_disjoint_gt','level','class_name','b0_miss_type']:data[c]=data[c].astype('category')
    write_json(out/'grouping_audit.json',{pool:{key:dict(counts) for key,counts in items.items()} for pool,items in audit.items()})
    w=data.x2-data.x1;h=data.y2-data.y1;area=w*h;relative=area/(data.original_h*data.original_w)
    data['area_bin']=np.select([area<32**2,area<96**2],['small_lt32sq','medium_32sq_to96sq'],default='large_ge96sq')
    data['relative_area_bin']=np.select([relative<.01,relative<.1],['lt1pct','1_to10pct'],default='ge10pct')
    aspect=w/h.where(h>0,np.nan)
    data['aspect_bin']=np.select([aspect<.5,aspect>2],['tall_lt0.5','wide_gt2'],default='0.5_to2')
    data['support_bin']=np.where(data.valid_fraction>=.9,'ge0.9','lt0.9')
    metrics=[];distributions=[];strata=[];overall=[];scales=[]
    for rule in ['disjoint_gt','iou_only']:
        eligible=data.loc[data.include_disjoint_gt] if rule=='disjoint_gt' else data
        for (source,pool,level),frame in eligible.groupby(['feature_source','source','level'],observed=True,sort=True):
            f=frame.copy();f['group']=f['group_'+rule].astype(str)
            counts=f.groupby(['group','valid'],observed=True).size()
            for group in ['known','unknown','background','ambiguous','invalid']:
                for flag in [True,False]:
                    part=f.loc[(f['group']==group)&(f.valid==flag)];count=int(counts.get((group,flag),0))
                    distributions.append(dict(method=source,pool=pool,rule=rule,level=level,group=group,valid=flag,count=count,
                        small_feature_count=int(part['small_feature_roi'].sum()),valid_fraction_mean=float(part.valid_fraction.mean()) if count else None))
            valid=f.loc[f.valid & np.isfinite(f.score)].copy()
            for group,sub in valid.groupby('group',observed=True):
                scores=sub.score.to_numpy();distributions.append(dict(method=source,pool=pool,rule=rule,level=level,group=group,valid=True,
                    distribution_record=True,count=len(scores),mean=float(scores.mean()),std=float(scores.std()),**{'q'+str(int(q*100)):float(np.quantile(scores,q)) for q in [.05,.25,.5,.75,.95]}))
            for positive in ['known','unknown','foreground']:
                metrics.append(dict(method=source,pool=pool,rule=rule,level=level,comparison=positive+'_vs_background',**metric_frame(valid,positive=positive)))
            if level=='overall':
                valid['method']=source;valid['pool']=pool;valid['rule']=rule;overall.append(valid)
                for kind in ['area_bin','relative_area_bin','aspect_bin','support_bin','small_feature_roi']:
                    for value,part in valid.groupby(kind,observed=True):
                        for positive in ['known','unknown','foreground']:
                            strata.append(dict(method=source,pool=pool,rule=rule,stratum=kind,value=value,comparison=positive+'_vs_background',**metric_frame(part,positive=positive)))
            elif rule=='disjoint_gt':
                sub=valid.loc[valid['group'].isin(['known','unknown','background']),['image_id','candidate_id','source','score']].copy();sub['method']=source;sub['level']=level;scales.append(sub)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(distributions).to_csv(out/'counts_distributions.csv',index=False);pd.DataFrame(strata).to_csv(out/'geometry_strata.csv',index=False)
    over=pd.concat(overall,ignore_index=True);over['area_pixels']=(over.x2-over.x1)*(over.y2-over.y1);over['relative_area']=over.area_pixels/(over.original_h*over.original_w);over['aspect_ratio']=(over.x2-over.x1)/(over.y2-over.y1)
    geometry=[]
    for keys,sub in over.groupby(['method','pool','rule','group'],observed=True):
        for name in ['area_pixels','relative_area','aspect_ratio','valid_fraction']:
            geometry.append(dict(zip(['method','pool','rule','group'],keys),quantity=name,count=len(sub),mean=float(sub[name].mean()),q05=float(sub[name].quantile(.05)),q50=float(sub[name].median()),q95=float(sub[name].quantile(.95))))
    pd.DataFrame(geometry).to_csv(out/'geometry_distributions.csv',index=False)
    unknown=[];perclass=[];coverage_context=[];gt_identity_counts={}
    for keys,frame in over.groupby(['method','pool','rule'],observed=True):
        source,pool,rule=keys
        if pool!='gt_reference':continue
        gt=frame.loc[(frame.origin=='annotated_gt')&(frame['group']=='unknown')];bg=frame.loc[frame['group']=='background']
        if rule=='disjoint_gt':
            all_gt=data.loc[(data.feature_source==source)&(data.source=='gt_reference')&(data.level=='overall')&(data.group_disjoint_gt=='unknown')]
            gt_identity_counts[source]=dict(total=len(all_gt),valid=int(all_gt.valid.sum()),covered_b0=int((all_gt.b0_max_iou>=.5).sum()),
                uncovered_b0=int((all_gt.b0_max_iou<.5).sum()),missed_B_no_candidate=int((all_gt.b0_miss_type=='B_no_candidate').sum()),
                covered_batch1=int((all_gt.batch1_max_iou>=.5).sum()),uncovered_batch1=int((all_gt.batch1_max_iou<.5).sum()))
        definitions={
            'b0_raw_covered':gt.b0_max_iou>=.5,'b0_raw_uncovered':gt.b0_max_iou<.5,
            'b0_missed_B_no_candidate':gt.b0_miss_type=='B_no_candidate',
            'batch1_raw_covered':gt.batch1_max_iou>=.5,'batch1_raw_uncovered':gt.batch1_max_iou<.5}
        for category,mask in definitions.items():
            pos=gt.loc[mask];selected=pd.concat([pos,bg])
            stat=dict(method=source,rule=rule,category=category,mean_score=float(pos.score.mean()) if len(pos) else None,
                median_score=float(pos.score.median()) if len(pos) else None,small_feature_fraction=float(pos.small_feature_roi.mean()) if len(pos) else None,
                valid_fraction_mean=float(pos.valid_fraction.mean()) if len(pos) else None,**metric_frame(selected))
            unknown.append(stat)
            for kind in ['class_name','area_bin','relative_area_bin','aspect_bin']:
                for value,part in pos.groupby(kind,observed=True):
                    # Fixed within-geometry stratum background, all reliable backgrounds for class analysis.
                    negative=bg.loc[bg[kind]==value] if kind!='class_name' else bg
                    perclass.append(dict(method=source,rule=rule,category=category,stratum=kind,value=value,mean_score=float(part.score.mean()),**metric_frame(pd.concat([part,negative]))))
        if rule=='disjoint_gt':
            coverage_context.append(dict(method=source,valid_unknown=len(gt),coverage_changed=int(((gt.b0_max_iou>=.5)!=(gt.batch1_max_iou>=.5)).sum()),
                b0_covered_to_batch1_uncovered=int(((gt.b0_max_iou>=.5)&(gt.batch1_max_iou<.5)).sum()),
                b0_uncovered_to_batch1_covered=int(((gt.b0_max_iou<.5)&(gt.batch1_max_iou>=.5)).sum()),mean_abs_max_iou_change=float((gt.b0_max_iou-gt.batch1_max_iou).abs().mean()),
                clipping_coverage_changed=int(((gt.batch1_max_iou>=.5)!=(gt.batch1_clipped_max_iou>=.5)).sum())))
    pd.DataFrame(unknown).to_csv(out/'unknown_coverage.csv',index=False);pd.DataFrame(perclass).to_csv(out/'unknown_class_size.csv',index=False)
    write_json(out/'coverage_context.json',dict(identity_counts=gt_identity_counts,valid_context=coverage_context,
        note='B0 first-occurrence batch5 raw coverage differs from official all-row matching and R0 batch1. B_no_candidate excludes officially detected but raw-first-uncovered objects.'))
    primary=over.loc[(over.rule=='disjoint_gt')&over['group'].isin(['known','unknown','background'])].copy();common=[];cis={};common_frames={}
    key=['pool','image_id','candidate_id','origin','group']
    left=primary.loc[primary.method=='S1',key+['score']];right=primary.loc[primary.method=='S2',key+['score']]
    paired=left.merge(right,on=key,suffixes=('_S1','_S2'),validate='one_to_one')
    for pool,frame in paired.groupby('pool',observed=True):
        common_frames[pool]=frame
        for source in ['S1','S2']:
            f=frame.rename(columns={'score_'+source:'score'})
            for positive in ['known','unknown','foreground']:common.append(dict(method=source,pool=pool,comparison=positive+'_vs_background',**metric_frame(f,positive=positive)))
        cis[pool]=bootstrap_pair(frame.loc[frame['group'].isin(['unknown','background'])],complete['image_ids'])
    pd.DataFrame(common).to_csv(out/'common_valid_metrics.csv',index=False);write_json(out/'cluster_bootstrap.json',cis)
    dominance=[]
    scale=pd.concat(scales,ignore_index=True);pivot=scale.pivot(index=['method','source','image_id','candidate_id'],columns='level',values='score').dropna()
    for (method,pool),sub in pivot.groupby(level=['method','source']):
        values=sub[LEVELS[:4]].to_numpy();denom=values.sum(1);share=np.divide(values,denom[:,None],out=np.zeros_like(values),where=denom[:,None]>0)
        for i,level in enumerate(LEVELS[:4]):dominance.append(dict(method=method,pool=pool,level=level,mean_mse=float(values[:,i].mean()),mean_share_in_raw_sum=float(share[:,i].mean()),common_valid_candidates=len(sub)))
    pd.DataFrame(dominance).to_csv(out/'scale_dominance.csv',index=False)
    # Exact scalar overall record table supports all diagnostics without a spatial cache.
    over.to_csv(out/'overall_records.csv.gz',index=False,compression='gzip')
    write_json(out/'analysis_facts.json',dict(evaluation_images=9119,evaluation_rows=count_rows,training_seeds=1,seed=73,primary_metric='higher reconstruction error: Unknown vs disjoint background AUROC',
        inference_unit='unique image; candidate observations within an image are dependent',no_test_selection=True,unknown_coverage=unknown,coverage_context=coverage_context,bootstrap=cis))
    # Standard static artifacts for a standalone server-side report.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    table=pd.DataFrame(metrics)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,pool in zip(axes,['gt_reference','query']):
        for source,color in [('S1','#0072B2'),('S2','#D55E00')]:
            part=table.loc[(table.method==source)&(table.pool==pool)&(table.rule=='disjoint_gt')&(table.comparison=='unknown_vs_background')].set_index('level').reindex(LEVELS)
            ax.plot(range(5),part.auroc,marker='o',color=color,label=source)
        ax.set(xticks=range(5),xticklabels=['8','16','32','64','mean'],ylim=(0,1),title=pool,xlabel='Stride / raw mean',ylabel='Unknown vs background AUROC');ax.axhline(.5,color='gray',ls='--');ax.legend()
    save(fig,'main_separability')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    coverage=pd.DataFrame(unknown)
    for ax,source in zip(axes,['S1','S2']):
        part=coverage.loc[(coverage.method==source)&(coverage.rule=='disjoint_gt')&coverage.category.isin(['b0_raw_covered','b0_raw_uncovered','b0_missed_B_no_candidate'])]
        ax.bar(range(len(part)),part.auroc,color=['#009E73','#D55E00','#CC79A7']);ax.set(xticks=range(len(part)),xticklabels=['covered','uncovered','missed B'],ylim=(0,1),title=source,ylabel='GT-reference AUROC');ax.axhline(.5,color='gray',ls='--')
    save(fig,'unknown_coverage')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for source in ['S1','S2']:
        state=json.loads((ROOT/'pipeline/stages'/f'{source}_train.json').read_text());lines=[json.loads(l) for l in (Path(state['artifact_directory'])/'train.jsonl').read_text().splitlines()]
        # Resumed steps may have earlier uncommitted log entries; last occurrence per fixed optimizer step is canonical.
        canonical={r['step']:r for r in lines};assert set(canonical)==set(range(1,2001))
        vals=np.array([canonical[k]['loss'] for k in range(1,2001)]);ax.plot(range(1,2001),pd.Series(vals).rolling(25,min_periods=1).mean(),label=source)
        pd.DataFrame([canonical[k] for k in sorted(canonical)]).to_json(out/f'{source}_canonical_train.jsonl',orient='records',lines=True)
    ax.set(xlabel='Optimizer step',ylabel='Mean valid MSE (25-step moving mean)',yscale='log');ax.legend();save(fig,'training_loss')
    catalog='''# Figure catalog\n\n- figures/main_separability.png / .svg: 每尺度及原始均值 Unknown/Background AUROC；数据 metrics.csv；无误差条，单训练seed；比较排名信号而非MSE幅值；两个候选池单独解释。\n- figures/unknown_coverage.png / .svg: B0原始Query已覆盖、未覆盖及官方漏检B类GT诊断；数据 unknown_coverage.csv；不表示自动发现目标，不能把11490和11434混用。\n- figures/training_loss.png / .svg: 固定2000步训练动态；25步滑动均值仅用于绘图；不用于选择步数或模型。\n\n每张图的确切数值见对应CSV。主图判断信号是否跨尺度一致；覆盖图判断信号是否也存在于给定的漏检GT区域；loss图排查非有限或未完成训练。\n'''
    (out/'figure-catalog.md').write_text(catalog)
    (out/'stats-appendix.md').write_text('''# Statistical appendix\n\n固定seed73，S1/S2各一次训练；没有训练seed方差或跨运行显著性结论。主要问题为误差越高越像前景的Unknown vs disjoint Background AUROC，同时报告AP及正例比例。所有候选按图像聚类，不能按候选数量声称独立样本数。cluster_bootstrap.json对共同有效候选进行200次固定seed的图像成对bootstrap，给出S1、S2及S2−S1的AUROC/AP 95%百分位区间。此区间只覆盖评价图像抽样不确定性；有限bootstrap重复使尾部区间存在Monte Carlo误差。无p值检验、无多重检验显著性主张。类别及几何分层仅作描述，缺任一组时指标未定义。\n\n效应量：AUROC及其共同集合成对差值；AP依赖正例比例，不能无视不同候选池和规则的样本分布。所有区间与分层均未用于模型、方向、尺度或掩码选择。单一固定seed不足以验证跨训练重复性。\n''')
    (out/'analysis-report.md').write_text('''# R0 evidence-first analysis\n\n主要比较：冻结B0时，同一固定训练预算下，S1/S2的空间重建误差能否区分Unknown和可靠背景；原定方向为高误差前景。完整9119唯一图验收后方生成此文件。\n\n精确数值：metrics.csv（两个池、两规则、五种尺度、Known/Unknown/合并前景）、common_valid_metrics.csv、cluster_bootstrap.json。有效/无效/Ambiguous/Invalid数量与分布见counts_distributions.csv。几何分层见geometry_strata.csv、geometry_distributions.csv；B0遗漏区域见unknown_coverage.csv、unknown_class_size.csv与coverage_context.json。\n\n## Claim candidates\n\n- Claim: 此固定seed、候选池及标注背景定义下存在或不存在排序可分性。\n  - Source evidence: metrics.csv, cluster_bootstrap.json。\n  - Allowed wording: 在本次固定条件下观察到的AUROC/AP及其图像抽样区间。\n  - Forbidden stronger wording: 跨训练seed稳定、语义因果、真实Unknown Discovery提升或创新性证明。\n  - Uncertainty: 单seed；GT参考候选；非穷尽标注；尺度与几何混杂。\n  - Next check: 对照面积形状及共同有效集合，不以测试GT调整本次方案。\n  - Decision: weaken（限定为本次空间区域诊断）。\n''')
    artifacts={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}
    write_json(out/'complete.json',dict(complete=True,evaluation_complete_sha256=source_sha,artifacts=artifacts))
    print('Analysis complete:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation');p.add_argument('--output');a=p.parse_args();analyze(a.evaluation,a.output)
