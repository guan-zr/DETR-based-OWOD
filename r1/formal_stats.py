"""Same-candidate, image-cluster paired diagnostics, no trained score fusion."""
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from r1.reuse import binary_metrics,write_json

SCORES=['raw_norm','pre_norm','l2_norm','gaussian_objectness','classifier_unknown','full_error','masked_error']
G={}

def clustered_samples(labels,scores,images,weights):
    if not len(scores) or not labels.any() or labels.all():return np.full((len(weights),2),np.nan)
    order=np.argsort(-scores,kind='stable');y=labels[order].astype(float);ids=images[order];s=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];result=[]
    for row in weights:
        w=row[ids].astype(float);tp=np.cumsum(w*y)[ends];fp=np.cumsum(w)[ends]-tp
        if tp[-1]<=0 or fp[-1]<=0:result.append((np.nan,np.nan));continue
        recall=tp/tp[-1];precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        result.append((np.trapz(np.r_[0.,recall],np.r_[0.,fp/fp[-1]]),np.sum(np.diff(np.r_[0.,recall])*precision)))
    return np.asarray(result)

def initialize(root):
    G['root']=Path(root);G['weights']=np.load(Path(root)/'bootstrap_weights.npy',mmap_mode='r')

def job_run(job):
    out=G['root'];dest=out/'jobs'/f"{job['id']:04d}.json";sample=dest.with_suffix('.npy')
    if dest.exists() and sample.exists():return json.loads(dest.read_text())
    folder=out/'arrays'/job['run'];meta=np.load(folder/'metadata.npy',mmap_mode='r');a=np.load(folder/'scores.npy',mmap_mode='r')
    group=meta[:,1];keep=(group==1)|(group==2)
    if job['pool']=='selected':keep&=meta[:,2].astype(bool)
    score=np.asarray(a[keep,SCORES.index(job['score'])])*job['sign'];labels=group[keep]==1
    if not np.isfinite(score).all():raise RuntimeError('Primary comparison requires common finite candidates')
    values=clustered_samples(labels,score,meta[keep,0],G['weights']);np.save(sample,values)
    row=dict(job,**binary_metrics(labels.astype(int),score))
    for k,m in enumerate(['auroc','auprc_average_precision']):
        v=values[:,k];v=v[np.isfinite(v)];row[m+'_ci_low']=float(np.quantile(v,.025)) if len(v) else None;row[m+'_ci_high']=float(np.quantile(v,.975)) if len(v) else None
    write_json(dest,row);return row

def save_arrays(data,folder,image_map):
    folder.mkdir(exist_ok=True,parents=True)
    groups=data['group'].map({'known':0,'unknown':1,'background':2,'ambiguous':3,'invalid':4})
    if groups.isna().any():raise RuntimeError('Unexpected group')
    image=data.image_id.map(image_map)
    if image.isna().any():raise RuntimeError('Unexpected image ID')
    meta=np.column_stack([image.to_numpy(np.int32),groups.to_numpy(np.int32),data.selected.to_numpy(np.int32),data.query_index.to_numpy(np.int32)])
    matrix=np.column_stack([data[c].to_numpy(float) if c in data else np.full(len(data),np.nan) for c in SCORES])
    np.save(folder/'metadata.npy',meta);np.save(folder/'scores.npy',matrix)
    return meta,matrix

def main(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True);(out/'jobs').mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    started=time.time();manifest=json.loads((root/'evaluation_cache/manifest.json').read_text());ids=manifest['image_ids'];assert len(ids)==9119
    image_map={s:i for i,s in enumerate(ids)};rng=np.random.default_rng(91);weights=np.stack([np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)).astype(np.uint16) for _ in range(200)]);np.save(out/'bootstrap_weights.npy',weights)
    summaries={'diagnostic':json.loads((root/'diagnostic/summary.json').read_text())}
    for mask in ['0.6','0','0.3']:
        for rep in ['l2','raw']:
            for layer in [1,3,6]:
                name=f'{rep}_L{layer}_mask{mask}';summaries[name]=json.loads((root/name/'evaluation/summary.json').read_text())
    flat=[];distributions=[];counts=[]
    for run,summary in summaries.items():
        if summary['images']!=9119 or summary['scope']!='real_offline_analysis' or summary['background_rule']!='disjoint_gt':raise RuntimeError('Incomplete or wrong summary')
        for layer,pools in summary['statistics'].items():
            for pool,stats in pools.items():
                counts.append(dict(run=run,layer=int(layer),pool=pool,**stats['counts']))
                if stats['counts'].get('invalid',0):raise RuntimeError('Invalid geometry/features require investigation')
                if pool=='all' and sum(stats['counts'].values())!=911900:raise RuntimeError('Missing/extra Query rows')
                for score,values in stats['scores'].items():
                    for group,d in values['distribution'].items():distributions.append(dict(run=run,layer=int(layer),pool=pool,score=score,group=group,**{k:v for k,v in d.items() if k!='quantiles_05_50_95'},quantiles_05_50_95=json.dumps(d['quantiles_05_50_95'])))
                    for direction,comparisons in values['directions'].items():
                        for comparison,m in comparisons.items():flat.append(dict(run=run,layer=int(layer),pool=pool,score=score,direction=direction,comparison=comparison,**m))
    pd.DataFrame(flat).to_csv(out/'all_metrics.csv',index=False);pd.DataFrame(distributions).to_csv(out/'score_distributions.csv',index=False);pd.DataFrame(counts).to_csv(out/'sample_counts.csv',index=False)
    topk=[]
    for run,summary in summaries.items():
        for key,value in summary.get('fixed_topk_background_counts',{}).items():
            layer,pool,direction,k=key.split('_');n=value['effective_k'];positive=n-value['error_background_count']-value['error_ambiguous_count']
            if positive<0:raise RuntimeError('TopK partition inconsistent')
            topk.append(dict(run=run,layer=int(layer[1:]),pool=pool,direction=direction,k=int(k[3:]),**value,error_known_unknown_count=positive,error_labeled_foreground_fraction=positive/n if n else None,error_background_fraction=value['error_background_count']/n if n else None,error_ambiguous_fraction=value['error_ambiguous_count']/n if n else None,gaussian_background_fraction=value['gaussian_background_count']/n if n else None))
    pd.DataFrame(topk).to_csv(out/'topk_softlabel_audit.csv',index=False)
    use=['image_id','layer','query_index','group','selected']+SCORES[:5]
    diag=pd.read_csv(root/'diagnostic/queries.csv.gz',usecols=use,dtype={'image_id':str});jobs=[]
    for layer in range(1,7):
        name=f'diagnostic_L{layer}';d=diag.loc[diag.layer==layer];save_arrays(d,out/'arrays'/name,image_map)
        for pool in ['all','selected']:
            for score in ['raw_norm','pre_norm','gaussian_objectness','classifier_unknown']:
                jobs.append(dict(id=len(jobs),run=name,layer=layer,pool=pool,score=score,sign=1,direction='higher'))
    del diag
    conditional=[];correlations=[];thresholds=[]
    for run,summary in summaries.items():
        if run=='diagnostic':continue
        layer=summary['trained_layer'];cols=use+['full_error','masked_error'];d=pd.read_csv(root/run/'evaluation/queries.csv.gz',usecols=cols,dtype={'image_id':str})
        if len(d)!=911900 or d.layer.nunique()!=1:raise RuntimeError('R1-1 Query population mismatch')
        meta,scores=save_arrays(d,out/'arrays'/run,image_map);base=out/'arrays'/f'diagnostic_L{layer}'
        if not np.array_equal(meta,np.load(base/'metadata.npy')) or not np.array_equal(scores[:,:5],np.load(base/'scores.npy')[:,:5]):raise RuntimeError('Different Query or score population; cannot pair baseline')
        for pool in ['all','selected']:
            keep=np.ones(len(d),bool) if pool=='all' else d.selected.to_numpy(bool)
            common=keep&(meta[:,1]<3)
            for baseline in ['raw_norm','pre_norm','gaussian_objectness','classifier_unknown']:
                x=scores[common,SCORES.index(baseline)];y=scores[common,5]
                from scipy.stats import spearmanr
                correlations.append(dict(run=run,layer=layer,pool=pool,group_scope='known_unknown_background_only',baseline=baseline,count=len(x),pearson=float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 else None,spearman=float(spearmanr(x,y).statistic) if np.ptp(x)>0 else None))
            for sign,direction in [(1,'higher'),(-1,'lower')]:jobs.append(dict(id=len(jobs),run=run,layer=layer,pool=pool,score='full_error',sign=sign,direction=direction))
            # Score cutoffs depend on all valid Queries in the pool, before GT grouping.
            g=scores[:,3];cuts=np.quantile(g[keep],np.arange(1,10)/10);low=float(np.quantile(g[keep],.25));bins=np.searchsorted(cuts,g,side='right')
            thresholds.append(dict(run=run,pool=pool,cuts=cuts.tolist(),bottom25_cutoff=low,actual_bottom_fraction=float((g[keep]<=low).mean())))
            for band in list(range(10))+['bottom25']:
                selected=keep&((bins==band) if isinstance(band,int) else (g<=low))&((meta[:,1]==1)|(meta[:,1]==2));labels=meta[selected,1]==1
                for score,sign,direction in [('gaussian_objectness',1,'higher'),('classifier_unknown',1,'higher'),('raw_norm',1,'higher'),('pre_norm',1,'higher'),('full_error',1,'higher'),('full_error',-1,'lower')]:
                    conditional.append(dict(run=run,layer=layer,pool=pool,gaussian_band=str(band),score=score,direction=direction,**binary_metrics(labels.astype(int),sign*scores[selected,SCORES.index(score)])))
        del d,scores,meta
    pd.DataFrame(conditional).to_csv(out/'conditional_metrics.csv',index=False);pd.DataFrame(correlations).to_csv(out/'score_correlations.csv',index=False);write_json(out/'conditional_thresholds.json',thresholds)
    write_json(out/'job_plan.json',jobs);results=[]
    with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(out),)) as pool:
        futures=[pool.submit(job_run,j) for j in jobs]
        for i,f in enumerate(as_completed(futures)):
            results.append(f.result())
            if i%10==0:write_json(out/'progress.json',dict(completed_jobs=i+1,total_jobs=len(jobs),seconds=time.time()-started));print('Cluster statistics',i+1,'/',len(jobs),flush=True)
    ci=pd.DataFrame(results).sort_values('id');ci.to_csv(out/'unknown_cluster_intervals.csv',index=False);lookup={(r['run'],r['pool'],r['score'],r['direction']):r for r in results};paired=[]
    for r in results:
        if r['score']!='full_error':continue
        for baseline in ['gaussian_objectness','classifier_unknown','raw_norm','pre_norm']:
            b=lookup[(f"diagnostic_L{r['layer']}",r['pool'],baseline,'higher')];aa=np.load(out/'jobs'/f"{r['id']:04d}.npy");bb=np.load(out/'jobs'/f"{b['id']:04d}.npy");delta=aa-bb
            row=dict(run=r['run'],layer=r['layer'],pool=r['pool'],direction=r['direction'],baseline=baseline,bootstrap_repeats=200)
            for k,m in enumerate(['auroc','auprc_average_precision']):
                observed=r[m]-b[m];v=delta[:,k];v=v[np.isfinite(v)];row[m+'_difference']=observed;row[m+'_ci_low']=float(np.quantile(v,.025));row[m+'_ci_high']=float(np.quantile(v,.975));row[m+'_centered_bootstrap_p']=(int((np.abs(v-observed)>=abs(observed)).sum())+1)/(len(v)+1)
            paired.append(row)
    paired=pd.DataFrame(paired)
    # Approximate centered-bootstrap tests are exploratory. Correct the full family,
    # including both signs; finite B=200 severely limits minimum adjusted p.
    for metric in ['auroc','auprc_average_precision']:
        p=paired[metric+'_centered_bootstrap_p'].to_numpy();order=np.argsort(p);adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);holm=np.empty(len(p));holm[order]=np.minimum(adjusted,1.);paired[metric+'_holm_p']=holm
    paired.to_csv(out/'paired_error_minus_baseline.csv',index=False)
    config_pairs=[]
    def compare_configs(left,right,kind,layer,pool,direction):
        aa=lookup[(left,pool,'full_error',direction)];bb=lookup[(right,pool,'full_error',direction)]
        delta=np.load(out/'jobs'/f"{aa['id']:04d}.npy")-np.load(out/'jobs'/f"{bb['id']:04d}.npy")
        row=dict(left=left,right=right,comparison_family=kind,layer=layer,pool=pool,direction=direction)
        for i,m in enumerate(['auroc','auprc_average_precision']):
            v=delta[:,i];v=v[np.isfinite(v)];row[m+'_difference']=aa[m]-bb[m];row[m+'_ci_low']=float(np.quantile(v,.025));row[m+'_ci_high']=float(np.quantile(v,.975))
        config_pairs.append(row)
    for layer in [1,3,6]:
        for pool in ['all','selected']:
            for direction in ['higher','lower']:
                for mask in ['0.6','0','0.3']:compare_configs(f'raw_L{layer}_mask{mask}',f'l2_L{layer}_mask{mask}','raw_minus_l2',layer,pool,direction)
                for rep in ['l2','raw']:
                    for mask in ['0','0.3']:compare_configs(f'{rep}_L{layer}_mask0.6',f'{rep}_L{layer}_mask{mask}','mask06_minus_control',layer,pool,direction)
    pd.DataFrame(config_pairs).to_csv(out/'configuration_pair_differences.csv',index=False)
    train=[]
    for run in summaries:
        if run=='diagnostic':continue
        rows=[json.loads(l) for l in (root/run/'train.jsonl').read_text().splitlines()];canonical={r['step']:r for r in rows}
        if set(canonical)!=set(range(1,2001)):raise RuntimeError('Training budget incomplete')
        sequence=[canonical[i]['image_id'] for i in range(1,2001)]
        if 'expected_sequence' not in locals():expected_sequence=sequence
        if sequence!=expected_sequence:raise RuntimeError('Training image streams differ across configurations')
        train.append(dict(run=run,optimizer_steps=2000,logged_rows=len(rows),unique_train_images=len(set(sequence)),first_loss=canonical[1]['full_mse'],last_loss=canonical[2000]['full_mse'],last100_mean=float(np.mean([canonical[i]['full_mse'] for i in range(1901,2001)]))))
    pd.DataFrame(train).to_csv(out/'training_summary.csv',index=False);(out/'train_image_sequence.txt').write_text('\n'.join(expected_sequence)+'\n')
    cases=pd.read_csv(root/'diagnostic/unknown_cases.csv.gz',dtype={'image_id':str});cover=[]
    for layer,d in cases.groupby('layer'):
        row=dict(layer=int(layer),unknown_gt=len(d),max_iou_mean=float(d.max_iou.mean()),max_iou_median=float(d.max_iou.median()))
        for k in ['coverage_03','coverage_05','coverage_07','candidate_but_final_unknown_missed','final_unknown_geometric_hit']:
            row[k+'_count']=int(d[k].sum());row[k+'_fraction']=float(d[k].mean())
        cover.append(row)
    pd.DataFrame(cover).to_csv(out/'unknown_coverage.csv',index=False)
    write_json(out/'complete.json',dict(complete=True,images=9119,training_configs=18,training_seeds=[91],bootstrap_unit='unique image',bootstrap_repeats=200,intervals='pointwise 95% paired image cluster percentile',centered_bootstrap_tests='exploratory approximate; Holm over full both-sign family',multiple_comparisons=len(paired),seconds=time.time()-started))
