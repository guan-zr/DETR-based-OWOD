"""Complete controls only: all methods, matched geometry, seeds, conditional Query information."""
import json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,kendalltau
from r0.round2.common import *
from r0.round2.prepare import masks
from r0.round2.bootstrap import initialize,run_job

def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=True);(out/'jobs').mkdir(exist_ok=True);(out/'indices').mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('Controls analysis already complete');return
    ev=json.loads((ROOT/'evaluation/complete.json').read_text());prep=json.loads((ROOT/'prepared/complete.json').read_text())
    if ev['scope']!='full_9119_unique' or ev['unique_images']!=9119 or not ev['complete']:raise RuntimeError('Need all 9119 evaluation images')
    meta=pd.read_csv(ROOT/'prepared/metadata.csv.gz',dtype={'image_id':str},float_precision='round_trip');n=len(meta)
    if not (out/'scalar_arrays_complete.json').exists():
        scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype=np.float64,shape=(n,2,len(METHODS),5));qscores=np.lib.format.open_memmap(out/'query_scores.npy',mode='w+',dtype=np.float64,shape=(n,2));common=np.ones((n,5),bool)
        for chunk,original in zip(ev['chunks'],prep['chunks']):
            manifest_check(Path(chunk['directory'])/'complete.json');z=np.load(Path(chunk['directory'])/'scores.npz');a=original['offset'];b=a+original['rows'];scores[a:b]=z['scores'];qscores[a:b]=z['query_scores'];common[a:b]=z['valid']&np.isfinite(z['scores']).all((1,2))
            q=(meta.source.iloc[a:b]=='query').to_numpy();common[a:b][q]&=np.isfinite(z['query_scores'][q]).all(1)[:,None]
        scores.flush();qscores.flush();np.save(out/'common_valid.npy',common)
        index={s:i for i,s in enumerate(ev['image_ids'])};image_index=np.asarray([index[s] for s in meta.image_id],dtype=np.int32);np.save(out/'image_index.npy',image_index)
        rng=np.random.default_rng(73);weights=np.stack([np.bincount(rng.integers(0,9119,9119),minlength=9119).astype(np.uint16) for _ in range(200)]);np.save(out/'bootstrap_weights.npy',weights)
        write_json(out/'scalar_arrays_complete.json',dict(complete=True,rows=n,common_valid_counts=common.sum(0).tolist(),evaluation_sha256=file_sha256(ROOT/'evaluation/complete.json')))
    scores=np.load(out/'scores.npy',mmap_mode='r');qscores=np.load(out/'query_scores.npy',mmap_mode='r');common=np.load(out/'common_valid.npy');positive=masks(meta);matched=np.load(ROOT/'prepared/geometry_matches.npz');jobs=[]
    area=((meta.x2-meta.x1)*(meta.y2-meta.y1)).to_numpy();small_labels=np.select([area<32**2,area<96**2],['small','medium'],default='large')
    def add_dataset(pool,comparison,mode,indices,level_indices):
        indices=np.asarray(indices,dtype=np.int64)
        for li in level_indices:
            selected=indices[common[indices,li]];labels=positive[comparison][selected]
            name=f'{pool}__{comparison}__{mode}__{li}';p=out/'indices'/(name+'.npz')
            if not p.exists():np.savez_compressed(p,indices=selected,labels=labels)
            methods=[(s,si,m,mi) for si,s in enumerate(SOURCES) for mi,m in enumerate(METHODS)]
            if pool=='query' and li==4:methods += [('B0',-1,'gaussian',0),('B0',-1,'classifier_unknown',1)]
            for source,si,method,mi in methods:jobs.append(dict(id=len(jobs),source=source,method=method,pool=pool,comparison=comparison,mode=mode,level=LEVELS[li],source_index=si,method_index=mi,level_index=li,index_file=str(p)))
    for pool in ['gt_reference','query']:
        poolmask=(meta.source==pool).to_numpy();bg=np.flatnonzero(poolmask&(meta['group']=='background').to_numpy())
        for name,mask in positive.items():
            if pool=='query' and name.startswith('b0_'):continue
            pos=np.flatnonzero(poolmask&mask);add_dataset(pool,name,'original',np.r_[pos,bg],range(5))
            pairs=matched[pool+'__'+name+'__pairs'];add_dataset(pool,name,'geometry_matched',pairs.reshape(-1),range(5))
            add_dataset(pool,name,'matched_positives_original_background',np.r_[pairs[:,0],bg],[4])
            if name in ['unknown','b0_raw_uncovered','b0_missed_B_no_candidate']:
                for size in ['small','medium','large']:
                    selected=pairs[small_labels[pairs[:,0]]==size];add_dataset(pool,name,'geometry_matched_'+size,selected.reshape(-1),[4])
    # Thresholds fixed from all query scores; no group labels used to set bins/cutoff.
    qi=np.flatnonzero((meta.source=='query').to_numpy()&common[:,4]);g=np.asarray(qscores[qi,0]);cuts=np.quantile(g,np.arange(1,10)/10);threshold=float(np.quantile(g,.25));bin_index=np.searchsorted(cuts,g,side='right')
    for decile in range(10):add_dataset('query','unknown',f'gaussian_decile_{decile:02d}',qi[(bin_index==decile)&meta['group'].iloc[qi].isin(['unknown','background']).to_numpy()],[4])
    add_dataset('query','unknown','gaussian_bottom25',qi[(g<=threshold)&meta['group'].iloc[qi].isin(['unknown','background']).to_numpy()],[4])
    write_json(out/'query_thresholds.json',dict(source='all 911900 same-forward query scores, before group labels',gaussian_bottom25=threshold,gaussian_decile_boundaries=cuts.tolist()))
    write_json(out/'job_plan.json',jobs);started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(out),)) as pool:
        futures=[pool.submit(run_job,job) for job in jobs]
        for k,future in enumerate(as_completed(futures)):
            results.append(future.result())
            if k%100==0:write_json(out/'progress.json',dict(completed_jobs=k+1,total_jobs=len(jobs),seconds=time.time()-started));print('statistics',k+1,'/',len(jobs),time.time()-started,flush=True)
    results=sorted(results,key=lambda x:x['id']);table=pd.DataFrame(results);table.to_csv(out/'metrics.csv',index=False)
    # zero-prediction and energy are exactly the same; export the alias, never independent evidence.
    alias=table.loc[table.method=='energy'].copy();alias['method']='zero_prediction_error';alias['alias_of']='energy';pd.concat([table,alias],ignore_index=True).to_csv(out/'all_controls_metrics.csv',index=False)
    paired=[];keys=['source','pool','comparison','mode','level']
    for identity,rows in table.loc[table.source!='B0'].groupby(keys):
        lookup={r['method']:r for r in rows.to_dict('records')}
        comparisons=[('trained73',m) for m in METHODS if m!='trained73']+[('trained74','random74'),('trained75','random75'),('trained74','energy'),('trained75','energy'),('trained74','mean_error'),('trained75','mean_error')]
        # Same-frame B0 comparison is only for Query overall.
        if identity[1]=='query' and identity[4]=='overall':
            b0=table.loc[(table.source=='B0')&(table.pool==identity[1])&(table.comparison==identity[2])&(table['mode']==identity[3])&(table.level==identity[4])]
            lookup.update({r['method']:r for r in b0.to_dict('records')});comparisons += [('trained73','gaussian'),('trained73','classifier_unknown')]
        for left,right in comparisons:
            if left not in lookup or right not in lookup:continue
            a,b=lookup[left],lookup[right];aa=np.load(out/'jobs'/f"{int(a['id']):05d}.npz")['samples'];bb=np.load(out/'jobs'/f"{int(b['id']):05d}.npz")['samples'];delta=aa-bb
            row=dict(zip(keys,identity),method_left=left,method_right=right)
            for j,metric in enumerate(['auroc','auprc_average_precision']):
                valid=delta[:,j][np.isfinite(delta[:,j])];row[metric+'_difference']=a[metric]-b[metric] if pd.notna(a[metric]) and pd.notna(b[metric]) else None;row[metric+'_difference_ci_low']=float(np.quantile(valid,.025)) if len(valid) else None;row[metric+'_difference_ci_high']=float(np.quantile(valid,.975)) if len(valid) else None
            paired.append(row)
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    stability=[]
    for kind in ['trained','random']:
        f=table.loc[table.method.isin([kind+str(k) for k in SEEDS])]
        for identity,rows in f.groupby(keys):
            for metric in ['auroc','auprc_average_precision']:
                vals=rows[metric].to_numpy();stability.append(dict(zip(keys,identity),family=kind,metric=metric,seeds=3,values=';'.join(str(v) for v in vals),mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else None,sample_std=float(np.nanstd(vals,ddof=1)) if np.isfinite(vals).sum()>1 else None))
    pd.DataFrame(stability).to_csv(out/'seed_stability.csv',index=False)
    # Ordering diagnostics: no learned fusion, no post-hoc score direction change.
    correlations=[];rank_changes=[];unknown_mask=(meta['group'].iloc[qi]=='unknown').to_numpy();low=g<=threshold;rank_gauss=rankdata(g)/len(g);classifier=np.asarray(qscores[qi,1]);rank_classifier=rankdata(classifier)/len(classifier)
    for si,source in enumerate(SOURCES):
        for mi,method in enumerate(METHODS):
            v=np.asarray(scores[qi,si,mi,4]);rank=rankdata(v)/len(v)
            for bname,bscore,brank in [('gaussian',g,rank_gauss),('classifier_unknown',classifier,rank_classifier)]:
                for subset_name,selected in [('all_queries',np.ones(len(qi),bool)),('true_unknown_queries',unknown_mask)]:
                    a,b=v[selected],bscore[selected]
                    rho=float(np.corrcoef(rankdata(a),rankdata(b))[0,1]) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None
                    tau=float(kendalltau(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None
                    correlations.append(dict(source=source,method=method,baseline=bname,subset=subset_name,count=int(selected.sum()),spearman=rho,kendall_tau=tau))
                for subset_name,selected in [('true_unknown_queries',unknown_mask),('low_gaussian_true_unknown',unknown_mask&low)]:
                    d=rank[selected]-brank[selected];rank_changes.append(dict(source=source,method=method,baseline=bname,subset=subset_name,count=len(d),median_rank_change=float(np.median(d)) if len(d) else None,q25=float(np.quantile(d,.25)) if len(d) else None,q75=float(np.quantile(d,.75)) if len(d) else None))
    pd.DataFrame(correlations).to_csv(out/'query_correlations.csv',index=False);pd.DataFrame(rank_changes).to_csv(out/'query_rank_changes.csv',index=False)
    retained=[]
    for pool in ['gt_reference','query']:
        for li,level in enumerate(LEVELS):
            f=(meta.source==pool).to_numpy();retained.append(dict(pool=pool,level=level,total=int(f.sum()),common_valid=int((f&common[:,li]).sum()),excluded=int((f&~common[:,li]).sum())))
    pd.DataFrame(retained).to_csv(out/'common_valid_counts.csv',index=False)
    write_json(out/'complete.json',dict(complete=True,unique_images=9119,scalar_rows=n,metric_rows=len(table),bootstrap_repeats=200,training_seeds=SEEDS,finished_at=stamp(),
        sha256={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.csv','.json'] and p.name!='complete.json'}))
if __name__=='__main__':main()
