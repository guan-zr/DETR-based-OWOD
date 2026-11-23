"""Exact tie-aware image-cluster bootstrap; shared draws make method differences paired."""
import json
from pathlib import Path
import numpy as np
from r0.metrics import binary_metrics
from r0.round2.common import METHODS,ROOT,write_json
G={}

def initialize(folder):
    folder=Path(folder);G.update(scores=np.load(folder/'scores.npy',mmap_mode='r'),query=np.load(folder/'query_scores.npy',mmap_mode='r'),images=np.load(folder/'image_index.npy',mmap_mode='r'),weights=np.load(folder/'bootstrap_weights.npy',mmap_mode='r'))

def clustered_samples(labels,scores,image_indices,weights):
    if not len(scores) or not labels.any() or labels.all():return np.full((len(weights),2),np.nan)
    order=np.argsort(-scores,kind='stable');y=labels[order].astype(float);images=image_indices[order];s=scores[order];ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];result=[]
    for row in weights:
        w=row[images].astype(float);tp=np.cumsum(w*y)[ends];fp=np.cumsum(w)[ends]-tp
        if tp[-1]<=0 or fp[-1]<=0:result.append((np.nan,np.nan));continue
        recall=tp/tp[-1];precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        result.append((np.trapz(np.r_[0.,recall],np.r_[0.,fp/fp[-1]]),np.sum(np.diff(np.r_[0.,recall])*precision)))
    return np.asarray(result)

def run_job(job):
    folder=ROOT/'analysis/jobs';path=folder/f"{job['id']:05d}.json";sample_path=path.with_suffix('.npz')
    if path.exists() and sample_path.exists():return json.loads(path.read_text())
    indices=np.load(job['index_file']);selected=indices['indices'];labels=indices['labels'].astype(bool)
    scores=np.asarray(G['query'][selected,job['method_index']]) if job['source_index']==-1 else np.asarray(G['scores'][selected,job['source_index'],job['method_index'],job['level_index']])
    if not np.isfinite(scores).all():raise RuntimeError('Common-valid candidate set contains nonfinite score')
    observed=binary_metrics(labels.astype(int),scores);samples=clustered_samples(labels,scores,G['images'][selected],G['weights'])
    row={k:v for k,v in job.items() if k not in ['source_index','method_index','level_index','index_file']};row.update(observed)
    for i,name in enumerate(['auroc','auprc_average_precision']):
        v=samples[:,i];v=v[np.isfinite(v)];row[name+'_ci_low']=float(np.quantile(v,.025)) if len(v) else None;row[name+'_ci_high']=float(np.quantile(v,.975)) if len(v) else None
    row['bootstrap_valid_repeats']=int(np.isfinite(samples).all(1).sum())
    for group,mask in [('positive',labels),('background',~labels)]:
        a=scores[mask];row[group+'_score_mean']=float(a.mean()) if len(a) else None;row[group+'_score_std']=float(a.std()) if len(a) else None;row[group+'_score_median']=float(np.median(a)) if len(a) else None
    np.savez_compressed(sample_path,samples=samples);write_json(path,row);return row
