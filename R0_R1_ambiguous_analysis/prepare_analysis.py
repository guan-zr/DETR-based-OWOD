"""Read existing scalar outputs only. No model import, forward pass, or training."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mpl_cache')
import json, hashlib, time, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
BASE=Path('/root/autodl-tmp')
R0=BASE/'experiments/R0'; R2=R0/'controls_round2'; R1=BASE/'experiments/R1/formal_20261112'
SOURCES=[]
def record(path, role, expected=None):
    path=Path(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    digest=h.hexdigest()
    if expected is not None:assert digest==expected, str(path)
    SOURCES.append(dict(path=str(path),role=role,bytes=path.stat().st_size,sha256=digest))
    return path

def read_json(path,role):return json.loads(record(path,role).read_text())
start=time.time()
plan=ROOT/'task_plan.md';print(plan.read_text(),flush=True)
protocol=read_json(R2/'protocol.json','R0 second-round frozen protocol')
ev=read_json(R2/'evaluation/complete.json','R0 second-round method/scale order and complete population')
assert ev['complete'] and ev['scope']=='full_9119_unique' and ev['unique_images']==9119
prep=read_json(R2/'prepared/complete.json','R0 metadata provenance')
mp=record(R2/'prepared/metadata.csv.gz','R0 second-round candidate keys, clipped boxes, disjoint_gt labels',prep['metadata_sha256'])
meta=pd.read_csv(mp,dtype={'image_id':str},float_precision='round_trip')
qi=np.flatnonzero(meta.source.to_numpy()=='query');q=meta.iloc[qi].copy().reset_index(drop=True)
assert len(q)==911900 and q.image_id.nunique()==9119 and not q.duplicated(['image_id','candidate_id']).any()
assert np.array_equal(q.candidate_id.to_numpy(),np.tile(np.arange(100),9119))
a=np.load(record(R2/'analysis/scores.npy','R0 second-round scores; select S2 / trained73 / overall'),mmap_mode='r')
source_index=['S1','S2'].index('S2');method_index=ev['methods'].index('trained73');level_index=ev['levels'].index('overall')
error0=np.asarray(a[qi,source_index,method_index,level_index]).copy()
assert np.allclose(error0,np.asarray(a[qi,source_index,method_index,:4]).mean(1),rtol=0,atol=1e-12)
cv=np.load(record(R2/'analysis/common_valid.npy','R0 common-valid mask'),mmap_mode='r')
assert cv[qi,level_index].all() and np.isfinite(error0).all()
qscores=np.load(record(R2/'analysis/query_scores.npy','R0 same-forward B0 Gaussian scalar baseline'),mmap_mode='r')
gauss0=np.asarray(qscores[qi,0]).copy()
print('R0 second-round errors loaded',len(q), 'elapsed',time.time()-start,flush=True)
# The second-round metadata omitted max_iou_gt. Restore ONLY geometry fields
# from its original candidate cache, never first-round reconstruction scores.
original=read_json(R0/'paired_evaluation/complete.json','Geometry-only original candidate-cache manifest')
ious=[]
cols=['image_id','source','candidate_id','feature_source','level','max_iou_gt','group_disjoint_gt']
for k,ch in enumerate(original['chunks']):
    path=record(Path(ch['directory'])/'regions.csv.gz','R0 geometry-only recovery: max_iou_gt and candidate identity',ch['sha256']['regions.csv.gz'])
    d=pd.read_csv(path,usecols=cols,dtype={'image_id':str},float_precision='round_trip')
    d=d.loc[(d.source=='query')&(d.feature_source=='S2')&(d.level=='overall')]
    ious.append(d[['image_id','candidate_id','group_disjoint_gt','max_iou_gt']])
    if k%25==0:print('Geometry cache read',k+1,'/',len(original['chunks']),'elapsed',round(time.time()-start,1),flush=True)
geo=pd.concat(ious,ignore_index=True)
assert list(zip(geo.image_id,geo.candidate_id))==list(zip(q.image_id,q.candidate_id))
assert np.array_equal(geo.group_disjoint_gt.to_numpy(),q['group'].to_numpy())
iou0=geo.max_iou_gt.to_numpy()
summary=read_json(R1/'l2_L6_mask0/evaluation/summary.json','R1 exact run config and original group counts')
cfg=read_json(R1/'configs/l2_L6_mask0.json','R1 formal configuration')
assert cfg['layer']==6 and cfg['representation']=='l2' and cfg['mask_ratio']==0 and cfg['seed']==91
assert summary['evaluation_config']==cfg and summary['trained_layer']==6 and summary['images']==9119 and summary['background_rule']=='disjoint_gt'
r=pd.read_csv(record(R1/'l2_L6_mask0/evaluation/queries.csv.gz','R1 formal all-query full_error, IoU, nearest GT and Gaussian'),dtype={'image_id':str},float_precision='round_trip')
assert len(r)==911900 and r.layer.eq(6).all() and not r.duplicated(['image_id','query_index']).any()
# A key join is explicit; no assumption about implicit R0/R1 row order.
idx=pd.MultiIndex.from_frame(q[['image_id','candidate_id']]);r=r.set_index(['image_id','query_index']).loc[idx].reset_index()
assert np.array_equal(r['group'].to_numpy(),q['group'].to_numpy())
assert np.array_equal(r.max_iou_gt.to_numpy(),iou0)
error1=r.full_error.to_numpy();gauss1=r.gaussian_objectness.to_numpy()
assert np.isfinite(error1).all() and np.isfinite(gauss0).all() and np.isfinite(gauss1).all()
assert (error0>=0).all() and (error1>=0).all() and ((iou0>=0)&(iou0<=1)).all()
# Verify same final-query clipped geometry from existing R1 cache. Read only
# box metadata members of each NPZ (no feature tensors, no model inference).
cache_manifest=read_json(R1/'evaluation_cache/manifest.json','R1 box-cache provenance')
record(R1/'evaluation_cache/index.jsonl','R1 existing cache index')
xml_paths=[];box_paths=[];high_difficult=0;high_not_difficult=0;ignored_other=0
for k,(stem,f) in enumerate(q.groupby('image_id',sort=False)):
    indices=f.index.to_numpy();path=R1/'evaluation_cache'/f'{stem}.npz'
    with np.load(path,allow_pickle=False) as z:
        layers=z['layer_ids'].tolist();boxes=z['boxes_xyxy'][layers.index(6)].copy();hw=z['original_hw']
    boxes[:,0::2]=np.clip(boxes[:,0::2],0,hw[1]);boxes[:,1::2]=np.clip(boxes[:,1::2],0,hw[0])
    assert np.array_equal(boxes,f[['x1','y1','x2','y2']].to_numpy().astype(np.float32)),stem
    box_paths.append(str(path))
    high=indices[(q['group'].to_numpy()[indices]=='ambiguous')&(iou0[indices]>=.5)]
    if len(high):
        xp=BASE/'datasets/OWOD_B0/Annotations'/f'{stem}.xml';tree=ET.parse(xp).getroot();objects=tree.findall('object');xml_paths.append(str(xp))
        for ix in high:
            obj=objects[int(r.matched_gt_index.iloc[ix])];difficult=bool(int(obj.findtext('difficult','0')))
            high_difficult+=int(difficult);high_not_difficult+=int(not difficult)
    if k%2000==0:print('Cached box identity checked',k+1,'/9119',flush=True)
assert high_not_difficult==0
# Audit original counts against both official summaries.
audit0=read_json(R0/'analysis/grouping_audit.json','Original R0 disjoint_gt Query counts')
expected0=audit0['query']['after_disjoint_gt'];expected1=summary['statistics']['6']['all']['counts']
counts={g:int(q['group'].eq(g).sum()) for g in ['known','unknown','background','ambiguous','invalid']}
assert all(counts[g]==expected0.get(g,0)==expected1.get(g,0) for g in counts)
assert sum(counts.values())==911900
assert (iou0[q['group'].eq('background').to_numpy()]==0).all()
assert (r.intersects_any_gt[q['group'].eq('ambiguous').to_numpy()]).all()
for g in ['known','unknown','background']:
    expected=summary['statistics']['6']['all']['scores']['full_error']['distribution'][g]
    assert np.isclose(error1[q['group'].eq(g)].mean(),expected['mean'],rtol=1e-12,atol=1e-15)
    assert np.isclose(np.median(error1[q['group'].eq(g)]),expected['quantiles_05_50_95'][1],rtol=1e-12,atol=1e-15)
code=np.array([{'known':0,'unknown':1,'background':2,'ambiguous':3}[g] for g in q['group']],dtype=np.uint8)
np.savez_compressed(ROOT/'paired_query_data.npz',image_id=q.image_id.to_numpy(dtype='U12'),query_id=q.candidate_id.to_numpy(np.int16),group=code,max_iou_gt=iou0,r0_error=error0,r1_error=error1,gaussian=gauss0,gaussian_r1=gauss1)
# Record every box/XML path, but avoid re-hashing multi-GB feature NPZ files:
# the accessed members are geometry only; cache index stores original hashes.
(ROOT/'geometry_cache_paths.json').write_text(json.dumps(dict(box_files=box_paths,annotation_files=xml_paths),ensure_ascii=False,indent=2))
manifest=dict(primary_question='DETR 中重建误差能否区分前景与背景；Ambiguous 中是否也有物体性诊断信息',source_files=SOURCES,r0_indices=dict(source=source_index,method=method_index,level=level_index),r0_counts=counts,r1_counts=counts,original_r0_counts=expected0,original_r1_counts=expected1,query_total=911900,images=9119,all_query_per_image=100,r0_r1_clipped_boxes_exact_equal=True,r0_r1_groups_exact_equal=True,r0_r1_max_iou_exact_equal=True,gaussian_max_abs_r0_r1=float(np.max(abs(gauss0-gauss1))),ambiguous_high_iou_difficult=high_difficult,ambiguous_high_iou_other_reason=high_not_difficult,excluded_missing_error=0,inference_performed=False,training_performed=False,geometry_recovery_note='Only max_iou_gt/key/group read from original R0 candidate cache. All R0 errors exclusively second-round S2 trained73 overall.',seconds=time.time()-start)
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('AUDIT',json.dumps({k:v for k,v in manifest.items() if k not in ['source_files','primary_question']},ensure_ascii=False),flush=True)
with (ROOT/'notes.md').open('a') as f:f.write('\nR0 第二轮元数据缺 max_iou_gt，已从原候选几何缓存恢复，未使用原缓存重建误差。R0/R1 所有 911900 个最终 Query 的裁剪框、标签、最大 IoU 完全一致。\n'+json.dumps(counts)+'\n')
s=plan.read_text().replace('- [ ] 核验逐 Query 数据、标签规则、IoU 及原始计数','- [x] 核验逐 Query 数据、标签规则、IoU 及原始计数').replace('核验数据。','计算描述统计与绘图。');plan.write_text(s)
