"""CPU-only reuse of exact round-one scalar records and geometry-only matching."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from r0.round2.common import *

def match_geometry(meta,positive,background,caliper=np.log(1.25)):
    def vectors(indices):
        f=meta.iloc[indices];area=(f.x2-f.x1)*(f.y2-f.y1);relative=area/(f.original_h*f.original_w);aspect=(f.x2-f.x1)/(f.y2-f.y1)
        return np.log(np.stack([area,relative,aspect],1))
    positive=np.asarray(positive,dtype=np.int64);background=np.asarray(background,dtype=np.int64)
    if not len(background):return np.empty((0,2),np.int64),positive
    pv,bv=vectors(positive),vectors(background);tree=cKDTree(bv);used=np.zeros(len(background),bool);pairs=[];unmatched=[]
    order=sorted(range(len(positive)),key=lambda j:hashlib.sha256(f"{meta.iloc[positive[j]].image_id}/{meta.iloc[positive[j]].source}/{meta.iloc[positive[j]].candidate_id}/73".encode()).digest())
    for j in order:
        candidates=np.asarray(tree.query_ball_point(pv[j],r=caliper,p=np.inf),dtype=np.int64)
        candidates=candidates[~used[candidates]]
        if not len(candidates):unmatched.append(positive[j]);continue
        distance=((bv[candidates]-pv[j])**2).sum(1)
        choice=candidates[np.lexsort((background[candidates],distance))[0]];used[choice]=True;pairs.append((positive[j],background[choice]))
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2),np.asarray(unmatched,dtype=np.int64)

def masks(meta):
    g=meta['group'].to_numpy();return dict(known=g=='known',unknown=g=='unknown',foreground=np.isin(g,['known','unknown']),
        b0_raw_uncovered=(g=='unknown')&(meta.b0_max_iou.to_numpy()<.5),b0_missed_B_no_candidate=(g=='unknown')&(meta.b0_miss_type=='B_no_candidate').to_numpy())

def main():
    out=ROOT/'prepared';out.mkdir(exist_ok=True);complete=json.loads((R0/'paired_evaluation/complete.json').read_text());metas=[];chunks=[];offset=0
    if (out/'complete.json').exists():print('Preparation already complete');return
    for k,original in enumerate(complete['chunks']):
        folder=out/f'{k:03d}';folder.mkdir(exist_ok=True);commit=folder/'complete.json'
        if not commit.exists():
            path=Path(original['directory'])/'regions.csv.gz'
            if file_sha256(path)!=original['sha256']['regions.csv.gz']:raise RuntimeError('Original score chunk changed')
            df=pd.read_csv(path,dtype={'image_id':str},float_precision='round_trip')
            base=df.loc[(df.feature_source=='S1')&(df.level=='overall')&df.include_disjoint_gt].copy()
            columns=['image_id','source','candidate_id','origin','group_disjoint_gt','x1','y1','x2','y2','original_h','original_w','class_name','gt_index','b0_max_iou','batch1_max_iou','b0_miss_type']
            meta=base[columns].rename(columns={'group_disjoint_gt':'group'}).reset_index(drop=True)
            scores=np.full((len(meta),2,5),np.nan);valid=np.zeros((len(meta),5),bool);fraction=np.zeros((len(meta),5));small=np.zeros((len(meta),5),bool)
            keys=list(zip(meta.image_id,meta.source,meta.candidate_id))
            for si,source in enumerate(SOURCES):
                for li,level in enumerate(LEVELS):
                    part=df.loc[(df.feature_source==source)&(df.level==level)&df.include_disjoint_gt]
                    if list(zip(part.image_id,part.source,part.candidate_id))!=keys:raise RuntimeError('Original candidate identity order mismatch')
                    scores[:,si,li]=part.score.to_numpy();v=part.valid.to_numpy()
                    if si==0:valid[:,li]=v;fraction[:,li]=part.valid_fraction;small[:,li]=part.small_feature_roi
                    elif not np.array_equal(v,valid[:,li]):raise RuntimeError('S1/S2 validity mismatch')
            meta.to_csv(folder/'metadata.csv.gz',index=False);np.savez_compressed(folder/'original_scores.npz',scores=scores,valid=valid,valid_fraction=fraction,small=small)
            write_json(commit,dict(complete=True,image_ids=original['image_ids'],rows=len(meta),sha256={p:file_sha256(folder/p) for p in ['metadata.csv.gz','original_scores.npz']}))
        ck=manifest_check(commit);meta=pd.read_csv(folder/'metadata.csv.gz',dtype={'image_id':str},float_precision='round_trip');metas.append(meta)
        chunks.append(dict(index=k,directory=str(folder),offset=offset,**ck));offset+=len(meta)
        if k%25==0:print('prepared',k,'/',len(complete['chunks']),flush=True)
    meta=pd.concat(metas,ignore_index=True);meta.index.name='global_candidate';meta.to_csv(out/'metadata.csv.gz',index=False)
    pos_masks=masks(meta);matching={};files={};audit=[]
    for pool in ['gt_reference','query']:
        bg=np.flatnonzero((meta.source==pool)&(meta['group']=='background'))
        for name,positive_mask in pos_masks.items():
            if pool=='query' and name.startswith('b0_'):continue
            pos=np.flatnonzero((meta.source==pool).to_numpy()&positive_mask);pairs,unmatched=match_geometry(meta,pos,bg)
            key=pool+'__'+name;files[key+'__pairs']=pairs;files[key+'__unmatched']=unmatched
            audit.append(dict(pool=pool,comparison=name,positive_count=len(pos),background_available=len(bg),matched=len(pairs),unmatched=len(unmatched),positive_retention=len(pairs)/len(pos) if len(pos) else None,matching='no replacement; fixed joint caliper ln1.25 on area/relative-area/aspect'))
            print('matching',key,len(pairs),'/',len(pos),flush=True)
    np.savez_compressed(out/'geometry_matches.npz',**files);pd.DataFrame(audit).to_csv(out/'matching_retention.csv',index=False)
    write_json(out/'complete.json',dict(complete=True,unique_images=9119,rows=len(meta),chunks=chunks,metadata_sha256=file_sha256(out/'metadata.csv.gz'),matches_sha256=file_sha256(out/'geometry_matches.npz'),finished_at=stamp()))
if __name__=='__main__':main()
