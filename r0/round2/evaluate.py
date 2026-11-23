"""Same-forward batch1 controls, durable scalar chunks, exact round-one candidate identity."""
import json,time
import numpy as np
import pandas as pd
import torch
from r0.round2.common import *
from r0.round2.train import make_bank
from r0.round2.scoring import score_batch
from r0.runtime import load_detector,pack_images,seed_all
from r0.data import ImageOnlyDataset
from r0.streaming import extract_both
from util.box_ops import box_cxcywh_to_xyxy

def evaluate(output=None,limit_chunks=None):
    out=Path(output or ROOT/'evaluation');out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('Evaluation already complete');return
    prep=json.loads((ROOT/'prepared/complete.json').read_text());manifest_check(ROOT/'training/complete.json')
    c=config();seed_all(73);extractor,_,baseline=load_detector(c,torch.device('cuda'))
    banks={s:{f'trained{k}':make_bank(k,s,True).eval() for k in SEEDS} | {f'random{k}':make_bank(k).eval() for k in SEEDS} for s in SOURCES}
    means_np=np.load(ROOT/'training/training_means.npz');means={s:torch.as_tensor(means_np[s],device='cuda') for s in SOURCES}
    provenance=dict(protocol_sha256=file_sha256(ROOT/'protocol.json'),prepared_sha256=file_sha256(ROOT/'prepared/complete.json'),source_sha256=source_hashes(),baseline=baseline,
        checkpoint_sha256={s:{str(k):file_sha256(train_path(s,k)) for k in SEEDS} for s in SOURCES},initialization_sha256={str(k):file_sha256(init_path(k)) for k in SEEDS},mean_sha256=file_sha256(ROOT/'training/training_means.npz'),limit_chunks=limit_chunks)
    if (out/'run.json').exists() and json.loads((out/'run.json').read_text())!=provenance:raise RuntimeError('Round2 evaluation provenance changed')
    write_json(out/'run.json',provenance)
    data=ImageOnlyDataset(c['data_root'],c['eval_split'],unique=True,**c['resize']);lookup={s:i for i,s in enumerate(data.ids)}
    chunks=[];ids=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    for original in prep['chunks'][:limit_chunks]:
        i=original['index'];parent=out/f'{i:03d}';parent.mkdir(exist_ok=True);commits=sorted(parent.glob('attempt_*/complete.json'))
        if len(commits)>1:raise RuntimeError('Duplicate score chunk commits')
        if commits:
            saved=manifest_check(commits[0]);chunks.append(dict(directory=str(commits[0].parent),**saved));ids+=saved['image_ids'];continue
        attempt=parent/f'attempt_{len(list(parent.glob("attempt_*"))):02d}';attempt.mkdir()
        meta=pd.read_csv(Path(original['directory'])/'metadata.csv.gz',dtype={'image_id':str},float_precision='round_trip');old=np.load(Path(original['directory'])/'original_scores.npz');n=len(meta)
        scores=np.full((n,2,len(METHODS),5),np.nan);query_scores=np.full((n,2),np.nan);valid=np.zeros((n,5),bool);raw_boxes=[];raw_query_rows=[];drift=0.;started_chunk=time.time()
        with torch.no_grad():
            for stem in original['image_ids']:
                selection=np.flatnonzero((meta.image_id==stem).to_numpy());frame=meta.iloc[selection]
                samples,g=pack_images([data[lookup[stem]]],torch.device('cuda'));batches,outputs=extract_both(extractor,samples,g)
                h,w=g[0].original_hw
                if not ((frame.original_h==h).all() and (frame.original_w==w).all()):raise RuntimeError('Image geometry identity mismatch')
                raw=box_cxcywh_to_xyxy(outputs['pred_boxes'][0])*outputs['pred_boxes'].new_tensor([w,h,w,h]);clipped=raw.detach().cpu().clone();clipped[:,0::2].clamp_(0,w);clipped[:,1::2].clamp_(0,h)
                q=frame.loc[frame.source=='query'];qindices=selection[np.flatnonzero((frame.source=='query').to_numpy())]
                if q.candidate_id.tolist()!=list(range(100)) or not torch.equal(clipped,torch.tensor(q[['x1','y1','x2','y2']].to_numpy(),dtype=torch.float32)):raise RuntimeError('Same-query batch1 boxes differ from round one')
                boxes=torch.tensor(frame[['x1','y1','x2','y2']].to_numpy(),dtype=torch.float32)
                values,v=score_batch(batches,banks,means,boxes);scores[selection]=values;valid[selection]=v
                old_scores=old['scores'][selection];delta=np.nanmax(np.abs(values[:,:,0,:]-old_scores));drift=max(drift,float(delta))
                if not np.allclose(values[:,:,0,:],old_scores,rtol=1e-5,atol=1e-6,equal_nan=True) or not np.array_equal(v,old['valid'][selection]):raise RuntimeError('Round-one seed73 scoring/validity mismatch')
                gaussian=torch.exp(-(1.3/256)*outputs['pred_obj'][0]);unknown=outputs['pred_logits'][0,:,-1].sigmoid()
                query_scores[qindices]=torch.stack([gaussian,unknown],1).cpu().numpy().astype(np.float64)
                raw_boxes.append(raw.cpu().numpy());raw_query_rows.extend(qindices.tolist())
        np.savez_compressed(attempt/'scores.npz',scores=scores,valid=valid,query_scores=query_scores,raw_query_boxes=np.concatenate(raw_boxes),raw_query_rows=np.asarray(raw_query_rows))
        record=dict(complete=True,index=i,image_ids=original['image_ids'],rows=n,seconds=time.time()-started_chunk,seed73_max_abs_drift=drift,
            finite_by_method={m:int(np.isfinite(scores[:,:,j,:]).sum()) for j,m in enumerate(METHODS)},same_forward_query_identity=True,sha256={'scores.npz':file_sha256(attempt/'scores.npz')})
        write_json(attempt/'complete.json',record);chunks.append(dict(directory=str(attempt),**record));ids+=record['image_ids']
        write_json(out/'progress.json',dict(completed_images=len(ids),total_images=9119,completed_chunks=len(chunks),seconds_this_invocation=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        print('controls evaluation',len(ids),'/9119','chunk_seconds',record['seconds'],'seed73_drift',drift,flush=True)
    expected=data.ids if limit_chunks is None else [x for chunk in prep['chunks'][:limit_chunks] for x in chunk['image_ids']]
    if ids!=expected or len(ids)!=len(set(ids)):raise RuntimeError('Missing or duplicate evaluation images')
    write_json(out/'complete.json',dict(complete=True,scope='full_9119_unique' if limit_chunks is None else 'limited_smoke',unique_images=len(ids),image_ids=ids,chunks=chunks,methods=METHODS,levels=LEVELS,
        seconds_this_invocation=time.time()-started,seed73_max_abs_drift=max(c['seed73_max_abs_drift'] for c in chunks),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
if __name__=='__main__':evaluate()
