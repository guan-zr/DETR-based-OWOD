"""Full paired S1/S2 evaluation with durable chunks, both fixed grouping rules and no feature cache."""
import argparse,csv,gzip,json,time,shutil
from collections import Counter
from pathlib import Path
import torch
from torchvision.ops import box_iou
from r0.acceptance import backend_state
from r0.data import ImageOnlyDataset,read_eval_gt
from r0.evaluate import candidate_pools,ensemble_errors,image_seed
from r0.labels import label_regions,GroupingAudit
from r0.reconstruction import FMRBank
from r0.runtime import load_detector,read_config,pack_images,seed_all,write_json,file_sha256,reconstruction_config_matches
from r0.streaming import extract_both,region_records
from util.box_ops import box_cxcywh_to_xyxy

ROOT=Path('/root/autodl-tmp/experiments/R0'); EVIDENCE=ROOT/'formal_20261111'

def source_manifest():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):file_sha256(p) for folder in ['r0','models','util','datasets'] for p in (root/folder).rglob('*.py') if '__pycache__' not in p.parts}

def run(output=None,limit=None,chunk_size=32,s1_checkpoint=None,s2_checkpoint=None,diagnostic_fresh=False):
    if diagnostic_fresh and (not output or not limit or limit<=0):raise RuntimeError('Fresh diagnostic AE requires explicit limited output; cannot be full')
    out=Path(output or ROOT/'paired_evaluation');out.mkdir(parents=True,exist_ok=True)
    configs={s:read_config(f'r0/configs/R0-{s}.json') for s in ('S1','S2')}
    config=configs['S2'];seed_all(73)
    checkpoint_paths={s:Path(c['output'])/'checkpoint.pth' for s,c in configs.items()}
    if s1_checkpoint:checkpoint_paths['S1']=Path(s1_checkpoint)
    if s2_checkpoint:checkpoint_paths['S2']=Path(s2_checkpoint)
    provenance=dict(configs=configs,checkpoints={} if diagnostic_fresh else {s:dict(path=str(p),sha256=file_sha256(p)) for s,p in checkpoint_paths.items()},diagnostic_fresh_ae=diagnostic_fresh,source_sha256=source_manifest(),
        split_sha256=file_sha256(Path(config['data_root'])/'ImageSets/TOWOD'/f"{config['eval_split']}.txt"),limit=limit,chunk_size=chunk_size,
        scope='full_9119_unique' if limit is None else 'limited_smoke',higher_error_is_foreground=True,
        grouping_rules=['disjoint_gt','iou_only'],pool_semantics='GT-reference and all 100 raw final queries remain separate',
        overall_definition='unweighted raw mean of all four scale MSE values, requiring all valid',
        coverage_definition='raw unbounded query IoU for batch1 and B0 original per_gt; labeling/ROI use clipped boxes')
    if (out/'run.json').exists():
        if json.loads((out/'run.json').read_text())!=provenance:raise RuntimeError('Evaluation provenance changed; refuse resume')
    else:write_json(out/'run.json',provenance)
    if (out/'complete.json').exists():
        committed=json.loads((out/'complete.json').read_text())
        if committed.get('scope')!=provenance['scope'] or not committed.get('complete'):raise RuntimeError('Completed evaluation scope mismatch')
        seen=[]
        for chunk in committed['chunks']:
            seen+=chunk['image_ids']
            for filename,digest in chunk['sha256'].items():
                if file_sha256(Path(chunk['directory'])/filename)!=digest:raise RuntimeError('Damaged completed evaluation')
        if seen!=committed['image_ids'] or len(set(seen))!=committed['unique_images']:raise RuntimeError('Incomplete completed image set')
        print('Verified already-complete evaluation; preserved all outputs',flush=True);return
    extractor,_,baseline=load_detector(config,torch.device('cuda'));banks={}
    for source,p in checkpoint_paths.items():
        banks[source]=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=config['mask_ratio']).cuda().eval()
        if not diagnostic_fresh:
            saved=torch.load(p,map_location='cpu')
            if not saved.get('complete') or saved['step']!=2000 or not reconstruction_config_matches(saved['config'],configs[source]) or saved['baseline']['sha256']!=baseline['sha256']:
                raise RuntimeError('Need matching complete 2000-step bank: '+source)
            banks[source].load_state_dict(saved['bank'],strict=True)
            del saved
    data=ImageOnlyDataset(config['data_root'],config['eval_split'],unique=True,**config['resize']);ids=data.ids[:limit] if limit else data.ids
    if limit is None and len(ids)!=9119:raise RuntimeError('Full evaluation must have 9119 unique images')
    # This table is read only after reconstruction training has completed.
    b0_rows=list(csv.DictReader(open(Path(config['b0_analysis_summary']).parent/'per_gt.csv')))
    b0={(r['image_id'],int(r['gt_index'])):r for r in b0_rows}
    manifests=[];completed_ids=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    for begin in range(0,len(ids),chunk_size):
        current_ids=ids[begin:begin+chunk_size];parent=out/'chunks'/f'{begin:05d}';parent.mkdir(parents=True,exist_ok=True)
        committed=sorted(parent.glob('attempt_*/complete.json'))
        if len(committed)>1:raise RuntimeError('Duplicate committed chunk')
        if committed:
            m=json.loads(committed[0].read_text());attempt=committed[0].parent
            if m['image_ids']!=current_ids:raise RuntimeError('Chunk image IDs mismatch')
            for filename,digest in m['sha256'].items():
                if file_sha256(attempt/filename)!=digest:raise RuntimeError('Damaged completed chunk')
            manifests.append(dict(directory=str(attempt),**m));completed_ids+=current_ids
            print(f'Reuse completed chunk {begin} ({len(completed_ids)}/{len(ids)})',flush=True);continue
        attempt=parent/f'attempt_{len(list(parent.glob("attempt_*"))):02d}';attempt.mkdir(exist_ok=False)
        audit=GroupingAudit();writer=None;row_count=0;chunk_started=time.time();image_records=[]
        with torch.no_grad(),gzip.open(attempt/'regions.csv.gz','wt',newline='',compresslevel=1) as output_file:
            for index in range(begin,begin+len(current_ids)):
                item=data[index];samples,geometry=pack_images([item],torch.device('cuda'));g=geometry[0];stem=g.image_id
                batches,outputs=extract_both(extractor,samples,geometry)
                if outputs['pred_boxes'].shape!=(1,100,4):raise RuntimeError('Expected exactly 100 raw queries')
                h,w=g.original_hw;query=(box_cxcywh_to_xyxy(outputs['pred_boxes'][0])*outputs['pred_boxes'].new_tensor([w,h,w,h])).cpu()
                gt=read_eval_gt(config['data_root'],stem);settings=dict(config['evaluation'],background_rule='iou_only')
                pools=candidate_pools(gt,query,g,image_seed(stem,73),settings,audit=audit)
                gt_clipped=dict(gt,boxes=pools['gt_reference']['boxes'][:len(gt['boxes'])])
                disjoint={pool:label_regions(p['boxes'],gt_clipped,settings['positive_iou'],settings['background_iou'],'disjoint_gt')[0] for pool,p in pools.items()}
                # Annotation identity is retained even for coincident or overlapping GT.
                disjoint['gt_reference'][:len(gt['boxes'])]=pools['gt_reference']['labels'][:len(gt['boxes'])]
                raw_max=box_iou(query,gt['boxes']).max(0).values if len(gt['boxes']) else torch.empty(0)
                clipped_max=box_iou(pools['query']['boxes'],gt_clipped['boxes']).max(0).values if len(gt['boxes']) else torch.empty(0)
                for source,batch in batches.items():
                    errors=ensemble_errors(banks[source],batch,settings['mask_ratio'],settings['repeats'],image_seed(stem,73))
                    records=region_records(batch,errors,pools,config['roi'])
                    for record in records:
                        pool=record['source'];candidate=record['candidate_id'];annotated=record['origin']=='annotated_gt'
                        group_new=disjoint[pool][candidate]
                        record.update(feature_source=source,group_iou_only=record['group'],group_disjoint_gt=group_new,
                            include_disjoint_gt=pool=='query' or annotated or group_new=='background',original_h=h,original_w=w,
                            class_name=gt['names'][candidate] if annotated else '',gt_index=candidate if annotated else '',
                            b0_max_iou='',batch1_max_iou='',batch1_clipped_max_iou='',b0_miss_type='',b0_official_hit='')
                        if annotated and not gt['difficult'][candidate]:
                            row=b0[(stem,candidate)]
                            if row['class_name']!=record['class_name'] or row['group']!=record['group']:raise RuntimeError('per_gt identity mismatch')
                            record.update(b0_max_iou=row['max_iou'],batch1_max_iou=float(raw_max[candidate]),batch1_clipped_max_iou=float(clipped_max[candidate]),b0_miss_type=row['miss_type'],b0_official_hit=row['official_unknown_hit'])
                        if writer is None:writer=csv.DictWriter(output_file,fieldnames=list(record));writer.writeheader()
                        writer.writerow(record);row_count+=1
                image_records.append(dict(image_id=stem,original_hw=g.original_hw,resized_hw=g.resized_hw,padded_hw=g.padded_hw,
                    shapes={s:[list(f.shape) for f in b.levels] for s,b in batches.items()},query_count=100,
                    regions_per_pool={p:len(v['boxes']) for p,v in pools.items()},available_disk_bytes=shutil.disk_usage(out).free))
                if any(p.grad is not None or p.requires_grad for p in extractor.detector.parameters()):raise RuntimeError('Detector gradient was enabled')
        (attempt/'images.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in image_records))
        write_json(attempt/'grouping_audit.json',audit.summarize())
        m=dict(complete=True,image_ids=current_ids,rows=row_count,seconds=time.time()-chunk_started,
            sha256={name:file_sha256(attempt/name) for name in ['regions.csv.gz','images.jsonl','grouping_audit.json']})
        write_json(attempt/'complete.json',m);manifests.append(dict(directory=str(attempt),**m));completed_ids+=current_ids
        write_json(out/'progress.json',dict(completed_images=len(completed_ids),total_images=len(ids),seconds_this_invocation=time.time()-started,
            last_chunk=str(attempt),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,backend=backend_state()))
        print(f'Committed evaluation {len(completed_ids)}/{len(ids)}; {row_count} rows; chunk {m["seconds"]:.1f}s',flush=True)
    if completed_ids!=ids or len(set(completed_ids))!=len(ids):raise RuntimeError('Missing or duplicate evaluation images')
    write_json(out/'complete.json',dict(complete=True,scope=provenance['scope'],unique_images=len(ids),image_ids=ids,chunks=manifests,
        rows=sum(m['rows'] for m in manifests),baseline=baseline,backend=backend_state(),seconds_this_invocation=time.time()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--limit',type=int);p.add_argument('--chunk-size',type=int,default=32);p.add_argument('--diagnostic-fresh',action='store_true');p.add_argument('--s1-checkpoint');p.add_argument('--s2-checkpoint');a=p.parse_args();run(a.output,a.limit,a.chunk_size,a.s1_checkpoint,a.s2_checkpoint,a.diagnostic_fresh)
