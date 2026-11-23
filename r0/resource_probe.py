"""Real paired-extraction equivalence, padding and streaming cost checks, before training."""
import copy,csv,gzip,json,time,shutil
from pathlib import Path
import torch
from r0.acceptance import backend_state
from r0.data import ImageOnlyDataset,read_eval_gt
from r0.features import FrozenSpatialExtractor
from r0.reconstruction import FMRBank
from r0.regions import mean_region_error
from r0.evaluate import candidate_pools,image_seed,ensemble_errors,region_records as original_records
from r0.streaming import extract_both,region_records
from r0.runtime import load_detector,pack_images,read_config,write_json,seed_all
from r0.validate_detector import compare
OUT=Path('/root/autodl-tmp/experiments/R0/formal_20261111')

def main():
    config=read_config('r0/configs/R0-S2.json');seed_all(73)
    extractor,_,baseline=load_detector(config,torch.device('cuda'));data=ImageOnlyDataset(config['data_root'],config['eval_split'],unique=True,**config['resize'])
    lookup={s:i for i,s in enumerate(data.ids)};ids=json.loads((OUT/'preflight.json').read_text())['samples']
    bank=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=.6).cuda().eval();writer=None;rows=0;costs=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),gzip.open(OUT/'probe_regions.csv.gz','wt',newline='') as f:
        for stem in ids:
            samples,geometry=pack_images([data[lookup[stem]]],torch.device('cuda'));torch.cuda.synchronize();started=time.perf_counter()
            batches,outputs=extract_both(extractor,samples,geometry)
            reference,reference_outputs=FrozenSpatialExtractor(extractor.detector,'S1')(samples,geometry);compare(outputs,reference_outputs)
            assert all(torch.equal(a,b) for a,b in zip(reference.levels,batches['S1'].levels))
            from util.box_ops import box_cxcywh_to_xyxy
            h,w=geometry[0].original_hw;queries=(box_cxcywh_to_xyxy(outputs['pred_boxes'][0])*outputs['pred_boxes'].new_tensor([w,h,w,h])).cpu()
            gt=read_eval_gt(config['data_root'],stem);pools=candidate_pools(gt,queries,geometry[0],image_seed(stem,73),dict(config['evaluation'],background_rule='iou_only'))
            for source,batch in batches.items():
                errors=ensemble_errors(bank,batch,.6,4,image_seed(stem,73))
                records=region_records(batch,errors,pools,config['roi']);old=original_records(batch,errors,pools,config['roi'])
                assert records==old,'vector transfer changed ROI records'
                for r in records:
                    r['feature_source']=source
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
                    writer.writerow(r);rows+=1
            torch.cuda.synchronize();costs.append(time.perf_counter()-started)
    # Actual padded image context, with no equality requirement across batch contexts.
    items=[data[lookup[ids[0]]],data[lookup[ids[1]]]]
    samples,geometry=pack_images(items,torch.device('cuda'));padded,_=extractor(samples,geometry)
    masks=[int(m.sum()) for m in padded.padding_masks];assert sum(masks)>0
    padding_checks=[]
    for maps,mask,stride in zip(padded.levels,padded.padding_masks,padded.strides):
        support=~mask[0];error=torch.full_like(maps[0,0],3.25).masked_fill(mask[0],1e9)
        h,w=geometry[0].original_hw
        check=mean_region_error(error,support,[[0,0,w,h],[w-2,h-2,w,h]],geometry[0],stride,**config['roi'])
        assert check.valid.all() and torch.allclose(check.scores,torch.full_like(check.scores,3.25),atol=1e-5)
        padding_checks.append(dict(masked_positions=int(mask.sum()),valid_fraction=check.valid_fraction.tolist(),scores=check.scores.tolist()))
    environment=dict(backend=backend_state(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    bytes_out=(OUT/'probe_regions.csv.gz').stat().st_size
    free=shutil.disk_usage(OUT).free
    assert free>10*2**30
    write_json(OUT/'resource_probe.json',dict(passed=True,sample_images=ids,paired_s1_hook_bitwise_equal=True,optimized_records_exactly_equal=True,
        seconds_per_image_including_reference_checks=costs,rows=rows,compressed_bytes=bytes_out,projected_9119_bytes=bytes_out/len(ids)*9119,
        available_bytes=free,cache_strategy='stream both sources; no full spatial feature cache',environment=environment,padding_checks=padding_checks,
        note='probe uses fresh random AE and is not a research result; time includes reference forwards and duplicate ROI checks'))
    print(json.dumps(dict(rows=rows,bytes=bytes_out,seconds_per_image=costs,environment=environment)),flush=True)
if __name__=='__main__':main()
