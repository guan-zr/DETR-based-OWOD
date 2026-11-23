"""Structural preflight and small real-CUDA validation; never selects a model by test scores."""
import copy,csv,json,math,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou
from r0.acceptance import backend_state,verify_artifacts
from r0.data import ImageOnlyDataset,read_eval_gt
from r0.evaluate import ensemble_errors,candidate_pools,region_records,image_seed
from r0.features import FrozenSpatialExtractor
from r0.reconstruction import FMRBank
from r0.regions import mean_region_error
from r0.runtime import read_config,load_detector,pack_images,seed_all,write_json,file_sha256,StepBatches
from r0.validate_detector import compare

OUT=Path('/root/autodl-tmp/experiments/R0/formal_20261111')

def main():
    config=read_config('r0/configs/R0-S1.json'); seed_all(73)
    initial_backend=backend_state(); acceptance=verify_artifacts(config)
    assert all(json.loads(l)['complete'] for l in (Path(config['baseline_checkpoint']).parent/'epochs.jsonl').read_text().splitlines())
    root=Path(config['data_root']); split=root/'ImageSets/TOWOD'
    train=(split/'owod_t1_train.txt').read_text().splitlines(); test=(split/'owod_all_task_test.txt').read_text().splitlines()
    ids=list(dict.fromkeys(test)); assert len(ids)==9119 and not set(train)&set(ids)
    c2=read_config('r0/configs/R0-S2.json'); differences={k:[config[k],c2[k]] for k in config if config[k]!=c2[k]}
    assert set(differences)=={'source','name','output'}
    # Full index/class/coordinate audit before any formal GPU run.
    import ast,xml.etree.ElementTree as ET
    from r0.data import ALIASES
    tree=ast.parse(Path('/root/autodl-tmp/OWOBJ-main/b0/analyze.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ground_truth')
    namespace={'np':np,'ET':ET,'ALIASES':ALIASES}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'B0_ground_truth','exec'),namespace)
    ground_truth=namespace['ground_truth']
    by_id={}; rows=list(csv.DictReader(open(Path(config['b0_analysis_summary']).parent/'per_gt.csv')))
    for r in rows:
        key=(r['image_id'],int(r['gt_index']));assert key not in by_id;by_id[key]=r
    sample={}; metadata=[]; matched=0
    for stem in ids:
        gt=read_eval_gt(root,stem); other=ground_truth(root/'Annotations'/f'{stem}.xml',list(__import__('r0.data',fromlist=['KNOWN']).KNOWN))
        np.testing.assert_array_equal(gt['boxes'].numpy(),other['boxes'])
        assert gt['names']==other['names'] and np.array_equal(gt['known'],other['known'])
        import xml.etree.ElementTree as ET
        x=ET.parse(root/'Annotations'/f'{stem}.xml').getroot();h=int(x.findtext('size/height'));w=int(x.findtext('size/width'))
        for j,box in enumerate(gt['boxes']):
            if gt['difficult'][j]:continue
            r=by_id[(stem,j)]; assert r['group']==('known' if gt['known'][j] else 'unknown') and r['class_name']==gt['names'][j]
            matched+=1
            area=float((box[2]-box[0])*(box[3]-box[1]));aspect=float((box[2]-box[0])/(box[3]-box[1])); rel=area/(h*w)
            metadata.append((stem,j,area,rel,aspect))
            sample.setdefault(('coco' if len(stem)==12 else 'voc')+'_'+r['group'],stem)
    assert matched==len(rows)==45131
    # Fixed structural extremes, independent of reconstruction scores.
    selected=list(dict.fromkeys(sample.values()))
    for key,rev in [(2,False),(3,True),(4,False),(4,True)]:
        selected.append(sorted(metadata,key=lambda r:r[key],reverse=rev)[0][0])
    selected=list(dict.fromkeys(selected))
    sequence=[train[b[0]] for b in StepBatches(len(train),1,0,2000,73)]
    write_json(OUT/'preflight.json',dict(acceptance=acceptance,matched_per_gt=matched,known=29300,unknown=15831,
        train_rows=len(train),train_unique=len(set(train)),eval_rows=len(test),eval_unique=len(ids),split_overlap=0,
        config_differences=differences,samples=selected,train_sequence_sha256=__import__('hashlib').sha256(('\n'.join(sequence)+'\n').encode()).hexdigest(),
        initial_backend=initial_backend,coordinate_check='R0 boxes exactly equal B0 analysis_utils.ground_truth for all XML'))
    (OUT/'train_image_sequence.txt').write_text('\n'.join(sequence)+'\n')
    extractor,post,baseline=load_detector(config,torch.device('cuda'));detector=extractor.detector
    dataset=ImageOnlyDataset(root,config['eval_split'],unique=True,**config['resize']);idx={s:i for i,s in enumerate(dataset.ids)}
    bank=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=.6).cuda().eval()
    results=[];torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for stem in selected:
            item=dataset[idx[stem]];samples,geometry=pack_images([item],torch.device('cuda'));g=geometry[0]
            torch.cuda.synchronize();t=time.perf_counter();before=detector(samples);torch.cuda.synchronize();elapsed=time.perf_counter()-t
            final_before=post['bbox'](copy.deepcopy(before),torch.tensor([g.original_hw],device='cuda'))
            gt=read_eval_gt(root,stem);h,w=g.original_hw
            from util.box_ops import box_cxcywh_to_xyxy
            query=(box_cxcywh_to_xyxy(before['pred_boxes'][0])*before['pred_boxes'].new_tensor([w,h,w,h])).cpu()
            pools=candidate_pools(gt,query,g,image_seed(stem,73),config['evaluation'])
            summary=dict(image_id=stem,original_hw=g.original_hw,resized_hw=g.resized_hw,detector_seconds=elapsed,sources={})
            for source in ['S1','S2']:
                reader=FrozenSpatialExtractor(detector,source);batch,after=reader(samples,geometry);compare(before,after)
                compare(final_before,post['bbox'](copy.deepcopy(after),torch.tensor([g.original_hw],device='cuda')))
                repeated,repeated_outputs=reader(samples,geometry);compare(after,repeated_outputs)
                for a,b in zip(batch.levels,repeated.levels):assert torch.equal(a,b)
                shapes=[list(f.shape) for f in batch.levels]
                for f,mask,stride in zip(batch.levels,batch.padding_masks,batch.strides):
                    assert f.shape[-2:]==(math.ceil(g.resized_hw[0]/stride),math.ceil(g.resized_hw[1]/stride))
                    assert not mask.any() and torch.isfinite(f).all() and not f.requires_grad
                errors=ensemble_errors(bank,batch,.6,4,image_seed(stem,73));repeat_errors=ensemble_errors(bank,repeated,.6,4,image_seed(stem,73))
                for a,b in zip(errors,repeat_errors):assert torch.equal(a['error'],b['error'])
                records=region_records(batch,errors,pools,config['roi'])
                # Real-device analytic geometry checks on actual shapes and padding support.
                probes=[[0,0,w,h],[0,0,1,1],[w-1,h-1,w,h],[0,0,w,1],[0,0,1,h],[-2,-2,3,3],[0,0,0,1],[float('nan'),0,1,1]]
                probe_results=[]
                for maps,stride in zip(errors,batch.strides):
                    support=maps['valid'][0];constant=torch.full_like(maps['error'][0],3.25)
                    check=mean_region_error(constant,support,probes,g,stride,**config['roi'])
                    assert not check.valid[-1] and not check.valid[-2]
                    assert torch.allclose(check.scores[check.valid],torch.full_like(check.scores[check.valid],3.25),atol=1e-5,rtol=1e-5)
                    assert torch.isfinite(check.valid_fraction).all()
                    probe_results.append(dict(valid=check.valid.tolist(),valid_fraction=check.valid_fraction.tolist(),small=check.small_feature_roi.tolist()))
                # A fresh one-step bank is diagnostic only and is never reused for training.
                with torch.enable_grad():
                    fresh=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=.6).cuda()
                    optimizer=torch.optim.AdamW(fresh.parameters(),**config['optimizer']);initial=next(fresh.parameters()).detach().clone()
                    loss,_=fresh(batch,seed=73);loss.backward();optimizer.step()
                    assert not torch.equal(initial,next(fresh.parameters())) and torch.isfinite(loss)
                    assert all(p.grad is None and not p.requires_grad for p in detector.parameters())
                del fresh,optimizer,loss
                summary['sources'][source]=dict(shapes=shapes,strides=batch.strides,valid_positions=[int((~m).sum()) for m in batch.padding_masks],
                    feature_bytes=sum(f.numel()*f.element_size() for f in batch.levels),raw_hook_bitwise_equal=True,post_hook_bitwise_equal=True,
                    features_repeat_bitwise_equal=True,error_repeat_bitwise_equal=True,roi_count=len(records),valid_roi_count=sum(r['valid'] for r in records),probes=probe_results)
            results.append(summary);write_json(OUT/'gpu_validation_progress.json',results);print('validated',stem,flush=True)
    import torchvision
    environment=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),backend=backend_state(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write_json(OUT/'gpu_validation.json',dict(passed=True,baseline=baseline,environment=environment,images=results,scope='small real-image preflight; fresh AE only; no scientific separability claim'))

if __name__=='__main__':main()
