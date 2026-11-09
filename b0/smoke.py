import argparse
import copy
import gc
import json
import time

import torch
from torch.utils.data import DataLoader

from b0.common import (ARTIFACTS, build, clone_outputs, datasets, json_write,
                       load_checkpoint, model_args, seed_all, to_device)
from b0.features import FeatureCapture
from util.misc import collate_fn


def subset(dataset, indices):
    result=copy.copy(dataset)
    for attr in ['images','annotations','imgids','image_set']:
        values=getattr(dataset,attr)
        setattr(result,attr,[values[i] for i in indices])
    return result


def equal(left,right):
    if torch.is_tensor(left):
        assert torch.equal(left,right), 'Feature hooks changed prediction tensors'
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):equal(a,b)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    options=parser.parse_args()
    args=model_args(options.config)
    start=time.time()
    from b0.train import run
    state,record=run(options.config,smoke_updates=2)
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    del state
    gc.collect()
    torch.cuda.empty_cache()
    seed_all(args.seed)
    model,criterion,postprocessors,_=build(args)
    ckpt=ARTIFACTS/f'checkpoints/smoke_b{args.batch_size}_a{args.accumulation_steps}/checkpoint.pth'
    checkpoint=load_checkpoint(model,ckpt)
    assert 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
    del checkpoint
    train_ds,val_ds=datasets(args)
    # Check training supervision on a deterministic sample and no cache mutation.
    for i in [0,len(train_ds)//2,len(train_ds)-1]:
        _,target=train_ds[i]
        assert (target['labels']<20).all()
    candidates=[]
    for i,stem in enumerate(val_ds.image_set):
        if len(candidates)<4 and len(stem)==6:
            candidates.append(i)
    coco=[i for i,stem in enumerate(val_ds.image_set) if len(stem)==12][:12]
    test=subset(val_ds,candidates+coco)
    loader=DataLoader(test,batch_size=args.batch_size,collate_fn=collate_fn,num_workers=0)
    model.eval()
    criterion.eval()
    shapes=None
    with torch.no_grad():
        for samples,targets in loader:
            samples,targets=to_device(samples,targets)
            before=model(samples)
            with FeatureCapture(model) as capture:
                after=model(samples)
                equal(before,after)
                if shapes is None:
                    shapes=capture.shapes()
            sizes=torch.stack([t['orig_size'] for t in targets])
            final_before=postprocessors['bbox'](clone_outputs(before),sizes)
            final_after=postprocessors['bbox'](clone_outputs(after),sizes)
            equal(final_before,final_after)
            assert after['pred_boxes'].shape[1:]==(100,4)
            assert after['pred_logits'].shape[1:]==(100,81)
            assert after['pred_obj'].shape[1]==100
    from engine import evaluate
    stats,_=evaluate(model,criterion,postprocessors,loader,test,torch.device('cuda'),str(ARTIFACTS/'evaluations/smoke'),args)
    # The matching/mAP values below are diagnostic only, never reported as B0 results.
    report={'passed':True,'type':'smoke_only_not_full_reproduction','optimizer_updates':2,
        'train_images':2*args.batch_size*args.accumulation_steps,'evaluation_images':len(test),
        'physical_batch_size':args.batch_size,'accumulation_steps':args.accumulation_steps,
        'checkpoint':str(ckpt),'checkpoint_strict_load':True,'feature_hooks_bitwise_equal':True,
        'feature_shapes':shapes,'training_record':record,'diagnostic_evaluation':stats,
        'seconds':time.time()-start,'peak_memory_mib':record['peak_memory_mib'],
        'kl_observation':'Official KL uses detached BN running statistics; no direct gradient. Preserved as released.'}
    json_write(ARTIFACTS/'reports/smoke.json',report)
    print('B0_SMOKE_PASSED '+json.dumps(report),flush=True)


if __name__=='__main__':
    main()
