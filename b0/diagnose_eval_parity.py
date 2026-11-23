"""Read-only staged diagnosis against the exact official evaluation arguments."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shlex
import numpy as np
import torch
from torch.utils.data import DataLoader
from b0.common import ARTIFACTS, REPO, model_args, build, load_checkpoint, clone_outputs, json_write
from b0.features import FeatureCapture
from main_open_world import get_args_parser,get_datasets
from util.misc import collate_fn


def official_args():
    text=(REPO/'b0/eval_task1.sh').read_text().replace('\\\n',' ')
    command=next(line for line in text.splitlines() if ' main_open_world.py ' in line)
    tokens=shlex.split(command);args=get_args_parser().parse_args(tokens[tokens.index('main_open_world.py')+1:])
    args.distributed=False;args.world_size=1;args.rank=0
    return args


def backend():
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}


def set_backend(profile):
    torch.backends.cuda.matmul.allow_tf32=profile['matmul_tf32']
    torch.backends.cudnn.allow_tf32=profile['cudnn_tf32']
    torch.backends.cudnn.benchmark=profile['cudnn_benchmark']
    torch.backends.cudnn.deterministic=profile['cudnn_deterministic']
    torch.use_deterministic_algorithms(profile['deterministic_algorithms'])


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def difference(a,b):
    if isinstance(a,torch.Tensor):
        return {'equal':torch.equal(a,b),'shape':list(a.shape),
                'max_abs':float((a.double()-b.double()).abs().max()) if a.numel() else 0.}
    if isinstance(a,dict):return {k:difference(a[k],b[k]) for k in a}
    return [difference(x,y) for x,y in zip(a,b)]


def run(output,batches=2):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    official=official_args();analysis=model_args()
    defaults=backend()
    torch.manual_seed(official.seed);np.random.seed(official.seed);random.seed(official.seed);torch.set_num_threads(8)
    model,criterion,posts,_=build(official);saved=load_checkpoint(model,official.pretrain);del saved
    model.eval();criterion.eval()
    stages={};handles=[]
    for name,module in model.named_modules():
        if name.startswith('backbone.') and isinstance(module,torch.nn.Conv2d) and len(handles)<8:
            def capture(module,inputs,output,name=name):stages[name]=output[0,:4,:32,:32].detach().cpu().clone()
            handles.append(module.register_forward_hook(capture))
    _,dataset=get_datasets(official);_,analysis_dataset=get_datasets(analysis)
    ids_equal=dataset.image_set==analysis_dataset.image_set
    assert ids_equal
    loader=DataLoader(dataset,batch_size=official.batch_size,shuffle=False,collate_fn=collate_fn,
                      num_workers=official.num_workers,pin_memory=True)
    old=dict(defaults,matmul_tf32=False,cudnn_tf32=False)
    report={'official_backend':defaults,'original_analysis_backend':old,
            'args_differences':{k:{'official':getattr(official,k,None),'analysis':getattr(analysis,k,None)}
                for k in sorted(set(vars(official))|set(vars(analysis))) if getattr(official,k,None)!=getattr(analysis,k,None)},
            'same_split_order_and_duplicates':ids_equal,'rows':len(dataset),'batch_rows':[]}
    with torch.no_grad():
        for batch_index,(samples,targets) in enumerate(loader):
            if batch_index>=batches:break
            first_index=batch_index*official.batch_size
            comparison=[analysis_dataset[first_index+i] for i in range(len(targets))]
            other_samples,other_targets=collate_fn(comparison)
            assert torch.equal(samples.tensors,other_samples.tensors) and torch.equal(samples.mask,other_samples.mask)
            samples=samples.to('cuda');sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
            set_backend(defaults);reference=model(samples);reference_stages=dict(stages)
            final_reference=posts['bbox'](clone_outputs(reference),sizes)
            set_backend(old);legacy=model(samples);legacy_stages=dict(stages)
            legacy_final=posts['bbox'](clone_outputs(legacy),sizes)
            set_backend(defaults)
            with FeatureCapture(model):fixed=model(samples)
            fixed_final=posts['bbox'](clone_outputs(fixed),sizes)
            ids=[dataset.convert_image_id(int(t['image_id']),to_string=True) for t in targets]
            row={'batch_index':batch_index,'image_ids':ids,'padded_hw':list(samples.tensors.shape[-2:]),
                 'preprocessing_bitwise_equal':True,'input_sha256':digest(samples.tensors),
                 'backbone_stage_samples':difference(reference_stages,legacy_stages),'legacy_outputs':difference(reference,legacy),'legacy_final':difference(final_reference,legacy_final),
                 'aligned_hook_outputs':difference(reference,fixed),'aligned_hook_final':difference(final_reference,fixed_final)}
            report['batch_rows'].append(row)
            for slot,stem in enumerate(ids):
                np.savez_compressed(out/f'batch{batch_index}_{stem}.npz',
                    **{f'{prefix}_{key}':result[slot][key].cpu().numpy()
                       for prefix,result in [('official',final_reference),('legacy',legacy_final),('aligned',fixed_final)]
                       for key in ['scores','labels','boxes']})
            print('DIAGNOSTIC_BATCH',batch_index,ids,'legacy',row['legacy_final'][0],'aligned',row['aligned_hook_final'][0],flush=True)
    for handle in handles:handle.remove()
    set_backend(defaults);json_write(out/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--batches',type=int,default=2)
    options=parser.parse_args();run(options.output,options.batches)
