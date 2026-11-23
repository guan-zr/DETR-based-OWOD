"""Future fixed-budget image-only FMR training. No eval/unknown labels imported."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from r0.data import ImageOnlyDataset,image_collate
from r0.reconstruction import FMRBank
from r0.runtime import (StepBatches,git_provenance,guard_b0_complete,load_detector,
                        pack_images,read_config,seed_all,write_json)


def run(config_path,output=None,resume=None,device='cuda'):
    config=read_config(config_path)
    guard_b0_complete(config)
    seed_all(config['seed'])
    out=Path(output or config['output'])
    if out.exists() and any(out.iterdir()) and not resume:
        raise RuntimeError('Refuse to overwrite existing R0 run; use a new output or --resume')
    out.mkdir(parents=True,exist_ok=True)
    extractor,_,baseline=load_detector(config,torch.device(device))
    bank=FMRBank(config['channels'],**config['autoencoder'],mask_ratio=config['mask_ratio']).to(device)
    optimizer=torch.optim.AdamW(bank.parameters(),**config['optimizer'])
    start=0
    if resume:
        saved=torch.load(resume,map_location=device)
        if saved['config'] != config or saved['baseline']['sha256'] != baseline['sha256']:
            raise RuntimeError('Resume config or frozen detector checksum differs')
        bank.load_state_dict(saved['bank'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step']
    dataset=ImageOnlyDataset(config['data_root'],config['train_split'],**config['resize'])
    sampler=StepBatches(len(dataset),config['batch_size'],start,config['steps'],config['seed'])
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=config['num_workers'],collate_fn=image_collate)
    write_json(out/'run.json',{'config':config,'baseline':baseline,'code':git_provenance(),
                             'train_annotations_loaded':False,'source':config['source']})
    started=time.time()
    bank.train()
    for step,items in enumerate(loader,start=start):
        samples,geometry=pack_images(items,torch.device(device))
        batch,_=extractor(samples,geometry)
        optimizer.zero_grad(set_to_none=True)
        loss,levels=bank(batch,seed=config['seed']+step*100)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(),1.0)
        optimizer.step()
        if any(p.grad is not None or p.requires_grad for p in extractor.detector.parameters()):
            raise RuntimeError('Detector was not completely frozen')
        record={'step':step+1,'image_ids':[item['image_id'] for item in items],
                'loss':float(loss.detach()),'seconds':time.time()-started,
                'levels':[{'valid_mse':float(item['valid_mse'].detach()),
                           'masked_mse':float(item['masked_mse'].detach()) if item['masked_mse'] is not None else None,
                           'valid_count':item['valid_count'],'masked_count':item['masked_count']} for item in levels]}
        with (out/'train.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        if (step+1)%10==0:print(json.dumps(record),flush=True)
        if (step+1)%100==0 or step+1==config['steps']:
            state={'bank':bank.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                   'config':config,'baseline':baseline,'code':git_provenance(),
                   'complete':step+1==config['steps']}
            temporary=out/'checkpoint.tmp';torch.save(state,temporary);temporary.replace(out/'checkpoint.pth')
    write_json(out/'training_complete.json',{'complete':True,'steps':config['steps'],'seconds':time.time()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output');parser.add_argument('--resume')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.config,args.output,args.resume,args.device)
