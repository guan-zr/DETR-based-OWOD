"""R1-1 image-only training: never imports diagnostic grouping or evaluation."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from r1.reconstruction import QueryAutoencoder
from r1.reuse import ImageOnlyDataset,image_collate,pack_images,StepBatches,seed_all,write_json
from r1.runtime import config_read,load_frozen


def train_step(model,queries,optimizer,seed):
    optimizer.zero_grad(set_to_none=True);result=model(queries,seed=seed)
    if not torch.isfinite(result['loss']):raise RuntimeError('Nonfinite query reconstruction loss')
    result['loss'].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite AE gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite AE parameter')
    return result


def run(path,output=None,resume=None,device='cuda'):
    config=config_read(path);extractor,_,baseline=load_frozen(config,torch.device(device));seed_all(config['seed'])
    actual_layers=len(extractor.detector.transformer.decoder.layers)
    layer=min(config['layer'],actual_layers)
    channels=extractor.detector.transformer.d_model
    out=Path(output or config['output'])
    if out.exists() and any(out.iterdir()) and not resume:raise RuntimeError('Choose empty R1-1 output or --resume')
    out.mkdir(parents=True,exist_ok=True)
    model=QueryAutoencoder(channels=channels,**config['autoencoder'],mask_ratio=config['mask_ratio']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),**config['optimizer']);start=0
    if resume:
        saved=torch.load(resume,map_location=device)
        if saved['config']!=config or saved['baseline']['sha256']!=baseline['sha256'] or saved['layer']!=layer:
            raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
    write_json(out/'run.json',{'config':config,'baseline':baseline,'requested_layer':config['layer'],
        'actual_layer':layer,'channels':channels,'train_annotation_access':False,'supervision':'feature reconstruction only'})
    data=ImageOnlyDataset(config['data_root'],config['train_split'],**config['resize'])
    sampler=StepBatches(len(data),config['batch_size'],start,config['steps'],config['seed'])
    loader=DataLoader(data,batch_sampler=sampler,num_workers=0,collate_fn=image_collate)
    for step,items in enumerate(loader,start):
        samples,_=pack_images(items,torch.device(device));queries,_=extractor(samples)
        result=train_step(model,queries.representation(config['representation'])[layer-1],optimizer,config['seed']+step)
        if any(p.grad is not None or p.requires_grad for p in extractor.detector.parameters()):raise RuntimeError('Detector received gradient')
        record={'step':step+1,'layer':layer,'full_mse':float(result['loss'].detach()),
                'masked_mse':float(result['masked_mse'].detach()) if result['masked_mse'] is not None else None}
        with (out/'train.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        if (step+1)%100==0 or step+1==config['steps']:
            temporary=out/'checkpoint.tmp'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                'config':config,'baseline':baseline,'layer':layer,'channels':channels,'complete':step+1==config['steps']},temporary)
            temporary.replace(out/'checkpoint.pth')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output');parser.add_argument('--resume')
    parser.add_argument('--device',default='cuda');args=parser.parse_args();run(args.config,args.output,args.resume,args.device)
