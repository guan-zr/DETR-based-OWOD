"""Independent AE banks share only frozen detector forwards and image order."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from r1.runtime import config_read,load_frozen
from r1.reuse import ImageOnlyDataset,image_collate,pack_images,StepBatches,seed_all,write_json,file_sha256
from r1.reconstruction import QueryAutoencoder
from r1.train import train_step


def run(paths,root,evaluate=True):
    configs=[config_read(p) for p in paths];first=configs[0];out=Path(root);out.mkdir(parents=True,exist_ok=True)
    for c in configs:
        for key in ['baseline_checkpoint','train_split','eval_split','data_root','resize','batch_size','seed','steps']:
            if c[key]!=first[key]:raise RuntimeError('Shared forward protocol mismatch: '+key)
    dataset=ImageOnlyDataset(first['data_root'],first['train_split'],**first['resize'])
    testids=set((Path(first['data_root'])/'ImageSets/TOWOD'/(first['eval_split']+'.txt')).read_text().splitlines())
    if set(dataset.ids)&testids:raise RuntimeError('Train/evaluation image ID overlap')
    extractor,_,baseline=load_frozen(first,torch.device('cuda'))
    if len(extractor.detector.transformer.decoder.layers)!=6 or extractor.detector.transformer.d_model!=256:raise RuntimeError('Unexpected decoder architecture')
    models=[];optimizers=[]
    for c in configs:
        seed_all(c['seed']);m=QueryAutoencoder(256,**c['autoencoder'],mask_ratio=c['mask_ratio']).cuda();models.append(m)
        optimizers.append(torch.optim.AdamW(m.parameters(),**c['optimizer']))
    bundle=out/'bundle.pth';start=0
    if bundle.exists():
        saved=torch.load(bundle,map_location='cuda')
        if saved['configs']!=configs or saved['baseline']!=baseline:raise RuntimeError('Shared bank resume mismatch')
        start=saved['step']
        for model,opt,state in zip(models,optimizers,saved['banks']):model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    else:
        for c in configs:
            p=Path(c['output'])
            if p.exists() and any(p.iterdir()):raise RuntimeError('Nonempty new training output')
    for c in configs:
        p=Path(c['output']);p.mkdir(parents=True,exist_ok=True)
        write_json(p/'run.json',dict(config=c,baseline=baseline,actual_layer=c['layer'],channels=256,train_annotation_access=False,shared_frozen_forward=True,supervision='feature reconstruction only'))
    loader=DataLoader(dataset,batch_sampler=StepBatches(len(dataset),1,start,first['steps'],first['seed']),num_workers=0,collate_fn=image_collate)
    started=time.time()
    for step,items in enumerate(loader,start):
        samples,_=pack_images(items,torch.device('cuda'));queries,_=extractor(samples)
        if tuple(queries.raw.shape)!=(6,1,100,256):raise RuntimeError('Unexpected Query tensor')
        for c,m,opt in zip(configs,models,optimizers):
            result=train_step(m,queries.representation(c['representation'])[c['layer']-1],opt,c['seed']+step)
            record=dict(step=step+1,image_id=items[0]['image_id'],layer=c['layer'],full_mse=float(result['loss'].detach()),masked_mse=float(result['masked_mse'].detach()) if result['masked_mse'] is not None else None)
            with (Path(c['output'])/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if any(p.grad is not None or p.requires_grad for p in extractor.detector.parameters()):raise RuntimeError('Frozen detector gradient violation')
        if (step+1)%100==0 or step+1==first['steps']:
            banks=[dict(model=m.state_dict(),optimizer=o.state_dict()) for m,o in zip(models,optimizers)]
            tmp=out/'bundle.tmp';torch.save(dict(configs=configs,baseline=baseline,step=step+1,banks=banks),tmp);tmp.replace(bundle)
            for c,state in zip(configs,banks):
                p=Path(c['output']);temp=p/'checkpoint.tmp';torch.save(dict(**state,config=c,baseline=baseline,step=step+1,layer=c['layer'],channels=256,complete=step+1==c['steps']),temp);temp.replace(p/'checkpoint.pth')
            write_json(out/'progress.json',dict(step=step+1,total_steps=first['steps'],banks=len(configs),seconds_this_invocation=time.time()-started))
            print('AE bank',step+1,'/',first['steps'],'models',len(configs),'seconds',time.time()-started,flush=True)
    hashes={c['name']+'_'+c['representation']:file_sha256(Path(c['output'])/'checkpoint.pth') for c in configs}
    # Evaluation is independent and CPU-only. Finish six preset configurations
    # concurrently, then the outer pipeline verifies/reuses each full summary.
    extractor.cpu()
    for model in models:model.cpu()
    torch.cuda.empty_cache()
    from r1.formal_evalgroup import run as evaluate_group
    write_json(out/'progress.json',dict(step=first['steps'],total_steps=first['steps'],banks=len(configs),state='cpu_evaluation'))
    if evaluate:evaluate_group(paths,out)
    write_json(out/'complete.json',dict(complete=True,steps=first['steps'],banks=len(configs),checkpoint_sha256=hashes,shared_forward_only=True,cpu_evaluations_complete=bool(evaluate)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.configs,a.output)
