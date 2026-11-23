"""Four new image-only AEs, same 2000 images; collect training-only channel means."""
import json,time,copy
from pathlib import Path
import numpy as np
import torch
from r0.round2.common import *
from r0.runtime import load_detector,pack_images,seed_all
from r0.data import ImageOnlyDataset
from r0.reconstruction import FMRBank
from r0.streaming import extract_both

def atomic_torch(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def ensure_initializations():
    folder=ROOT/'initializations';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():manifest_check(folder/'complete.json');return
    for seed in SEEDS:
        p=init_path(seed)
        if p.exists():continue
        seed_all(seed);temporary,_,baseline=load_detector(config(),torch.device('cpu'))
        bank=FMRBank(config()['channels'],**config()['autoencoder'],mask_ratio=config()['mask_ratio'])
        atomic_torch(dict(bank=bank.state_dict(),seed=seed,baseline=baseline,scope='UNTRAINED random initialization; not a training checkpoint'),p)
        del temporary,bank
    write_json(folder/'complete.json',dict(complete=True,sha256={f'seed{s}.pth':file_sha256(init_path(s)) for s in SEEDS}))

def make_bank(seed,source=None,trained=False):
    c=config();bank=FMRBank(c['channels'],**c['autoencoder'],mask_ratio=c['mask_ratio'])
    saved=torch.load(train_path(source,seed) if trained else init_path(seed),map_location='cpu')
    if trained and (not saved.get('complete') or saved['step']!=2000):raise RuntimeError('Incomplete AE')
    bank.load_state_dict(saved['bank'],strict=True)
    return bank.cuda()

def main():
    out=ROOT/'training';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('New training complete; preserved');return
    ensure_initializations();c=config();seed_all(73);extractor,_,baseline=load_detector(c,torch.device('cuda'))
    banks={(s,k):make_bank(k).train() for s in SOURCES for k in [74,75]};optimizers={key:torch.optim.AdamW(bank.parameters(),**c['optimizer']) for key,bank in banks.items()}
    means={s:[torch.zeros(256,device='cuda',dtype=torch.float64) for _ in range(4)] for s in SOURCES};counts=[0]*4;start=0
    if (out/'bundle.pth').exists():
        state=torch.load(out/'bundle.pth',map_location='cpu');start=state['step']
        if state['sequence_sha256']!=file_sha256(R0/'formal_20261111/train_image_sequence.txt'):raise RuntimeError('Training sequence changed')
        for key in banks:banks[key].load_state_dict(state['banks'][key]);optimizers[key].load_state_dict(state['optimizers'][key])
        means={s:[v.cuda() for v in state['means'][s]] for s in SOURCES};counts=state['counts']
    dataset=ImageOnlyDataset(c['data_root'],c['train_split'],**c['resize']);lookup={s:i for i,s in enumerate(dataset.ids)};ids=sequence();assert len(ids)==len(set(ids))==2000
    write_json(out/'run.json',dict(base_config=c,training_seeds=[74,75],image_order_seed=73,image_sequence_sha256=file_sha256(R0/'formal_20261111/train_image_sequence.txt'),annotations_loaded=False,baseline=baseline,protocol_sha256=file_sha256(ROOT/'protocol.json')))
    started=time.time();torch.cuda.reset_peak_memory_stats()
    for step in range(start,2000):
        item=dataset[lookup[ids[step]]];samples,g=pack_images([item],torch.device('cuda'));batches,_=extract_both(extractor,samples,g)
        for source,batch in batches.items():
            for li,(feature,padding) in enumerate(zip(batch.levels,batch.padding_masks)):
                valid=~padding;means[source][li]+=feature.double().masked_fill(padding[:,None],0).sum((0,2,3))
                if source=='S1':counts[li]+=int(valid.sum())
            for seed in [74,75]:
                key=(source,seed);bank=banks[key];optimizer=optimizers[key];optimizer.zero_grad(set_to_none=True)
                loss,levels=bank(batch,seed=seed+step*100)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite AE loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(bank.parameters(),1.0);optimizer.step()
                folder=out/f'{source}_seed{seed}';folder.mkdir(exist_ok=True)
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,image_id=ids[step],seed=seed,loss=float(loss.detach()),levels=[float(x['valid_mse'].detach()) for x in levels],seconds=time.time()-started))+'\n')
        if any(p.requires_grad or p.grad is not None for p in extractor.detector.parameters()):raise RuntimeError('Detector gradient violation')
        if (step+1)%100==0 or step==1999:
            state=dict(step=step+1,banks={k:v.state_dict() for k,v in banks.items()},optimizers={k:o.state_dict() for k,o in optimizers.items()},means=means,counts=counts,
                sequence_sha256=file_sha256(R0/'formal_20261111/train_image_sequence.txt'),baseline=baseline,complete=step==1999)
            atomic_torch(state,out/'bundle.pth');write_json(out/'progress.json',dict(step=step+1,total=2000,active_banks=4,seconds=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
            print('four-bank training',step+1,'/2000',time.time()-started,flush=True)
    for (source,seed),bank in banks.items():
        p=train_path(source,seed)
        if not p.exists():atomic_torch(dict(bank=bank.state_dict(),step=2000,complete=True,seed=seed,config=dict(config(source),seed=seed,image_order_seed=73),baseline=baseline,image_sequence_sha256=file_sha256(R0/'formal_20261111/train_image_sequence.txt')),p)
    mean_file=out/'training_means.npz';np.savez_compressed(mean_file,**{s:np.stack([(v/counts[i]).float().cpu().numpy() for i,v in enumerate(values)]) for s,values in means.items()},counts=np.asarray(counts))
    write_json(out/'complete.json',dict(complete=True,steps_per_bank=2000,banks=4,new_seeds=[74,75],seconds_this_invocation=time.time()-started,means_definition='valid-spatial-position weighted per-channel mean over the fixed 2000 training images; no XML',sha256={str(train_path(s,k).relative_to(out)):file_sha256(train_path(s,k)) for s in SOURCES for k in [74,75]}|{'training_means.npz':file_sha256(mean_file)}))
if __name__=='__main__':main()
