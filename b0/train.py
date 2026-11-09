"""Single-GPU execution of the official OWOBJ model/criterion with accumulation.

No architecture, query-count, loss, augmentation or inference changes. The
effective batch is the 4x5 images used by the provided 4-GPU recipe.
"""
import argparse
import datetime
import json
import math
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from b0.common import ARTIFACTS, build, datasets, json_write, model_args, optimizer_for, seed_all, to_device
from util.misc import collate_fn


def save(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def run(config=None, resume=None, smoke_updates=0):
    args = model_args(config)
    seed_all(args.seed)
    if not smoke_updates:
        audit = json.loads((ARTIFACTS/'reports/data_audit.json').read_text())
        smoke = json.loads((ARTIFACTS/'reports/smoke.json').read_text())
        memory = json.loads((ARTIFACTS/'reports/memory_stress.json').read_text())
        assert not audit['errors'] and smoke['passed'], 'Data audit and smoke are mandatory'
        assert smoke['physical_batch_size']==args.batch_size and memory['passed'] and memory['batch_size']==args.batch_size
        assert 'B0_CUDA_OPS_PASSED' in (ARTIFACTS/'logs/cuda_test_fast_nondet.log').read_text()
        assert shutil.disk_usage(ARTIFACTS).free > 15*1024**3, 'Need 15GiB free before full training'
    model, criterion, postprocessors, _ = build(args)
    train_ds, val_ds = datasets(args)
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    out = ARTIFACTS/f'checkpoints/smoke_b{args.batch_size}_a{args.accumulation_steps}' if smoke_updates else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    effective = args.batch_size * args.accumulation_steps
    start_epoch, offset = 0, 0
    elapsed_before = 0.0
    rng = None
    checkpoint_path = Path(resume) if resume else out/'checkpoint.pth'
    if checkpoint_path.exists() and not smoke_updates:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + (1 if checkpoint.get('epoch_complete', True) else 0)
        offset = checkpoint.get('next_micro_index', 0) if not checkpoint.get('epoch_complete', True) else 0
        elapsed_before = checkpoint.get('training_seconds',0)
        rng = checkpoint.get('rng')
        print(f'Resume {checkpoint_path}, epoch={start_epoch}, offset={offset}', flush=True)
    git_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if not smoke_updates:
        assert not dirty, 'Commit implementation changes before formal training'
    json_write(out/'config.json', vars(args))
    stop = {'requested':False}
    def request_stop(*unused):
        stop['requested'] = True
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum,request_stop)
    started = time.time()
    total_updates = 0
    torch.cuda.reset_peak_memory_stats()
    def state_for(epoch,complete,next_micro_index):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'lr_scheduler':scheduler.state_dict(),
            'epoch':epoch,'epoch_complete':complete,'next_micro_index':next_micro_index,
            'args':args,'git_commit':git_commit,'training_seconds':elapsed_before+time.time()-started,
            'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
            'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
                   'numpy':np.random.get_state(),'python':random.getstate()}}
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()
        order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        order = order[:len(order)//effective*effective]
        if smoke_updates:
            order = order[:smoke_updates*effective]
        begin = offset if epoch == start_epoch else 0
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=order[begin*args.batch_size:],
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
        if rng is not None:
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state_all(rng['cuda'])
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
            rng = None
        epoch_start = time.time()
        sums, num_micro, grad_norm = {}, 0, 0.0
        optimizer.zero_grad(set_to_none=True)
        for local_i,(samples,targets) in enumerate(loader):
            micro_index = begin+local_i
            samples,targets = to_device(samples,targets)
            assert all((target['labels'] < args.CUR_INTRODUCED_CLS).all() for target in targets), 'Unknown supervision'
            outputs = model(samples)
            loss_dict = criterion(outputs,targets)
            loss = sum(value*criterion.weight_dict[key] for key,value in loss_dict.items() if key in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss epoch={epoch} micro={micro_index}: {loss_dict}')
            (loss/args.accumulation_steps).backward()
            for key,value in loss_dict.items():
                sums[key] = sums.get(key,0)+float(value.detach())
            sums['weighted_total'] = sums.get('weighted_total',0)+float(loss.detach())
            num_micro += 1
            if (local_i+1)%args.accumulation_steps == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_max_norm))
                if not math.isfinite(grad_norm):
                    raise RuntimeError('Nonfinite gradient norm')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_updates += 1
                if total_updates%10 == 0 or total_updates == 1:
                    print(json.dumps({'epoch':epoch,'micro_index':micro_index+1,'micro_total':len(order)//args.batch_size,
                        'updates_this_process':total_updates,'loss':float(loss.detach()),'grad_norm':grad_norm,
                        'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':time.time()-epoch_start,
                        'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2}),flush=True)
                json_write(ARTIFACTS/'reports/train_status.json', {'state':'smoke' if smoke_updates else 'training',
                    'pid':os.getpid(),'epoch_index':epoch,'next_micro_index':micro_index+1,
                    'epochs_target':args.epochs,'updates_this_process':total_updates,
                    'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'training_seconds':elapsed_before+time.time()-started,
                    'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2})
                if not smoke_updates and total_updates%100==0:
                    save(out/'checkpoint.pth',state_for(epoch,False,micro_index+1))
            if stop['requested'] and (local_i+1)%args.accumulation_steps == 0:
                break
        complete = not stop['requested']
        if complete:
            scheduler.step()
        torch.cuda.synchronize()
        state = state_for(epoch,complete,0 if complete else micro_index+1)
        save(out/'checkpoint.pth',state)
        if complete and epoch in [0,args.lr_drop-1,args.epochs-1]:
            save(out/f'checkpoint{epoch:04}.pth',state)
        record = {'epoch':epoch,'complete':complete,'metrics':{k:v/max(num_micro,1) for k,v in sums.items()},
                  'seconds':time.time()-epoch_start,'training_seconds':state['training_seconds'],
                  'peak_memory_mib':state['peak_memory_mib']}
        if not smoke_updates and complete and (epoch in [0,1,args.epochs-1] or epoch%args.eval_every==0):
            from engine import evaluate
            stats,_ = evaluate(model,criterion,postprocessors,val_loader,val_ds,torch.device('cuda'),str(out),args)
            record['evaluation'] = stats['metrics']
            json_write(ARTIFACTS/'evaluations'/f'epoch{epoch:04}.json',stats)
        with (out/'epochs.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        if smoke_updates:
            return state,record
        if not complete:
            print('Stopped safely at an optimizer update; checkpoint saved',flush=True)
            return
        from b0.write_reports import main as update_reports
        update_reports()
    json_write(ARTIFACTS/'reports/train_status.json',{'state':'training_complete','completed_epochs':args.epochs,
        'training_seconds':elapsed_before+time.time()-started,'checkpoint':str(out/f'checkpoint{args.epochs-1:04}.pth'),
        'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--resume')
    parser.add_argument('--smoke-updates',type=int,default=0)
    options=parser.parse_args()
    run(options.config,options.resume,options.smoke_updates)
