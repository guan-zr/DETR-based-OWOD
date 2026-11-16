"""B0 single-GPU accumulation, official Task2 model/loss/scheduling, durable resume."""
import argparse, collections, datetime, json, math, os, signal, subprocess, time
from pathlib import Path
from torch.utils.data import DataLoader
from b1.common import *
from b0.common import optimizer_for
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from util.misc import collate_fn

class SeededDataset(torch.utils.data.Dataset):
 """Per-example RNG makes epoch/mid-epoch replay independent of worker prefetch."""
 def __init__(self,ds,epoch,seed):self.ds,self.epoch,self.seed=ds,epoch,seed
 def __len__(self):return len(self.ds)
 def __getitem__(self,i):
  s=(self.seed+self.epoch*1000003+i*97)%2**32
  random.seed(s);np.random.seed(s);torch.manual_seed(s)
  return self.ds[i]

def run(stage,smoke=0,resume=None):
 a=args_for(stage);seed();disk_check()
 if not smoke:
  assert json.loads((E/'reports/data_audit.json').read_text())['errors']==[]
  assert json.loads((E/f'reports/smoke_{stage}.json').read_text())['passed']
  assert 'B0_CUDA_OPS_PASSED' in (E/'logs/cuda_ops.log').read_text()
  assert json.loads((E/'reports/storage_budget.json').read_text())['passed']
 if stage=='stage_a': assert (E/'replay/learned_owod_t1_ft.provenance.json').exists()
 else: assert (E/'replay/learned_owod_t2_ft.provenance.json').exists() or smoke
 model,criterion,post,selection=build(a);ck=strict_load(model,resume or a.pretrain)
 expected=40 if stage=='stage_a' else 50
 if not resume: assert ck['epoch']==expected and ck.get('epoch_complete',True)
 out=E/'checkpoints'/('smoke_'+stage) if smoke else Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 optimizer=optimizer_for(model,a);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,a.lr_drop)
 start=ck['epoch']+1; offset=0; updates=0; training_seconds=0.; rng=None; sums={};num_micro=0
 if resume:
  assert Path(resume).resolve().is_relative_to(Path(a.output_dir).resolve()),'Must resume B1 same stage'
  assert ck['args'].train_set==a.train_set and ck['args'].PREV_INTRODUCED_CLS==20 and ck['args'].CUR_INTRODUCED_CLS==20
  assert ck.get('stage')==stage
  optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['lr_scheduler'])
  start=ck['epoch']+(1 if ck['epoch_complete'] else 0);offset=ck.get('next_micro_index',0) if not ck['epoch_complete'] else 0
  updates=ck.get('updates',0);training_seconds=ck.get('training_seconds',0);rng=ck['rng']
  if offset:sums=ck.get('epoch_sums',{});num_micro=ck.get('epoch_num_micro',0)
 print(f'STAGE {stage} start_epoch={start} target_exclusive={a.epochs} resume_offset={offset}',flush=True)
 before_head=model.class_embed[0].weight.detach().cpu().clone();before_bn=model.prob_obj_head[0].objectness_bn.running_mean.detach().cpu().clone()
 write(out/'config.json',vars(a));del ck
 ds=OWDetection(a,a.data_root,image_set=a.train_set,transforms=make_coco_transforms(a.train_set),dataset=a.dataset)
 stop={'signal':None}
 def request(signum,frame): stop['signal']=signum
 for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,request)
 started=time.time();torch.cuda.reset_peak_memory_stats(); stop_why=None
 def state(epoch,complete,next_micro):
  return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),epoch=epoch,epoch_complete=complete,next_micro_index=next_micro,args=a,stage=stage,rng=rng_state(),updates=updates,training_seconds=training_seconds+time.time()-started,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,epoch_sums=sums,epoch_num_micro=num_micro,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 first=True; label_counts=collections.Counter(); micro_since_start=0;smoke_updates=0
 for epoch in range(start,a.epochs):
  model.train();criterion.train()
  if a.freeze_prob_model and first:
   for obj in model.prob_obj_head:obj.freeze_prob_model()
   freeze_after_call=[obj.objectness_bn.training for obj in model.prob_obj_head]
   # Match official main -> train_one_epoch exactly: its model.train() undoes freeze.
   model.train()
   print('OFFICIAL_FREEZE_BN '+json.dumps(dict(after_flag=freeze_after_call,after_train=[obj.objectness_bn.training for obj in model.prob_obj_head],affine=False)),flush=True)
  first=False
  effective=a.batch_size*a.accumulation_steps
  order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
  order=order[:len(order)//effective*effective]
  if smoke:order=order[:smoke*effective]
  begin=offset if epoch==start else 0
  loader=DataLoader(SeededDataset(ds,epoch,a.seed),batch_size=a.batch_size,sampler=order[begin*a.batch_size:],num_workers=a.num_workers,collate_fn=collate_fn,pin_memory=True,generator=torch.Generator().manual_seed(a.seed+epoch))
  if rng:restore_rng(rng);rng=None
  if not begin:sums={};num_micro=0
  epoch_start=time.time();optimizer.zero_grad(set_to_none=True)
  for local_i,(samples,targets) in enumerate(loader):
   idx=begin+local_i;samples,targets=move(samples,targets)
   for t in targets:
    labels=t['labels'];assert ((labels>= (20 if stage=='stage_a' else 0)) & (labels<40)).all(),'Invalid/unknown supervision'
    label_counts.update(labels.detach().cpu().tolist())
   outputs=model(samples);losses=criterion(outputs,targets);loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
   if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {stage}:{epoch}:{idx}')
   (loss/a.accumulation_steps).backward()
   for k,v in losses.items():sums[k]=sums.get(k,0)+float(v.detach())
   sums['weighted_total']=sums.get('weighted_total',0)+float(loss.detach());num_micro+=1;micro_since_start+=1
   if (local_i+1)%a.accumulation_steps==0:
    grad=model.class_embed[0].weight.grad.detach();head_grad=dict(old=float(grad[:20].norm()),new=float(grad[20:40].norm()),future=float(grad[40:80].norm()),unknown=float(grad[80:].norm()))
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip_max_norm));assert math.isfinite(norm)
    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;smoke_updates+=1
    status=dict(at=now(),state='smoke' if smoke else 'training',pid=os.getpid(),stage=stage,epoch=epoch,next_micro_index=idx+1,total_micro=len(order)//a.batch_size,updates=updates,lr=[g['lr'] for g in optimizer.param_groups],loss=float(loss.detach()),grad_norm=norm,head_gradient_norms=head_grad,training_seconds=training_seconds+time.time()-started,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
    write(E/'run_state'/f'{stage}_status.json',status)
    if updates%10==0 or smoke or smoke_updates==1:print(json.dumps(status),flush=True)
    # Reserve makes it possible to save safely before the actual filesystem fills.
    space=shutil.disk_usage(E)
    if space.free<7*GIB:stop_why='disk_free_below_7GiB'
    if updates%100==0 and not smoke:save(out/'checkpoint.pth',state(epoch,False,idx+1))
    if stop['signal'] or stop_why:
     save(out/'checkpoint.pth',state(epoch,False,idx+1));write(E/'run_state'/f'{stage}_stopped.json',dict(at=now(),signal=stop['signal'],reason=stop_why,checkpoint=str(out/'checkpoint.pth')));return 75
  scheduler.step();torch.cuda.synchronize();st=state(epoch,True,0);save(out/'checkpoint.pth',st)
  after=model.class_embed[0].weight.detach().cpu();diff=(after-before_head).abs()
  record=dict(epoch=epoch,complete=True,micro_batches=num_micro,updates=updates,seconds=time.time()-epoch_start,training_seconds=st['training_seconds'],peak_memory_mib=st['peak_memory_mib'],metrics={k:v/max(num_micro,1) for k,v in sums.items()},classifier_abs_change=dict(old=float(diff[:20].max()),new=float(diff[20:40].max()),future=float(diff[40:80].max()),unknown=float(diff[80:].max())),bn_running_mean_max_change=float((model.prob_obj_head[0].objectness_bn.running_mean.detach().cpu()-before_bn).abs().max()),supervised_labels_seen=dict(label_counts),lr_after_scheduler=[g['lr'] for g in optimizer.param_groups])
  with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print('EPOCH_COMPLETE '+json.dumps(record),flush=True)
  if smoke:
   assert diff[20:40].max()>0 and head_grad['new']>0 and head_grad['future']==0
   write(E/f'reports/smoke_{stage}.json',dict(passed=True,stage=stage,updates=smoke_updates,record=record,backend=backend(),checkpoint=str(out/'checkpoint.pth'),head_gradient_norms=head_grad));return 0
  offset=0
  if epoch==a.epochs-1:final_link(out/'checkpoint.pth',out/f'checkpoint{epoch:04}.pth')
 final_link(out/'checkpoint.pth',out/f'checkpoint{a.epochs-1:04}.pth')
 write(E/'run_state'/f'{stage}_complete.json',dict(at=now(),stage=stage,exit_code=0,final_epoch=a.epochs-1,checkpoint=str(out/f'checkpoint{a.epochs-1:04}.pth'),training_seconds=training_seconds+time.time()-started,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2))
 return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['stage_a','stage_b'],required=True);p.add_argument('--smoke-updates',type=int,default=0);p.add_argument('--resume');o=p.parse_args();raise SystemExit(run(o.stage,o.smoke_updates,o.resume))
