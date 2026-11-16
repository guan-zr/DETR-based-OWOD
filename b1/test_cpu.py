"""Meaningful recovery/scheduling checks without claiming a full experiment test."""
import os, tempfile
from pathlib import Path
import torch
from b1.common import E,write,args_for,save,final_link,sha
from models.prob_deformable_sketch_detr import ProbObjectnessHead

def main():
 a,b=args_for('stage_a'),args_for('stage_b')
 assert (a.PREV_INTRODUCED_CLS,a.CUR_INTRODUCED_CLS,a.epochs)==(20,20,51)
 assert (b.PREV_INTRODUCED_CLS,b.CUR_INTRODUCED_CLS,b.epochs)==(20,20,111)
 assert a.freeze_prob_model and not b.freeze_prob_model
 assert a.lr==2e-5 and b.lr==2e-4 and b.lr_drop==40
 assert len(range(41,a.epochs))==10 and len(range(51,b.epochs))==60
 # pretrain -> fresh scheduler. Check effective LR drop after40 stage steps.
 model=torch.nn.Linear(4,2);opt=torch.optim.AdamW(model.parameters(),lr=b.lr);sch=torch.optim.lr_scheduler.StepLR(opt,b.lr_drop)
 history=[]
 for i in range(60):opt.step();sch.step();history.append(opt.param_groups[0]['lr'])
 assert history[38]==2e-4 and abs(history[39]-2e-5)<1e-12 and abs(history[59]-2e-5)<1e-12
 # Atomic latest/previous rotation and final inode reuse; small synthetic state only.
 with tempfile.TemporaryDirectory(dir=E/'run_state',prefix='cpu_checkpoint_') as tmp:
  latest=Path(tmp)/'checkpoint.pth';final=Path(tmp)/'checkpoint0110.pth'
  save(latest,dict(epoch=51,marker=torch.tensor([1.])));save(latest,dict(epoch=52,marker=torch.tensor([2.])))
  assert torch.load(latest)['epoch']==52 and torch.load(Path(tmp)/'checkpoint.previous.pth')['epoch']==51
  final_link(latest,final);assert latest.stat().st_ino==final.stat().st_ino and sha(latest)==sha(final)
 # Resume optimizer/scheduler/RNG after an interrupted update reproduces next dropout update.
 import copy
 model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2));opt=torch.optim.AdamW(model.parameters(),lr=2e-5);sch=torch.optim.lr_scheduler.StepLR(opt,35)
 x=torch.ones(2,4);torch.manual_seed(42)
 for _ in range(2):opt.zero_grad();model(x).square().mean().backward();opt.step()
 snapshot=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=torch.get_rng_state()))
 opt.zero_grad();model(x).square().mean().backward();opt.step();expected=copy.deepcopy(model.state_dict())
 model.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);sch.load_state_dict(snapshot['scheduler']);torch.set_rng_state(snapshot['rng'])
 opt.zero_grad();model(x).square().mean().backward();opt.step()
 assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
 # The freeze bug is observed directly, not assumed from flag name.
 head=ProbObjectnessHead(4);head.train();head.freeze_prob_model();assert not head.objectness_bn.training and not head.objectness_gau_bn.training
 head.train();assert head.objectness_bn.training and head.objectness_gau_bn.training
 # A dedicated DataLoader generator must not advance model dropout RNG on iterator creation.
 from torch.utils.data import DataLoader,TensorDataset
 torch.manual_seed(42);state=torch.get_rng_state().clone()
 loader=DataLoader(TensorDataset(torch.ones(4,1)),batch_size=2,generator=torch.Generator().manual_seed(42));next(iter(loader))
 assert torch.equal(state,torch.get_rng_state())
 report=dict(passed=True,checks=['Task2_20_20_and_epoch_ranges','fresh_stageB_LR_drop_after40_steps','atomic_latest_previous_checkpoint_and_final_hardlink','actual_official_freeze_train_reset','loader_generator_does_not_advance_model_RNG','model_optimizer_scheduler_rng_resume_reproduces_next_dropout_update'])
 write(E/'reports/cpu_checks.json',report);print(report,flush=True)
if __name__=='__main__':main()
