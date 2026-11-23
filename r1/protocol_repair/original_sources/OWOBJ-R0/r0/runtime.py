"""Real-detector entrypoints are gated until the running B0 has completed."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from r0.features import FrozenSpatialExtractor
from r0.types import ImageGeometry

REPO = Path(__file__).resolve().parents[1]


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config['source'] not in ('S1','S2') or config['task'] != 1:
        raise ValueError('Only R0 spatial S1/S2, M-OWODB Task1 are supported')
    if not 0 <= config['mask_ratio'] <= 1 or config['channels'] != [256]*4:
        raise ValueError('Invalid mask ratio or official projected channels')
    return config


def guard_b0_complete(config):
    state_path = Path(config['b0_status'])
    state = json.loads(state_path.read_text())
    if state.get('state') != 'training_complete' or state.get('completed_epochs') != 41:
        raise RuntimeError('B0尚未完成41 epoch；禁止启动真实R0训练/特征提取。当前只允许CPU合成Smoke。')
    checkpoint = Path(config['baseline_checkpoint'])
    if not checkpoint.is_file():
        raise RuntimeError('Final B0 checkpoint does not exist: '+str(checkpoint))
    summary_path=Path(config['b0_analysis_summary'])
    if not summary_path.is_file():
        raise RuntimeError('B0训练已结束，但正式评估/Query分析尚未完成，R0仍被阻止。')
    summary=json.loads(summary_path.read_text())
    if summary.get('scope')!='full_task1' or not summary.get('official_entrypoint_metrics_identical'):
        raise RuntimeError('B0最终分析和官方评估一致性尚未确认。')
    pid_path=Path(config['b0_pipeline_pid'])
    if pid_path.is_file():
        process=Path('/proc')/pid_path.read_text().strip()
        if process.exists():
            command=(process/'cmdline').read_bytes()
            state=(process/'stat').read_text().split()[2]
            if b'b0/pipeline.sh' in command and state!='Z':
                raise RuntimeError('B0流水线仍运行，禁止并发启动真实R0。')
    if 'B0_PIPELINE_COMPLETE' not in Path(config['b0_pipeline_log']).read_text():
        raise RuntimeError('B0完整流水线结束标记尚未出现。')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def git_provenance():
    return {'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'worktree_status':subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()}


def load_detector(config,device):
    guard_b0_complete(config)  # before model import, checkpoint loading or device allocation
    path = Path(config['baseline_checkpoint'])
    checkpoint = torch.load(path,map_location='cpu')
    if checkpoint['epoch'] != 40 or not checkpoint.get('epoch_complete',False):
        raise RuntimeError('A complete final B0 epoch40 checkpoint is required')
    args = copy.deepcopy(checkpoint['args'])
    if isinstance(args,dict):args=argparse.Namespace(**args)
    if args.dataset != 'TOWOD' or args.CUR_INTRODUCED_CLS != 20 or args.num_queries != 100:
        raise RuntimeError('Baseline checkpoint protocol/architecture mismatch')
    args.device = str(device)
    from models import build_model
    previous = os.getcwd()
    try:
        os.chdir(REPO)  # official DINO load uses a relative path in this worktree
        model,_,postprocessors,_ = build_model(args,mode='sketch')
    finally:
        os.chdir(previous)
    model.load_state_dict(checkpoint['model'],strict=True)
    provenance={'path':str(path),'sha256':file_sha256(path),
                'training_commit':checkpoint.get('git_commit'),'epoch':checkpoint['epoch']}
    summary=json.loads(Path(config['b0_analysis_summary']).read_text())
    if summary['checkpoint']['sha256']!=provenance['sha256']:
        raise RuntimeError('B0正式分析所用权重与当前冻结权重不一致。')
    del checkpoint
    model.to(device)
    return FrozenSpatialExtractor(model,config['source']),postprocessors,provenance


def pack_images(items,device):
    from util.misc import nested_tensor_from_tensor_list
    samples=nested_tensor_from_tensor_list([item['image'].to(device) for item in items])
    padded_hw=tuple(samples.tensors.shape[-2:])
    geometry=[ImageGeometry(item['image_id'],tuple(item['original_hw']),tuple(item['resized_hw']),padded_hw)
              for item in items]
    return samples,geometry


class StepBatches:
    """Fixed image order at a fixed step, including after a resumed run."""
    def __init__(self,length,batch_size,start,steps,seed):
        if length <= 0:raise ValueError('Empty dataset')
        self.length,self.batch_size,self.start,self.steps,self.seed=length,batch_size,start,steps,seed

    def __len__(self):return self.steps-self.start

    def __iter__(self):
        cached_epoch,order=None,None
        for step in range(self.start,self.steps):
            indices=[]
            for offset in range(self.batch_size):
                epoch,position=divmod(step*self.batch_size+offset,self.length)
                if epoch != cached_epoch:
                    order=torch.randperm(self.length,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                    cached_epoch=epoch
                indices.append(order[position])
            yield indices
