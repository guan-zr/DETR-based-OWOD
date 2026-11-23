import json
from pathlib import Path
import torch
from r1.features import FrozenQueryExtractor
from r1.reuse import guard_b0_complete,load_detector

ROOT=Path(__file__).resolve().parents[1]


def config_read(path):
    config=json.loads(Path(path).read_text())
    if config['task']!=1 or config['representation'] not in ('raw','l2','pre_layernorm'):
        raise ValueError('Unsupported Task1 query configuration')
    if not isinstance(config['layer'],int) or config['layer']<1 or config['steps']<1 or not 0<=config['mask_ratio']<=1:
        raise ValueError('Layer IDs are 1-based; steps and mask ratio must be valid')
    return config


def load_frozen(config,device):
    guard_b0_complete(config)
    spatial,postprocessors,baseline=load_detector(dict(config,source='S1'),device)
    return FrozenQueryExtractor(spatial.detector),postprocessors,baseline


def clipped_boxes(boxes,original_hw):
    boxes=torch.as_tensor(boxes,dtype=torch.float32).clone()
    height,width=original_hw
    boxes[:,0::2]=boxes[:,0::2].clamp(0,width);boxes[:,1::2]=boxes[:,1::2].clamp(0,height)
    return boxes
