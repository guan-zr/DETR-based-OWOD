import json
import random
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = Path('/root/autodl-tmp/experiments/B0_PROB_OWOBJ')


def model_args(config=None):
    from main_open_world import get_args_parser
    args = get_args_parser().parse_args([])
    args.__dict__.update(json.loads(Path(config or REPO/'b0/config.json').read_text()))
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build(args):
    from models import build_model
    model, criterion, postprocessors, exemplar = build_model(args, mode=args.model_type)
    return model.cuda(), criterion.cuda(), postprocessors, exemplar


def datasets(args):
    from main_open_world import get_datasets
    return get_datasets(args)


def optimizer_for(model, args):
    def match(name, keywords):
        return any(keyword in name for keyword in keywords)
    groups = [
        {'params': [p for n,p in model.named_parameters() if p.requires_grad and
                    not match(n,args.lr_backbone_names) and not match(n,args.lr_linear_proj_names)], 'lr':args.lr},
        {'params': [p for n,p in model.named_parameters() if p.requires_grad and match(n,args.lr_backbone_names)], 'lr':args.lr_backbone},
        {'params': [p for n,p in model.named_parameters() if p.requires_grad and match(n,args.lr_linear_proj_names)], 'lr':args.lr*args.lr_linear_proj_mult}
    ]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def to_device(samples, targets):
    return samples.to(torch.device('cuda')), [{k:v.cuda() for k,v in t.items()} for t in targets]


def clone_outputs(outputs):
    if torch.is_tensor(outputs):
        return outputs.detach().clone()
    if isinstance(outputs, dict):
        return {k:clone_outputs(v) for k,v in outputs.items()}
    if isinstance(outputs, list):
        return [clone_outputs(v) for v in outputs]
    return outputs


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    # Strict is essential: never silently evaluate random/missing heads.
    model.load_state_dict(checkpoint['model'], strict=True)
    return checkpoint
