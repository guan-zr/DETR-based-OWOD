"""Offline region labels relative to available 80-class benchmark annotations."""
from collections import Counter
import torch
from torchvision.ops import box_iou


def label_regions(boxes, gt, positive_iou=0.5, background_iou=0.1, background_rule='disjoint_gt'):
    if not 0 <= background_iou < positive_iou <= 1:
        raise ValueError('Require 0 <= background_iou < positive_iou <= 1')
    if background_rule not in ('disjoint_gt','iou_only'):
        raise ValueError('Unknown background rule: '+str(background_rule))
    boxes = torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4)
    labels = ['ambiguous'] * len(boxes)
    finite = torch.isfinite(boxes).all(1)
    valid = finite & (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
    maximum = torch.zeros(len(boxes))
    nearest = torch.full((len(boxes),),-1,dtype=torch.long)
    for index in (~valid).nonzero(as_tuple=True)[0].tolist():
        labels[index] = 'invalid'
    selected = valid.nonzero(as_tuple=True)[0]
    if not len(gt['boxes']):
        for index in selected.tolist(): labels[index] = 'background'
        return labels, maximum, nearest
    gt_boxes = torch.as_tensor(gt['boxes'],dtype=torch.float32).cpu().reshape(-1,4)
    if not (torch.isfinite(gt_boxes).all() and (gt_boxes[:,2:] > gt_boxes[:,:2]).all()):
        raise ValueError('Invalid GT geometry; cannot certify background')
    overlaps = box_iou(boxes[selected], gt_boxes)
    # Use intersection geometry, not IoU or candidate/GT area thresholds.
    sides = torch.minimum(boxes[selected,None,2:],gt_boxes[None,:,2:])-torch.maximum(boxes[selected,None,:2],gt_boxes[None,:,:2])
    intersects = (sides > 0).all(2).any(1)
    values, indices = overlaps.max(1)
    maximum[selected], nearest[selected] = values, indices
    for index, value, matched, touching in zip(selected.tolist(),values.tolist(),indices.tolist(),intersects.tolist()):
        if (not touching if background_rule=='disjoint_gt' else value < background_iou):
            labels[index] = 'background'
        elif value >= positive_iou and not gt['difficult'][matched]:
            labels[index] = 'known' if gt['known'][matched] else 'unknown'
    return labels, maximum, nearest


def background_candidates(original_hw, seed, per_image=64):
    """Fixed image-only proposals; evaluation then filters against ALL GT.

    Same seed and image produce identical proposal pool for S1/S2. No error or
    unknown labels select proposal geometry. Not used by the training program.
    """
    height,width = original_hw
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(per_image,4,generator=generator)
    box_w = (0.05 + 0.35*values[:,0])*width
    box_h = (0.05 + 0.35*values[:,1])*height
    x = values[:,2]*(width-box_w); y = values[:,3]*(height-box_h)
    return torch.stack([x,y,x+box_w,y+box_h],1)


GROUPS = ('known','unknown','background','ambiguous','invalid')


def grouping_audit(boxes,gt,positive_iou=.5,background_iou=.1):
    """Geometry-only paired counts on the exact same candidate pool; no scores."""
    old=label_regions(boxes,gt,positive_iou,background_iou,'iou_only')[0]
    new=label_regions(boxes,gt,positive_iou,background_iou,'disjoint_gt')[0]
    before=Counter(old);after=Counter(new);transitions=Counter(a+'->'+b for a,b in zip(old,new))
    return {'before_iou_only':{g:before[g] for g in GROUPS},
            'after_disjoint_gt':{g:after[g] for g in GROUPS},
            'delta':{g:after[g]-before[g] for g in GROUPS},'transitions':dict(transitions)}


class GroupingAudit:
    def __init__(self):self.pools={}

    def add(self,pool,boxes,gt,positive_iou=.5,background_iou=.1):
        target=self.pools.setdefault(pool,{key:Counter() for key in
            ('before_iou_only','after_disjoint_gt','delta','transitions')})
        for key,counts in grouping_audit(boxes,gt,positive_iou,background_iou).items():target[key].update(counts)

    def summarize(self):
        return {pool:{key:dict(counts) for key,counts in data.items()} for pool,data in self.pools.items()}
