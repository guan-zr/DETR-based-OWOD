"""Offline region labels relative to available 80-class benchmark annotations."""
import torch
from torchvision.ops import box_iou


def label_regions(boxes, gt, positive_iou=0.5, background_iou=0.1):
    if not 0 <= background_iou < positive_iou <= 1:
        raise ValueError('Require 0 <= background_iou < positive_iou <= 1')
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
    overlaps = box_iou(boxes[selected], gt['boxes'].cpu())
    values, indices = overlaps.max(1)
    maximum[selected], nearest[selected] = values, indices
    for index, value, matched in zip(selected.tolist(),values.tolist(),indices.tolist()):
        if value < background_iou:
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
