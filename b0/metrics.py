"""Offline-only geometry and grouping; never imported by the training loss."""
import numpy as np


def iou(boxes, gt):
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    gt=np.asarray(gt,dtype=np.float64).reshape(-1,4)
    lo=np.maximum(boxes[:,None,:2],gt[None,:,:2])
    hi=np.minimum(boxes[:,None,2:],gt[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(-1)
    gt_area=np.maximum(gt[:,2:]-gt[:,:2],0).prod(-1)
    return intersection/np.maximum(area[:,None]+gt_area[None,:]-intersection,1e-15)


def query_groups(boxes, gt_boxes, gt_known, difficult, positive_iou=0.5, background_iou=0.1):
    overlaps=iou(boxes,gt_boxes)
    groups=np.full(len(boxes),'ambiguous',dtype='<U10')
    if overlaps.shape[1]==0:
        groups[:]='background'
        return groups
    closest=overlaps.argmax(1) # ties follow deterministic XML order
    maximum=overlaps[np.arange(len(boxes)),closest]
    groups[maximum<background_iou]='background'
    positive=(maximum>=positive_iou)&~np.asarray(difficult,dtype=bool)[closest]
    groups[positive&np.asarray(gt_known,dtype=bool)[closest]]='known'
    groups[positive&~np.asarray(gt_known,dtype=bool)[closest]]='unknown'
    return groups


def official_unknown_hits(lines, gt_by_image):
    """Replay released VOC evaluator's score rounding, IoU>0.5, greedy matches.

    lines are the evaluator's own formatted unknown-class prediction strings.
    gt_by_image uses original XML integer boxes, including difficult objects.
    Return unique image/GT indices; official denominator may contain duplicates.
    """
    values=[line.split() for line in lines]
    scores=np.array([float(row[1]) for row in values])
    hits=set()
    for index in np.argsort(-scores):
        row=values[index]
        stem=row[0]
        gt=gt_by_image[stem]
        candidates=np.flatnonzero(~gt['known'])
        if not len(candidates):
            continue
        reference=gt['xml_boxes'][candidates]
        box=np.array([float(x) for x in row[2:]],dtype=float)
        overlap_lo=np.maximum(reference[:,:2],box[:2])
        overlap_hi=np.minimum(reference[:,2:],box[2:])
        intersection=np.maximum(overlap_hi-overlap_lo+1,0).prod(-1)
        area=((box[2]-box[0]+1)*(box[3]-box[1]+1))
        gt_area=(reference[:,2:]-reference[:,:2]+1).prod(-1)
        overlaps=intersection/(area+gt_area-intersection)
        nearest=int(candidates[overlaps.argmax()])
        if overlaps.max()>0.5 and not gt['difficult'][nearest]:
            hits.add((stem,nearest))
    return hits
