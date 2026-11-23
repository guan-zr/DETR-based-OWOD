"""All scores use the original per-point -> support-normalized ROI operator."""
import numpy as np
import torch
from r0.round2.common import *
from r0.evaluate import ensemble_errors,image_seed
from r0.regions import mean_region_error

def roi_maps(batch,maps,boxes):
    per=[];valid=[];fraction=[]
    for error,padding,stride in zip(maps,batch.padding_masks,batch.strides):
        scored=mean_region_error(error[0],~padding[0],boxes,batch.geometry[0],stride,**config()['roi'])
        per.append(scored.scores.cpu().numpy().astype(np.float64));valid.append(scored.valid.cpu().numpy());fraction.append(scored.valid_fraction.cpu().numpy())
    values=np.stack(per,1);v=np.stack(valid,1);fractions=np.stack(fraction,1)
    overall=np.mean(values,axis=1);overall[~v.all(1)]=np.nan
    return np.column_stack([values,overall]),np.column_stack([v,v.all(1)]),np.column_stack([fractions,fractions.min(1)])

def simple_maps(batch,mean):
    result={name:[] for name in ['l2','energy','channel_variance','mean_error']}
    for i,feature in enumerate(batch.levels):
        result['l2'].append(feature.square().sum(1).sqrt())
        result['energy'].append(feature.square().mean(1))
        result['channel_variance'].append((feature-feature.mean(1,keepdim=True)).square().mean(1))
        result['mean_error'].append((feature-mean[i][None,:,None,None]).square().mean(1))
    return result

@torch.no_grad()
def score_batch(batches,banks,means,boxes):
    n=len(boxes);scores=np.full((n,2,len(METHODS),5),np.nan);valid=None
    for si,source in enumerate(SOURCES):
        batch=batches[source];seed=image_seed(batch.geometry[0].image_id,73)
        for method,bank in banks[source].items():
            errors=ensemble_errors(bank,batch,.6,4,seed)
            values,v,_=roi_maps(batch,[level['error'] for level in errors],boxes)
            scores[:,si,METHODS.index(method),:]=values
            if valid is None:valid=v
            elif not np.array_equal(valid,v):raise RuntimeError('Method ROI geometry differs')
        for method,maps in simple_maps(batch,means[source]).items():
            values,v,_=roi_maps(batch,maps,boxes);scores[:,si,METHODS.index(method),:]=values
            if not np.array_equal(valid,v):raise RuntimeError('Simple-statistic ROI geometry differs')
    return scores,valid
