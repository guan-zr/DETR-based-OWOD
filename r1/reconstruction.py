import torch
from torch import nn


class QueryAutoencoder(nn.Module):
    """One shared MLP for all B*Q slots; explicit binary mask channels as input."""
    def __init__(self,channels=256,hidden=128,bottleneck=32,mask_ratio=.6):
        super().__init__()
        if min(channels,hidden,bottleneck)<=0 or not 0<=mask_ratio<=1:raise ValueError('Invalid capacity/mask')
        self.channels=channels;self.mask_ratio=mask_ratio
        self.network=nn.Sequential(nn.Linear(2*channels,hidden),nn.GELU(),nn.Linear(hidden,bottleneck),
                                   nn.GELU(),nn.Linear(bottleneck,hidden),nn.GELU(),nn.Linear(hidden,channels))

    def forward(self,query,seed=None,mask_ratio=None):
        if query.ndim!=3 or query.shape[-1]!=self.channels:raise ValueError('Expected [B,Q,C]')
        target=query.detach();ratio=self.mask_ratio if mask_ratio is None else mask_ratio
        if not 0<=ratio<=1:raise ValueError('Invalid mask ratio')
        generator=None if seed is None else torch.Generator(device=query.device).manual_seed(seed)
        mask=torch.rand(query.shape,device=query.device,generator=generator)<ratio
        # Actual zeros (mask=0) remain distinguishable from masked dimensions (mask=1).
        masked_input=torch.cat([target.masked_fill(mask,0),mask.to(target.dtype)],-1)
        prediction=self.network(masked_input)
        squared=(prediction-target).square();full=squared.mean(-1);count=mask.sum(-1)
        masked=(squared*mask).sum(-1)/count.clamp(min=1)
        return {'prediction':prediction,'mask':mask,'masked_input':masked_input,
                'full_error':full,'masked_error':masked,'masked_count':count,
                'loss':full.mean(),'masked_mse':(squared*mask).sum()/mask.sum().clamp(min=1) if mask.any() else None}
