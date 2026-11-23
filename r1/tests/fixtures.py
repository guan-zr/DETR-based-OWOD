from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from r1.features import FrozenQueryExtractor,image_frames
from r1.reuse import read_eval_gt
from r0.types import ImageGeometry


class FakeLayer(nn.Module):
    def __init__(self,channels):
        super().__init__();self.linear=nn.Linear(channels,channels);self.norm3=nn.LayerNorm(channels)
    def forward(self,x):return self.norm3(self.linear(x))


class FakeDecoder(nn.Module):
    def __init__(self,layers,channels):
        super().__init__();self.layers=nn.ModuleList([FakeLayer(channels) for _ in range(layers)])
    def forward(self,x):
        results=[]
        for layer in self.layers:x=layer(x);results.append(x)
        return torch.stack(results),None


class FakeTransformer(nn.Module):
    def __init__(self,layers,channels):
        super().__init__();self.d_model=channels;self.decoder=FakeDecoder(layers,channels)


class FakeObjectness(nn.Module):
    def __init__(self,channels):super().__init__();self.bn=nn.BatchNorm1d(channels,affine=False)
    def forward(self,x):
        batch,queries,channels=x.shape
        return self.bn(x.reshape(-1,channels)).reshape(batch,queries,channels).square().sum(-1)


class FakeDetector(nn.Module):
    def __init__(self,layers=6,queries=7,channels=8):
        super().__init__();self.transformer=FakeTransformer(layers,channels)
        self.query=nn.Parameter(torch.randn(queries,channels));self.classifier=nn.Linear(channels,81)
        self.objectness=FakeObjectness(channels);self.aux=True
    def forward(self,samples):
        batch=len(samples.tensors);hs,_=self.transformer.decoder(self.query[None].expand(batch,-1,-1))
        outputs=[]
        for layer,x in enumerate(hs):
            boxes=x.new_zeros(batch,x.shape[1],4)
            boxes[:,:,0]=.1+torch.arange(x.shape[1],device=x.device)*.12
            boxes[:,:,1]=.12+layer*.05;boxes[:,:,2:]=.08
            outputs.append({'pred_boxes':boxes,'pred_logits':self.classifier(x),
                            'pred_obj':self.objectness(x),'penu_layer':x})
        final=dict(outputs[-1])
        if self.aux:final['aux_outputs']=outputs[:-1]
        return final


def synthetic():
    torch.manual_seed(91);torch.set_num_threads(1)
    model=FakeDetector().eval()
    samples=SimpleNamespace(tensors=torch.zeros(2,3,64,64),mask=torch.ones(2,64,64,dtype=torch.bool))
    samples.mask[0,:50,:60]=False;samples.mask[1,:40,:50]=False
    geometry=[ImageGeometry('one',(100,120),(50,60),(64,64)),ImageGeometry('two',(80,100),(40,50),(64,64))]
    extractor=FrozenQueryExtractor(model);batch,outputs=extractor(samples)
    frames=list(image_frames(batch,geometry,list(range(20,80)),1.3/8))
    gt={}
    for frame in frames:
        # Annotated known/unknown boxes at each own decoder-layer geometry.
        boxes=np.concatenate([frame['boxes_xyxy'][i,:2] for i in range(3)])
        boxes=np.concatenate([boxes,frame['boxes_xyxy'][1,2:3]])
        gt[frame['image_id']]={'boxes':torch.from_numpy(boxes.copy()),
            'known':torch.tensor([True,False]*3+[False]),
            'difficult':torch.tensor([False]*6+[True])}
    return model,extractor,samples,geometry,batch,outputs,frames,gt
