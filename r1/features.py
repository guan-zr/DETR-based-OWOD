"""Actual decoder output and each layer's own auxiliary predictions, read-only."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


def preferred_layers(count):
    if count<1:raise ValueError('Decoder has no layers')
    return sorted(set(min(index,count) for index in (1,3,6)))


@dataclass
class QueryBatch:
    raw: torch.Tensor              # [L,B,Q,C], actual post-norm3 decoder output
    pre_layernorm: torch.Tensor    # [L,B,Q,C], input to each layer's norm3
    boxes: torch.Tensor            # [L,B,Q,4], that layer's normalized cxcywh
    logits: torch.Tensor           # [L,B,Q,K], that layer's classifier logits
    distance: torch.Tensor         # [L,B,Q], that layer's Gaussian squared distance

    def representation(self,name):
        if name=='raw':return self.raw
        if name=='l2':return F.normalize(self.raw,dim=-1)
        if name=='pre_layernorm':return self.pre_layernorm
        raise ValueError('Unknown query representation: '+name)


class FrozenQueryExtractor(nn.Module):
    def __init__(self,detector):
        super().__init__();self.detector=detector
        detector.requires_grad_(False);detector.eval()

    def train(self,mode=True):
        super().train(mode);self.detector.eval();return self

    @torch.no_grad()
    def forward(self,samples):
        self.detector.eval();captured={};pre={};handles=[]
        decoder=self.detector.transformer.decoder
        if any(not hasattr(layer,'norm3') for layer in decoder.layers):
            raise RuntimeError('Cannot capture actual pre-norm3 query at every layer')
        def decoder_output(module,inputs,output):captured['raw']=output[0].detach().clone()
        handles.append(decoder.register_forward_hook(decoder_output))
        for index,layer in enumerate(decoder.layers):
            def before_norm(module,inputs,index=index):pre[index]=inputs[0].detach().clone()
            handles.append(layer.norm3.register_forward_pre_hook(before_norm))
        try:outputs=self.detector(samples)
        finally:
            for handle in handles:handle.remove()
        raw=captured['raw'];layers=list(outputs.get('aux_outputs',[]))+[outputs]
        if raw.ndim!=4 or len(layers)!=raw.shape[0] or len(pre)!=raw.shape[0]:
            raise RuntimeError('Missing own-layer features/boxes; never substitute final-layer predictions')
        for index,prediction in enumerate(layers):
            if 'penu_layer' not in prediction or not torch.equal(prediction['penu_layer'],raw[index]):
                raise RuntimeError(f'Query/auxiliary prediction mismatch at layer {index+1}')
            if prediction['pred_boxes'].shape[:2]!=raw.shape[1:3] or prediction['pred_obj'].shape!=raw.shape[1:3]:
                raise RuntimeError('Query count/score shape mismatch')
        batch=QueryBatch(raw,torch.stack([pre[i] for i in range(len(pre))]),
            torch.stack([x['pred_boxes'].detach().clone() for x in layers]),
            torch.stack([x['pred_logits'].detach().clone() for x in layers]),
            torch.stack([x['pred_obj'].detach().clone() for x in layers]))
        return batch,outputs


def selection(logits,invalid_classes,pred_per_image=100):
    """Exact OWOBJ global query/class TopK, including repeated query IDs."""
    masked=logits.detach().clone();masked[...,invalid_classes]=-10e10
    probabilities=masked.sigmoid();batch,queries,classes=probabilities.shape
    indices=probabilities.reshape(batch,-1).topk(pred_per_image,dim=1).indices
    query_ids=indices//classes;labels=indices%classes
    scores=probabilities.reshape(batch,-1).gather(1,indices)
    selected=torch.zeros(batch,queries,dtype=torch.bool,device=logits.device)
    selected.scatter_(1,query_ids,True)
    return {'query_ids':query_ids,'labels':labels,'scores':scores,'selected_unique':selected}


def image_frames(batch,geometry,invalid_classes,temperature,pred_per_image=100,layers=None):
    layers=preferred_layers(batch.raw.shape[0]) if layers is None else layers
    if not layers or any(index<1 or index>batch.raw.shape[0] for index in layers):raise ValueError('Layer out of range')
    chosen=torch.tensor([index-1 for index in layers],device=batch.raw.device)
    selected=[selection(batch.logits[index],invalid_classes,pred_per_image) for index in range(batch.raw.shape[0])]
    for b,image in enumerate(geometry):
        height,width=image.original_hw
        boxes=batch.boxes[:,b]
        xyxy=torch.cat([boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2],-1)
        xyxy=xyxy*xyxy.new_tensor([width,height,width,height])
        frame={'image_id':image.image_id,'original_hw':image.original_hw,'resized_hw':image.resized_hw,
            'padded_hw':image.padded_hw,'layer_ids':layers,'decoder_layer_count':batch.raw.shape[0],
            'raw':batch.raw[chosen,b].cpu().numpy(),
            'l2':F.normalize(batch.raw[chosen,b],dim=-1).cpu().numpy(),
            'pre_layernorm':batch.pre_layernorm[chosen,b].cpu().numpy(),
            'boxes_cxcywh':batch.boxes[chosen,b].cpu().numpy(),'boxes_xyxy':xyxy[chosen].cpu().numpy(),
            'logits':batch.logits[chosen,b].cpu().numpy(),
            'gaussian_distance':batch.distance[chosen,b].cpu().numpy(),
            'gaussian_objectness':torch.exp(-temperature*batch.distance[chosen,b]).cpu().numpy(),
            'classifier_unknown':batch.logits[chosen,b,:,-1].sigmoid().cpu().numpy(),
            'selected_unique':torch.stack([selected[index-1]['selected_unique'][b] for index in layers]).cpu().numpy(),
            'selected_pair_query_ids':torch.stack([selected[index-1]['query_ids'][b] for index in layers]).cpu().numpy(),
            'selected_pair_labels':torch.stack([selected[index-1]['labels'][b] for index in layers]).cpu().numpy(),
            'selected_pair_scores':torch.stack([selected[index-1]['scores'][b] for index in layers]).cpu().numpy(),
            'final_pair_query_ids':selected[-1]['query_ids'][b].cpu().numpy(),
            'final_pair_labels':selected[-1]['labels'][b].cpu().numpy(),
            'final_pair_scores':selected[-1]['scores'][b].cpu().numpy(),
            'final_boxes_xyxy':xyxy[-1,selected[-1]['query_ids'][b]].cpu().numpy()}
        yield frame
