"""Read-only forward hooks. No alternate forward path and no changed outputs.

Capture is deliberately opt-in and must be used in model.eval() with no_grad().
Tensors remain on their current device; export only selected examples.
"""
import torch


class FeatureCapture:
    def __init__(self, model):
        self.model = model
        self.data = {}
        self.handles = []

    def __enter__(self):
        assert not self.model.training, 'Offline feature capture requires model.eval()'
        def backbone(module, inputs, output):
            levels, positions = output
            self.data['backbone'] = [x.tensors.detach() for x in levels]
            self.data['backbone_masks'] = [x.mask.detach() for x in levels]
        def transformer_in(module, inputs):
            srcs, masks, positions = inputs[:3]
            self.data['projected'] = [x.detach() for x in srcs]
            self.data['projected_masks'] = [x.detach() for x in masks]
            self.data['spatial_shapes'] = [list(x.shape[-2:]) for x in srcs]
        def encoder(module, inputs, output):
            self.data['encoder'] = output.detach()  # [B,sum(H_l*W_l),256]
        def decoder(module, inputs, output):
            self.data['decoder_queries'] = output[0].detach()  # [L,B,Q,256]
            self.data['decoder_references'] = output[1].detach()
        def predictions(module, inputs, output):
            layers = list(output.get('aux_outputs', [])) + [output]
            self.data['layer_boxes'] = torch.stack([x['pred_boxes'].detach() for x in layers])
            self.data['layer_logits'] = torch.stack([x['pred_logits'].detach() for x in layers])
            self.data['layer_distance'] = torch.stack([x['pred_obj'].detach() for x in layers])
        self.handles = [
            self.model.backbone.register_forward_hook(backbone),
            self.model.transformer.register_forward_pre_hook(transformer_in),
            self.model.transformer.encoder.register_forward_hook(encoder),
            self.model.transformer.decoder.register_forward_hook(decoder),
            self.model.register_forward_hook(predictions),
        ]
        return self

    def shapes(self):
        def shape(value):
            if torch.is_tensor(value):
                return list(value.shape)
            if isinstance(value,list):
                return [shape(x) for x in value]
            return value
        return {key:shape(value) for key,value in self.data.items()}

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
