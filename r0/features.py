"""Read-only hooks at the actual official OWOBJ Deformable DETR boundaries."""
import torch
from torch import nn

from r0.types import SpatialBatch


def unflatten_memory(memory, spatial_shapes):
    if memory.ndim != 3:
        raise ValueError('Encoder memory must be [B,S,C]')
    sizes = [int(h) * int(w) for h, w in spatial_shapes]
    if sum(sizes) != memory.shape[1]:
        raise ValueError('Encoder token count does not equal sum(H_l*W_l)')
    return [piece.transpose(1, 2).reshape(memory.shape[0], memory.shape[2], int(h), int(w))
            for piece, (h, w) in zip(memory.split(sizes, dim=1), spatial_shapes)]


class FrozenSpatialExtractor(nn.Module):
    def __init__(self, detector, source='S1'):
        super().__init__()
        if source not in ('S1', 'S2'):
            raise ValueError('source must be S1 or S2')
        self.detector = detector
        self.source = source
        self.detector.requires_grad_(False)
        self.detector.eval()

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        return self

    @torch.no_grad()
    def forward(self, samples, geometry):
        self.detector.eval()
        captured = {}
        def transformer_input(module, inputs):
            srcs, masks = inputs[:2]
            captured['srcs'] = [x.detach() for x in srcs]
            captured['masks'] = [x.detach() for x in masks]
            captured['shapes'] = [tuple(x.shape[-2:]) for x in srcs]
        def encoder_output(module, inputs, output):
            captured['memory'] = output.detach()
        handles = [self.detector.transformer.register_forward_pre_hook(transformer_input)]
        if self.source == 'S2':
            handles.append(self.detector.transformer.encoder.register_forward_hook(encoder_output))
        try:
            outputs = self.detector(samples)
        finally:
            for handle in handles:
                handle.remove()
        if 'srcs' not in captured:
            raise RuntimeError('Official transformer input hook did not fire')
        levels = captured['srcs'] if self.source == 'S1' else unflatten_memory(captured['memory'], captured['shapes'])
        strides = list(self.detector.backbone.strides)
        while len(strides) < len(levels):
            strides.append(strides[-1] * 2)
        if len(strides) != len(levels):
            raise RuntimeError('Backbone stride count disagrees with feature levels')
        # Out-of-place masking: original detector tensors and predictions stay intact.
        levels = [x.masked_fill(mask[:, None], 0).detach() for x, mask in zip(levels, captured['masks'])]
        batch = SpatialBatch(levels, captured['masks'], strides, geometry, self.source).validate()
        return batch, outputs
