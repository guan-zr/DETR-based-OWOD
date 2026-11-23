from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class ImageGeometry:
    image_id: str
    original_hw: Tuple[int, int]
    resized_hw: Tuple[int, int]
    padded_hw: Tuple[int, int]

    def __post_init__(self):
        if min(*self.original_hw, *self.resized_hw, *self.padded_hw) <= 0:
            raise ValueError('Image dimensions must be positive')
        if any(a > b for a, b in zip(self.resized_hw, self.padded_hw)):
            raise ValueError('Resized image exceeds padded canvas')


@dataclass
class SpatialBatch:
    levels: List[torch.Tensor]       # [B,C,H_l,W_l], detached
    padding_masks: List[torch.Tensor]  # [B,H_l,W_l], True = padding
    strides: List[int]              # nominal CNN strides, not padded_size / ceil(size)
    geometry: List[ImageGeometry]
    source: str

    def validate(self):
        if not (len(self.levels) == len(self.padding_masks) == len(self.strides)):
            raise ValueError('Inconsistent level counts')
        for feature, mask in zip(self.levels, self.padding_masks):
            if feature.ndim != 4 or mask.dtype != torch.bool:
                raise ValueError('Expected BCHW features and boolean BHW masks')
            if feature.shape[0] != len(self.geometry) or mask.shape != feature.shape[:1] + feature.shape[2:]:
                raise ValueError('Feature, geometry and padding mask disagree')
        return self
