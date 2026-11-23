"""Original-image boxes -> resized image -> nominal feature stride -> ROIAlign."""
from dataclasses import dataclass
import torch
from torchvision.ops import roi_align


@dataclass
class RegionScores:
    scores: torch.Tensor
    valid: torch.Tensor
    valid_fraction: torch.Tensor
    small_feature_roi: torch.Tensor
    clipped_original_boxes: torch.Tensor


def mean_region_error(error, support, boxes_original, geometry, stride,
                      output_size=7, sampling_ratio=2, min_box_size=1.0, min_valid_fraction=0.5):
    if error.ndim != 2 or support.shape != error.shape:
        raise ValueError('Expected matching HW error and support maps')
    if stride <= 0:
        raise ValueError('stride must be positive')
    boxes = torch.as_tensor(boxes_original, device=error.device, dtype=error.dtype).reshape(-1, 4).clone()
    n = len(boxes)
    scores = error.new_full((n,), float('nan'))
    fraction = error.new_zeros(n)
    small = torch.zeros(n, device=error.device, dtype=torch.bool)
    original_h, original_w = geometry.original_hw
    resized_h, resized_w = geometry.resized_hw
    finite = torch.isfinite(boxes).all(1)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, original_w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, original_h)
    resized = boxes * error.new_tensor([resized_w/original_w, resized_h/original_h] * 2)
    widths = resized[:, 2] - resized[:, 0]
    heights = resized[:, 3] - resized[:, 1]
    valid = finite & (widths >= min_box_size) & (heights >= min_box_size)
    small[:] = (widths < stride) | (heights < stride)
    selected = valid.nonzero(as_tuple=True)[0]
    if len(selected):
        feature_boxes = resized[selected] / float(stride)
        support = support.to(error.dtype)
        # Sampling numerator and support together excludes padded values even
        # for fractional bins touching the valid/padded boundary.
        planes = torch.stack([error * support, support])[None]
        pooled = roi_align(planes, [feature_boxes], (output_size, output_size),
                           spatial_scale=1.0, sampling_ratio=sampling_ratio, aligned=True)
        denominator = pooled[:, 1].sum((1, 2))
        fraction[selected] = pooled[:, 1].mean((1, 2))
        good = (fraction[selected] >= min_valid_fraction) & (denominator > 0)
        scores[selected[good]] = pooled[good, 0].sum((1, 2)) / denominator[good]
        valid[selected] = good
    return RegionScores(scores, valid, fraction, small, boxes)
