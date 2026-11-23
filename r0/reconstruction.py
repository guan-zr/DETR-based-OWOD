import torch
from torch import nn
import torch.nn.functional as F


def spatial_mask(padding_mask, ratio, generator=None):
    """One Bernoulli mask per spatial position, shared across feature channels."""
    if not 0 <= ratio <= 1:
        raise ValueError('mask ratio must be in [0,1]')
    return (torch.rand(padding_mask.shape, device=padding_mask.device, generator=generator) < ratio) & ~padding_mask


class ConvAutoencoder(nn.Module):
    def __init__(self, channels=256, hidden=64, bottleneck=32, depth=1):
        super().__init__()
        if min(channels, hidden, bottleneck, depth) <= 0:
            raise ValueError('Autoencoder capacity must be positive')
        layers = [nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
                  nn.Conv2d(hidden, bottleneck, 3, stride=2, padding=1), nn.GELU()]
        for _ in range(depth):
            layers += [nn.Conv2d(bottleneck, bottleneck, 3, padding=1), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(nn.Conv2d(bottleneck, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden, channels, 3, padding=1))

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.interpolate(encoded, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.decoder(encoded)


class FMRBank(nn.Module):
    """Independent, equally sized AE per scale; no gradients into its input."""
    def __init__(self, channels=(256, 256, 256, 256), hidden=64, bottleneck=32, depth=1, mask_ratio=0.6):
        super().__init__()
        self.models = nn.ModuleList([ConvAutoencoder(c, hidden, bottleneck, depth) for c in channels])
        self.mask_ratio = mask_ratio

    def forward(self, batch, mask_ratio=None, seed=None):
        batch.validate()
        if len(batch.levels) != len(self.models):
            raise ValueError('Reconstruction scale count mismatch')
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        results = []
        losses = []
        for level, (features, padding, model) in enumerate(zip(batch.levels, batch.padding_masks, self.models)):
            target = features.detach().masked_fill(padding[:, None], 0)
            generator = None if seed is None else torch.Generator(device=features.device).manual_seed(seed + level)
            mask = spatial_mask(padding, ratio, generator)
            masked_input = target.masked_fill(mask[:, None], 0)
            prediction = model(masked_input)
            error = (prediction - target).square().mean(dim=1)
            valid = ~padding
            count = valid.sum()
            loss = (error * valid).sum() / count.clamp(min=1)
            masked_count = mask.sum()
            masked_loss = (error * mask).sum() / masked_count.clamp(min=1) if masked_count.item() else None
            losses.append(loss)
            results.append({'reconstruction': prediction, 'error': error.masked_fill(padding, 0),
                            'valid': valid, 'masked': mask, 'valid_mse': loss,
                            'masked_mse': masked_loss, 'valid_count': int(count),
                            'masked_count': int(masked_count)})
        return torch.stack(losses).mean(), results
