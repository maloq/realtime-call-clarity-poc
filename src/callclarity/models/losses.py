from __future__ import annotations

import torch


def magnitude_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = torch.sum(pred * target, dim=-1, keepdim=True) / (torch.sum(target**2, dim=-1, keepdim=True) + 1e-8)
    projection = scale * target
    noise = pred - projection
    ratio = torch.sum(projection**2, dim=-1) / (torch.sum(noise**2, dim=-1) + 1e-8)
    return -10.0 * torch.log10(ratio + 1e-8).mean()
