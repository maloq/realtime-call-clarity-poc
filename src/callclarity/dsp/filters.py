from __future__ import annotations

import torch


def preemphasis(waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    if waveform.shape[-1] <= 1:
        return waveform
    first = waveform[..., :1]
    rest = waveform[..., 1:] - coeff * waveform[..., :-1]
    return torch.cat([first, rest], dim=-1)
