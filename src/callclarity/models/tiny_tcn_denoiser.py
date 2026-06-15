from __future__ import annotations

import torch
from torch import nn


class TinyTcnDenoiser(nn.Module):
    def __init__(self, channels: int = 48, layers: int = 4, kernel_size: int = 5) -> None:
        super().__init__()
        blocks = []
        in_ch = 1
        for idx in range(layers):
            dilation = 2**idx
            pad = dilation * (kernel_size - 1)
            blocks.append(nn.ConstantPad1d((pad, 0), 0.0))
            blocks.append(nn.Conv1d(in_ch, channels, kernel_size, dilation=dilation))
            blocks.append(nn.ReLU())
            in_ch = channels
        blocks.append(nn.Conv1d(channels, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        return torch.tanh(self.net(waveform)).squeeze(1)
