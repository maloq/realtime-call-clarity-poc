from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class _TinyBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.act(self.depthwise(x)))


class TinyDecrackler(nn.Module):
    """Small causal waveform TCN that predicts a bounded residual repair."""

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 5,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        max_correction: float = 0.75,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.max_correction = float(max_correction)
        self.input = _CausalConv1d(1, self.channels, self.kernel_size, 1)
        self.blocks = nn.Sequential(
            *[_TinyBlock(self.channels, self.kernel_size, dilation) for dilation in self.dilations]
        )
        self.output = nn.Conv1d(self.channels, 1, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * (1 + sum(self.dilations))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = waveform.float()
        squeeze = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got shape {tuple(x.shape)}")
        feats = self.input(x.unsqueeze(1))
        feats = self.blocks(feats)
        correction = torch.tanh(self.output(feats).squeeze(1)) * self.max_correction
        enhanced = torch.clamp(x + correction, -1.0, 1.0)
        if squeeze:
            return enhanced.squeeze(0), correction.squeeze(0)
        return enhanced, correction

    def config_dict(self) -> dict[str, object]:
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "max_correction": self.max_correction,
        }
