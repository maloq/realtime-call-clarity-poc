from __future__ import annotations

import torch
from torch import nn


class NeuralRateTcn(nn.Module):
    def __init__(self, n_mels: int = 40, channels: int = 48, layers: int = 4) -> None:
        super().__init__()
        blocks = []
        in_ch = n_mels
        for idx in range(layers):
            dilation = 2**idx
            blocks.append(nn.ConstantPad1d((2 * dilation, 0), 0.0))
            blocks.append(nn.Conv1d(in_ch, channels, kernel_size=3, dilation=dilation))
            blocks.append(nn.ReLU())
            in_ch = channels
        self.encoder = nn.Sequential(*blocks)
        self.fast_head = nn.Conv1d(channels, 1, 1)
        self.rate_head = nn.Conv1d(channels, 1, 1)
        self.speech_head = nn.Conv1d(channels, 1, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(features)
        return {
            "fast_speech_probability": torch.sigmoid(self.fast_head(z)),
            "syllables_per_sec": torch.relu(self.rate_head(z)),
            "speech_probability": torch.sigmoid(self.speech_head(z)),
        }
