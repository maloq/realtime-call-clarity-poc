from __future__ import annotations

import torch
from torch import nn

from callclarity.models.stft_features import istft_from_mag_phase, log_mag_stft


class TinyMaskGru(nn.Module):
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        hidden_size: int = 96,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        bins = n_fft // 2 + 1
        self.gru = nn.GRU(bins, hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, bins)

    def forward(
        self,
        waveform: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        length = int(waveform.shape[-1])
        feats, spec = log_mag_stft(waveform, self.n_fft, self.hop_length)
        feats_bt = feats.transpose(1, 2)
        encoded, hidden_out = self.gru(feats_bt, hidden)
        mask = torch.sigmoid(self.proj(encoded)).transpose(1, 2)
        masked_mag = spec.abs() * mask
        enhanced = istft_from_mag_phase(masked_mag, spec, length=length, hop_length=self.hop_length)
        return enhanced, hidden_out, mask
