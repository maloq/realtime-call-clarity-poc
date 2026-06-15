from __future__ import annotations

import torch


def stft_mag(waveform: torch.Tensor, n_fft: int = 512, hop_length: int = 160) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    window = torch.hann_window(n_fft, device=waveform.device)
    spec = torch.stft(
        waveform.float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
    )
    return spec.abs()
