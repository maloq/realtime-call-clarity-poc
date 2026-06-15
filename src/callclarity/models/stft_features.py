from __future__ import annotations

import torch


def log_mag_stft(waveform: torch.Tensor, n_fft: int = 256, hop_length: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if waveform.ndim == 3:
        waveform = waveform.mean(dim=1)
    elif waveform.ndim == 2 and waveform.shape[0] <= 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    window = torch.hann_window(n_fft, device=waveform.device)
    spec = torch.stft(
        waveform.float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.log1p(spec.abs()), spec


def istft_from_mag_phase(masked_mag: torch.Tensor, noisy_spec: torch.Tensor, length: int, hop_length: int) -> torch.Tensor:
    phase = noisy_spec / (noisy_spec.abs() + 1e-8)
    spec = masked_mag * phase
    n_fft = (spec.shape[-2] - 1) * 2
    window = torch.hann_window(n_fft, device=spec.device)
    return torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=True, length=length)
