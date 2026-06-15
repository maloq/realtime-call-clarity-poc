from __future__ import annotations

import torch


def synthetic_wave_batch(batch_size: int, samples: int, sample_rate: int = 16000) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(samples).float() / float(sample_rate)
    freqs = torch.linspace(180.0, 420.0, batch_size).unsqueeze(1)
    clean = 0.12 * torch.sin(2.0 * torch.pi * freqs * t.unsqueeze(0))
    clean += 0.04 * torch.sin(2.0 * torch.pi * (freqs * 2.1) * t.unsqueeze(0))
    noisy = clean + 0.03 * torch.randn_like(clean)
    return noisy.clamp(-1, 1), clean.clamp(-1, 1)
