from __future__ import annotations

import torch

from callclarity.dsp.envelope import db_to_linear, linear_to_db


def limit_peak(waveform: torch.Tensor, ceiling_dbfs: float = -1.5) -> tuple[torch.Tensor, dict[str, float]]:
    ceiling = db_to_linear(float(ceiling_dbfs))
    peak = float(waveform.detach().abs().max().item()) if waveform.numel() else 0.0
    if peak <= ceiling or peak <= 0.0:
        return waveform, {
            "peak_dbfs": linear_to_db(max(peak, 1e-12)),
            "clipping_count": int((waveform.detach().abs() >= 1.0).sum().item()),
            "limiter_gain_reduction_db": 0.0,
        }
    gain = ceiling / peak
    limited = waveform * gain
    return limited, {
        "peak_dbfs": linear_to_db(ceiling),
        "clipping_count": int((limited.detach().abs() >= ceiling).sum().item()),
        "limiter_gain_reduction_db": linear_to_db(gain),
    }
