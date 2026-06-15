from __future__ import annotations

import torch

from callclarity.dsp.envelope import db_to_linear, linear_to_db, rms_dbfs


def static_compress(
    waveform: torch.Tensor,
    threshold_dbfs: float = -22.0,
    ratio: float = 2.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    level = rms_dbfs(waveform)
    if level <= threshold_dbfs:
        return waveform, {"compressor_gain_reduction_db": 0.0}
    over = level - threshold_dbfs
    reduced_over = over / max(float(ratio), 1.0)
    gain_db = reduced_over - over
    return waveform * db_to_linear(gain_db), {"compressor_gain_reduction_db": gain_db}


def apply_gain_db(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    return waveform * db_to_linear(float(gain_db))
