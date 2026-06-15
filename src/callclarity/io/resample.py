from __future__ import annotations

import torch

from callclarity.io.audio_io import resample_if_needed


def resample(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    return resample_if_needed(waveform, source_sr, target_sr)
