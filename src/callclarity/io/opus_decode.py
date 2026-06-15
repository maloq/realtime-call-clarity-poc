from __future__ import annotations

from pathlib import Path

import torch

from callclarity.io.audio_io import load_audio


def decode_opus(path: str | Path, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    return load_audio(path, target_sample_rate=sample_rate, channels="mono")
