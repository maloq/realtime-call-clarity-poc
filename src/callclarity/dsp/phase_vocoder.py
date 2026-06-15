from __future__ import annotations

import torch

from callclarity.dsp.wsola import wsola_time_scale


def phase_vocoder_time_scale(waveform: torch.Tensor, sample_rate: int, tempo: float) -> torch.Tensor:
    # Placeholder baseline: routes through the same compact TSM kernel with a higher
    # latency declaration in the processor. It keeps the comparison surface ready.
    return wsola_time_scale(waveform, sample_rate, tempo, frame_ms=64, analysis_hop_ms=16)
