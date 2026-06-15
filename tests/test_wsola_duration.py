import torch

from callclarity.dsp.wsola import wsola_time_scale


def test_wsola_output_duration_approximately_matches_tempo():
    sr = 16000
    waveform = torch.sin(2 * torch.pi * 220 * torch.arange(sr).float() / sr).unsqueeze(0)
    out = wsola_time_scale(waveform, sr, tempo=0.9)
    expected = int(round(waveform.shape[-1] / 0.9))
    assert abs(out.shape[-1] - expected) <= 2
