import math

import torch

from callclarity.registry import create_method
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk


def _sine(freq_hz: float, seconds: float = 0.1, sample_rate: int = 16000) -> torch.Tensor:
    n = int(round(seconds * sample_rate))
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    return 0.2 * torch.sin(2.0 * math.pi * float(freq_hz) * t).unsqueeze(0)


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2) + 1e-12).item())


def test_warm_rounded_voice_is_causal_and_preserves_duration():
    pipeline = Pipeline.from_config(
        {
            "name": "warm_test",
            "stages": [
                {
                    "type": "tone",
                    "name": "warm_rounded_voice",
                    "config": {"mode": "safe_warm"},
                }
            ],
        }
    )
    waveform = torch.randn(1, 1600) * 0.03

    result = RealtimeSimulator(pipeline, chunk_ms=10, max_added_latency_ms=50).run(waveform, 16000)

    assert result.output.shape == waveform.shape
    assert result.latency_summary["declared_algorithmic_latency_ms"] == 0.0
    assert result.latency_summary["latency_budget_violation_count"] == 0


def test_warm_rounded_dynamic_harsh_cut_targets_presence_band():
    cfg = {
        "mode": "warm_smooth",
        "leveler": {"enabled": False},
        "compressor": {"enabled": False},
        "saturation": {"enabled": False},
        "eq": {
            "warmth": {"enabled": False},
            "mud": {"enabled": False},
            "nasal": {"enabled": False},
            "clarity": {"enabled": False},
        },
        "dynamic_eq": {
            "harsh": {
                "enabled": True,
                "low_hz": 2500.0,
                "high_hz": 4500.0,
                "threshold_dbfs": -80.0,
                "max_reduction_db": 6.0,
                "slope": 1.0,
                "attack_ms": 0.1,
                "release_ms": 100.0,
            },
            "deess": {"enabled": False},
        },
    }
    harsh = _sine(3200.0)
    warm_mid = _sine(500.0)
    harsh_proc = create_method("tone", "warm_rounded_voice", cfg)
    mid_proc = create_method("tone", "warm_rounded_voice", cfg)

    harsh_result = harsh_proc.process(AudioChunk(harsh, 16000, 0.0))
    mid_result = mid_proc.process(AudioChunk(warm_mid, 16000, 0.0))

    assert harsh_result.metrics["harsh_reduction_db"] > 3.0
    assert _rms(harsh_result.chunk.samples) < 0.9 * _rms(harsh)
    assert _rms(mid_result.chunk.samples) > 0.97 * _rms(warm_mid)
