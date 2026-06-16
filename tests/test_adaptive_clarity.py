from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import signal

from callclarity.config import compose_config
from callclarity.methods.filter.adaptive_clarity import AdaptiveClarityProcessor
from callclarity.registry import create_method
from callclarity.types import AudioChunk


def _config(**updates):
    cfg = {
        "enabled": True,
        "analysis": {
            "speech_prob_threshold": 0.35,
            "min_active_rms_dbfs": -58.0,
            "smoothing_ms": 120.0,
        },
        "filters": {
            "highpass": {"enabled": True, "cutoff_hz": 80.0, "order": 2},
            "static_eq": {
                "enabled": True,
                "mud_center_hz": 320.0,
                "mud_q": 0.9,
                "mud_gain_db": -1.0,
                "presence_center_hz": 2400.0,
                "presence_q": 0.85,
                "presence_gain_db": 0.9,
            },
            "dynamic_demud": {
                "enabled": True,
                "band_hz": [180.0, 560.0],
                "compare_band_hz": [1200.0, 4200.0],
                "ratio_threshold_db": 5.0,
                "max_cut_db": 3.0,
                "attack_ms": 15.0,
                "release_ms": 180.0,
            },
            "dynamic_presence": {
                "enabled": True,
                "band_hz": [1600.0, 3600.0],
                "reference_band_hz": [250.0, 1200.0],
                "missing_threshold_db": 4.0,
                "max_boost_db": 2.0,
                "attack_ms": 20.0,
                "release_ms": 220.0,
                "noise_guard_high_band_dbfs": -48.0,
            },
            "consonant_lift": {
                "enabled": True,
                "band_hz": [3200.0, 5200.0],
                "max_boost_db": 0.9,
                "transient_extra_db": 0.35,
                "attack_ms": 6.0,
                "release_ms": 100.0,
                "only_when_speech": True,
            },
            "harshness_guard": {
                "enabled": True,
                "band_hz": [2800.0, 5200.0],
                "threshold_dbfs": -39.0,
                "max_cut_db": 5.0,
                "attack_ms": 4.0,
                "release_ms": 130.0,
            },
            "deesser": {
                "enabled": True,
                "band_hz": [5000.0, 7600.0],
                "threshold_dbfs": -52.0,
                "max_cut_db": 6.0,
                "attack_ms": 3.0,
                "release_ms": 100.0,
            },
            "saturation": {
                "enabled": False,
                "drive": 1.25,
                "mix": 0.0,
                "disable_above_peak": 0.92,
            },
        },
        "final": {"dry_wet": 1.0, "ceiling": 0.98},
    }
    cfg.update(updates)
    return cfg


def _deep_update(base, updates):
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _tone(freq_hz: float, duration_sec: float = 1.0, sample_rate: int = 16000) -> torch.Tensor:
    t = torch.arange(int(duration_sec * sample_rate), dtype=torch.float32) / float(sample_rate)
    return torch.sin(2.0 * torch.pi * float(freq_hz) * t)


def _muffled_speech_like(sample_rate: int = 16000) -> torch.Tensor:
    t = torch.arange(sample_rate, dtype=torch.float32) / float(sample_rate)
    voiced = 0.13 * torch.sin(2.0 * torch.pi * 170.0 * t)
    voiced += 0.16 * torch.sin(2.0 * torch.pi * 320.0 * t)
    voiced += 0.06 * torch.sin(2.0 * torch.pi * 720.0 * t)
    detail = 0.012 * torch.sin(2.0 * torch.pi * 2900.0 * t)
    consonant = 0.008 * torch.sin(2.0 * torch.pi * 4600.0 * t)
    return (voiced + detail + consonant).unsqueeze(0)


def _harsh_speech_like(sample_rate: int = 16000) -> torch.Tensor:
    t = torch.arange(sample_rate, dtype=torch.float32) / float(sample_rate)
    x = 0.06 * torch.sin(2.0 * torch.pi * 180.0 * t)
    x += 0.22 * torch.sin(2.0 * torch.pi * 3600.0 * t)
    x += 0.19 * torch.sin(2.0 * torch.pi * 6200.0 * t)
    return x.unsqueeze(0)


def _band_rms(samples: torch.Tensor, band: tuple[float, float], sample_rate: int = 16000) -> float:
    sos = signal.butter(4, band, btype="bandpass", fs=sample_rate, output="sos")
    x = samples.detach().cpu().float().numpy()
    y = signal.sosfilt(sos, x, axis=-1)
    return float(np.sqrt(np.mean(np.square(y)) + 1e-12))


def _band_ratio_db(
    samples: torch.Tensor,
    numerator: tuple[float, float],
    denominator: tuple[float, float],
    sample_rate: int = 16000,
) -> float:
    return 20.0 * np.log10(
        _band_rms(samples, numerator, sample_rate)
        / max(_band_rms(samples, denominator, sample_rate), 1e-12)
    )


def test_disabled_returns_passthrough_exactly() -> None:
    processor = create_method("filter", "adaptive_clarity", {"enabled": False})
    samples = torch.randn(1, 160)
    chunk = AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 1.0})

    result = processor.process(chunk)

    assert result.chunk is chunk
    assert result.chunk.samples is samples
    assert result.algorithmic_latency_ms == 0.0


def test_muffled_speech_gets_more_presence_and_consonant_ratio() -> None:
    processor = AdaptiveClarityProcessor(_config())
    samples = _muffled_speech_like()
    chunk = AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95})

    result = processor.process(chunk)

    before = _band_ratio_db(samples, (1800.0, 6800.0), (250.0, 1200.0))
    after = _band_ratio_db(result.chunk.samples, (1800.0, 6800.0), (250.0, 1200.0))
    assert after > before + 1.0
    assert result.metrics["presence_boost_db"] > 0.0 or result.metrics["consonant_boost_db"] > 0.0
    assert result.chunk.metadata["adaptive_clarity"] is True


def test_harsh_signal_triggers_harshness_and_deesser_reductions() -> None:
    processor = AdaptiveClarityProcessor(_config())
    samples = _harsh_speech_like()

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert result.metrics["harshness_cut_db"] > 0.0
    assert result.metrics["deess_cut_db"] > 0.0


def test_noise_only_does_not_receive_clarity_boosts() -> None:
    processor = AdaptiveClarityProcessor(_config())
    generator = torch.Generator().manual_seed(1337)
    noise = torch.randn(1, 16000, generator=generator) * 0.001

    result = processor.process(AudioChunk(noise, 16000, 0.0, metadata={"speech_prob": 0.0}))

    assert result.metrics["clarity_speech_active"] is False
    assert abs(result.metrics["presence_boost_db"]) < 1e-6
    assert abs(result.metrics["consonant_boost_db"]) < 1e-6


def test_output_length_shape_and_dtype_are_preserved() -> None:
    processor = AdaptiveClarityProcessor(_config())
    samples = _muffled_speech_like(16000).repeat(2, 1).to(dtype=torch.float64)

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert result.chunk.samples.shape == samples.shape
    assert result.chunk.samples.dtype == samples.dtype
    assert result.chunk.samples.shape[-1] == samples.shape[-1]


def test_output_has_no_nan_or_inf() -> None:
    processor = AdaptiveClarityProcessor(_config())
    samples = _muffled_speech_like()

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert torch.isfinite(result.chunk.samples).all()


def test_reset_clears_dynamic_gains_and_filter_states() -> None:
    processor = AdaptiveClarityProcessor(_config())
    samples = _muffled_speech_like()
    processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95}))

    assert processor.consonant_boost_db > 0.0
    assert processor.highpass is not None
    assert processor.consonant is not None

    processor.reset()

    assert processor.highpass is None
    assert processor.demud is None
    assert processor.presence is None
    assert processor.consonant is None
    assert processor.harshness is None
    assert processor.deess is None
    assert processor.demud_cut_db == 0.0
    assert processor.presence_boost_db == 0.0
    assert processor.consonant_boost_db == 0.0
    assert processor.harshness_cut_db == 0.0
    assert processor.deess_cut_db == 0.0


def test_create_method_works() -> None:
    processor = create_method("filter", "adaptive_clarity", _config())

    assert isinstance(processor, AdaptiveClarityProcessor)


def test_pipeline_configs_include_adaptive_clarity_stage() -> None:
    for pipeline_name in ["dpdfnet_clarity", "dpdfnet_detail_rescue_clarity"]:
        cfg = compose_config([f"pipeline={pipeline_name}", "denoise=dpdfnet"])
        stages = OmegaConf.to_container(cfg.pipeline.stages, resolve=True)

        assert any(
            stage["type"] == "filter" and stage["name"] == "adaptive_clarity"
            for stage in stages
        )
        assert not any(stage["type"] == "leveler" for stage in stages)
