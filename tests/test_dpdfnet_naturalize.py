from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import signal

from callclarity.config import compose_config
from callclarity.methods.filter.dpdfnet_naturalize import DpdfnetNaturalizeProcessor
from callclarity.registry import create_method
from callclarity.types import AudioChunk


def _config(**updates: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "analysis": {
            "speech_prob_threshold": 0.35,
            "min_active_rms_dbfs": -58.0,
        },
        "filters": {
            "highpass": {"enabled": True, "cutoff_hz": 65.0, "order": 2},
            "body": {
                "enabled": True,
                "band_hz": [140.0, 420.0],
                "reference_band_hz": [900.0, 2400.0],
                "base_boost_db": 1.6,
                "max_boost_db": 3.2,
                "thinness_threshold_db": 5.0,
                "attack_ms": 30.0,
                "release_ms": 240.0,
            },
            "lower_presence": {
                "enabled": True,
                "band_hz": [850.0, 2200.0],
                "reference_band_hz": [180.0, 900.0],
                "base_boost_db": 0.6,
                "max_boost_db": 1.5,
                "missing_threshold_db": 3.0,
                "attack_ms": 35.0,
                "release_ms": 260.0,
            },
            "phone_soften": {
                "enabled": True,
                "band_hz": [2400.0, 4200.0],
                "reference_band_hz": [140.0, 1200.0],
                "base_cut_db": 0.5,
                "dominance_threshold_db": 8.0,
                "max_cut_db": 2.4,
                "attack_ms": 12.0,
                "release_ms": 180.0,
            },
            "metal_smoother": {
                "enabled": True,
                "band_hz": [3600.0, 7600.0],
                "floor_dbfs": -90.0,
                "base_cut_db": 0.9,
                "threshold_dbfs": -62.0,
                "max_cut_db": 3.0,
                "attack_ms": 8.0,
                "release_ms": 160.0,
            },
            "deesser": {
                "enabled": True,
                "band_hz": [5200.0, 7600.0],
                "threshold_dbfs": -58.0,
                "max_cut_db": 3.0,
                "attack_ms": 4.0,
                "release_ms": 110.0,
            },
            "density": {
                "enabled": True,
                "band_hz": [120.0, 2400.0],
                "detail_lowpass_hz": 2600.0,
                "drive": 1.15,
                "mix": 0.025,
                "disable_above_peak": 0.92,
            },
        },
        "final": {"dry_wet": 1.0, "output_gain_db": 0.0, "ceiling": 0.98},
    }
    cfg.update(updates)
    return cfg


def _thin_metallic_signal(sample_rate: int = 16000, duration_sec: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / float(sample_rate)
    x = 0.035 * torch.sin(2.0 * torch.pi * 210.0 * t)
    x += 0.02 * torch.sin(2.0 * torch.pi * 340.0 * t)
    x += 0.08 * torch.sin(2.0 * torch.pi * 1450.0 * t)
    x += 0.12 * torch.sin(2.0 * torch.pi * 3100.0 * t)
    x += 0.06 * torch.sin(2.0 * torch.pi * 5200.0 * t)
    return x.unsqueeze(0)


def _sibilant_signal(sample_rate: int = 16000, duration_sec: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / float(sample_rate)
    x = 0.05 * torch.sin(2.0 * torch.pi * 220.0 * t)
    x += 0.06 * torch.sin(2.0 * torch.pi * 1300.0 * t)
    x += 0.18 * torch.sin(2.0 * torch.pi * 6300.0 * t)
    return x.unsqueeze(0)


def _band_rms(samples: torch.Tensor, band: tuple[float, float], sample_rate: int = 16000) -> float:
    sos = signal.butter(4, band, btype="bandpass", fs=sample_rate, output="sos")
    x = samples.detach().cpu().float().numpy()
    y = signal.sosfilt(sos, x, axis=-1)
    return float(np.sqrt(np.mean(np.square(y)) + 1e-12))


def _ratio_db(
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
    processor = create_method("filter", "dpdfnet_naturalize", {"enabled": False})
    samples = torch.randn(1, 160)
    chunk = AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9})

    result = processor.process(chunk)

    assert result.chunk is chunk
    assert result.chunk.samples is samples
    assert result.algorithmic_latency_ms == 0.0


def test_thin_metallic_signal_gets_more_body_and_less_phone_ring() -> None:
    processor = DpdfnetNaturalizeProcessor(_config())
    samples = _thin_metallic_signal()

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95}))

    before_body_vs_phone = _ratio_db(samples, (140.0, 420.0), (2400.0, 4200.0))
    after_body_vs_phone = _ratio_db(result.chunk.samples, (140.0, 420.0), (2400.0, 4200.0))
    assert after_body_vs_phone > before_body_vs_phone + 0.8
    assert result.metrics["body_boost_db"] > 0.0
    assert result.metrics["phone_cut_db"] > 0.0
    assert result.metrics["metal_cut_db"] > 0.0
    assert result.chunk.metadata["dpdfnet_naturalize"] is True


def test_sibilant_signal_triggers_deesser_without_increasing_sibilance() -> None:
    processor = DpdfnetNaturalizeProcessor(_config())
    samples = _sibilant_signal()

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert result.metrics["deess_cut_db"] > 0.0
    assert _band_rms(result.chunk.samples, (5200.0, 7600.0)) <= _band_rms(
        samples,
        (5200.0, 7600.0),
    )


def test_inactive_noise_releases_boosts_and_density() -> None:
    processor = DpdfnetNaturalizeProcessor(_config())
    noise = torch.randn(1, 16000, generator=torch.Generator().manual_seed(5)) * 0.001

    result = processor.process(AudioChunk(noise, 16000, 0.0, metadata={"speech_prob": 0.0}))

    assert result.metrics["naturalize_speech_active"] is False
    assert abs(result.metrics["body_boost_db"]) < 1e-6
    assert abs(result.metrics["lower_presence_boost_db"]) < 1e-6
    assert result.metrics["density_mix"] == 0.0


def test_output_shape_length_dtype_and_finiteness_are_preserved() -> None:
    processor = DpdfnetNaturalizeProcessor(_config())
    samples = _thin_metallic_signal(duration_sec=0.25).repeat(2, 1).to(dtype=torch.float64)

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert result.chunk.samples.shape == samples.shape
    assert result.chunk.samples.shape[-1] == samples.shape[-1]
    assert result.chunk.samples.dtype == samples.dtype
    assert torch.isfinite(result.chunk.samples).all()


def test_reset_clears_dynamic_gains_and_filter_states() -> None:
    processor = DpdfnetNaturalizeProcessor(_config())
    samples = _thin_metallic_signal()
    processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95}))

    assert processor.body is not None
    assert processor.body_boost_db > 0.0
    assert processor.phone_cut_db > 0.0

    processor.reset()

    assert processor.body is None
    assert processor.phone is None
    assert processor.metal is None
    assert processor.deess is None
    assert processor.body_boost_db == 0.0
    assert processor.lower_presence_boost_db == 0.0
    assert processor.phone_cut_db == 0.0
    assert processor.metal_cut_db == 0.0
    assert processor.deess_cut_db == 0.0
    assert processor.density_mix == 0.0


def test_create_method_works() -> None:
    processor = create_method("filter", "dpdfnet_naturalize", _config())

    assert isinstance(processor, DpdfnetNaturalizeProcessor)


def test_pipeline_configs_include_naturalize_without_agc() -> None:
    for pipeline_name in ["dpdfnet_naturalize", "dpdfnet_detail_rescue_naturalize"]:
        cfg = compose_config(
            [
                f"pipeline={pipeline_name}",
                "denoise=dpdfnet",
                "enhance=dpdfnet_detail_rescue",
            ]
        )
        stages = OmegaConf.to_container(cfg.pipeline.stages, resolve=True)

        assert any(
            stage["type"] == "filter" and stage["name"] == "dpdfnet_naturalize"
            for stage in stages
        )
        assert not any(stage["type"] == "leveler" for stage in stages)
