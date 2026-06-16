from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import signal

from callclarity.config import compose_config
from callclarity.methods.enhance import dpdfnet_remaster as remaster_module
from callclarity.methods.enhance.dpdfnet_remaster import DpdfnetRemasterEnhancer
from callclarity.registry import create_method
from callclarity.types import AudioChunk, ProcessResult


class FakeDpdfnetDenoiser:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.reset_calls = 0
        self.last_wet: torch.Tensor | None = None

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(self.config.get("algorithmic_latency_ms", 20.0))

    @property
    def lookahead_ms(self) -> float:
        return float(self.config.get("lookahead_ms", 10.0))

    def reset(self) -> None:
        self.reset_calls += 1
        self.last_wet = None

    def warmup(self, sample_rate: int) -> None:
        del sample_rate

    def process(self, chunk: AudioChunk) -> ProcessResult:
        x = chunk.samples.detach().float()
        original_shape = x.shape
        x_2d = x.unsqueeze(0) if x.ndim == 1 else x.reshape(-1, x.shape[-1])
        wet_np = []
        for channel in x_2d.cpu().numpy().astype(np.float32, copy=False):
            sos = signal.butter(4, 3600.0, btype="lowpass", fs=chunk.sample_rate, output="sos")
            wet_np.append((0.24 * signal.sosfilt(sos, channel)).astype(np.float32))
        wet = torch.from_numpy(np.stack(wet_np, axis=0)).to(
            device=chunk.samples.device,
            dtype=chunk.samples.dtype,
        )
        wet = wet.reshape(original_shape).contiguous()
        self.last_wet = wet.detach().clone()
        metadata = dict(chunk.metadata)
        metadata.update(denoised=True, neural_denoiser="DPDFNet")
        return ProcessResult(
            chunk=AudioChunk(wet, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={"neural_denoiser": "DPDFNet", "dpdfnet_model": "fake"},
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )


def _patch_fake_denoiser(monkeypatch) -> None:
    monkeypatch.setattr(remaster_module, "DpdfnetDenoiser", FakeDpdfnetDenoiser)


def _config(**updates: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "denoise": {
            "enabled": True,
            "model": "dpdfnet2",
            "algorithmic_latency_ms": 20,
            "lookahead_ms": 10,
        },
        "analysis": {"speech_prob_threshold": 0.35, "min_active_rms_dbfs": -58.0},
        "leveler": {
            "enabled": True,
            "target_rms_dbfs": -22.4,
            "max_boost_db": 14.0,
            "max_cut_db": 8.0,
            "attack_ms": 35.0,
            "release_ms": 260.0,
            "compressor": {
                "enabled": True,
                "threshold_dbfs": -22.0,
                "ratio": 2.2,
                "attack_ms": 10.0,
                "release_ms": 90.0,
            },
        },
        "tone": {
            "highpass_hz": 55.0,
            "body_band_hz": [120.0, 720.0],
            "body_boost_db": 5.2,
            "mud_guard_dbfs": -18.0,
            "lower_presence_band_hz": [850.0, 2400.0],
            "lower_presence_boost_db": -0.8,
            "phone_band_hz": [2000.0, 4800.0],
            "phone_cut_db": 7.0,
            "phone_dominance_db": 11.0,
            "max_phone_cut_db": 10.0,
            "deess_band_hz": [5200.0, 7600.0],
            "deess_threshold_dbfs": -47.0,
            "max_deess_cut_db": 5.0,
            "attack_ms": 25.0,
            "release_ms": 220.0,
            "phone_attack_ms": 8.0,
            "phone_release_ms": 160.0,
            "deess_attack_ms": 3.0,
            "deess_release_ms": 90.0,
        },
        "exciter": {
            "enabled": True,
            "source_band_hz": [1000.0, 3400.0],
            "target_band_hz": [4200.0, 7600.0],
            "envelope_lowpass_hz": 65.0,
            "drive": 2.4,
            "harmonic_mix": 0.72,
            "noise_mix": 0.28,
            "target_high_relative_to_source_db": -16.0,
            "max_high_band_dbfs": -46.0,
            "min_gap_db": 2.0,
            "full_mix_gap_db": 12.0,
            "max_exciter_gain_db": 34.0,
            "attack_ms": 8.0,
            "release_ms": 120.0,
        },
        "density": {
            "enabled": True,
            "band_hz": [120.0, 2600.0],
            "detail_lowpass_hz": 2600.0,
            "drive": 1.25,
            "mix": 0.055,
            "disable_above_peak": 0.92,
        },
        "final": {"dry_wet": 1.0, "ceiling": 0.98},
    }
    cfg.update(updates)
    return cfg


def _speech_like_signal(num_samples: int = 16000, sample_rate: int = 16000) -> torch.Tensor:
    t = torch.arange(num_samples, dtype=torch.float32) / float(sample_rate)
    x = 0.08 * torch.sin(2.0 * torch.pi * 180.0 * t)
    x += 0.05 * torch.sin(2.0 * torch.pi * 620.0 * t)
    x += 0.04 * torch.sin(2.0 * torch.pi * 1450.0 * t)
    x += 0.03 * torch.sin(2.0 * torch.pi * 2900.0 * t)
    window = torch.hann_window(160, periodic=False)
    for start in range(1200, num_samples - 160, 2400):
        burst_t = torch.arange(160, dtype=torch.float32) / float(sample_rate)
        x[start : start + 160] += 0.08 * torch.sin(2.0 * torch.pi * 3200.0 * burst_t) * window
    return x.unsqueeze(0)


def _rms(samples: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(samples.detach().float() ** 2) + 1e-12))


def _band_rms(samples: torch.Tensor, band: tuple[float, float], sample_rate: int = 16000) -> float:
    sos = signal.butter(4, band, btype="bandpass", fs=sample_rate, output="sos")
    x = samples.detach().cpu().float().numpy()
    y = signal.sosfilt(sos, x, axis=-1)
    return float(np.sqrt(np.mean(np.square(y)) + 1e-12))


def test_disabled_returns_passthrough_exactly() -> None:
    processor = create_method("enhance", "dpdfnet_remaster", {"enabled": False})
    samples = torch.randn(1, 160)
    chunk = AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9})

    result = processor.process(chunk)

    assert result.chunk is chunk
    assert result.chunk.samples is samples
    assert result.algorithmic_latency_ms == 0.0


def test_remaster_is_louder_and_adds_synthetic_highband(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetRemasterEnhancer(_config())
    dry = _speech_like_signal()

    result = processor.process(AudioChunk(dry, 16000, 0.0, metadata={"speech_prob": 0.95}))
    assert isinstance(processor.denoiser, FakeDpdfnetDenoiser)
    assert processor.denoiser.last_wet is not None
    wet = processor.denoiser.last_wet

    assert _rms(result.chunk.samples) > _rms(wet) * 3.0
    assert _band_rms(result.chunk.samples, (4200.0, 7600.0)) > _band_rms(wet, (4200.0, 7600.0)) * 5.0
    assert result.metrics["remaster_level_gain_db"] > 6.0
    assert result.metrics["remaster_exciter_mix"] > 0.0
    assert result.chunk.metadata["neural_denoiser"] == "DPDFNet+remaster"
    assert result.chunk.metadata["dpdfnet_remaster"] is True


def test_inactive_signal_does_not_get_level_or_exciter_boost(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetRemasterEnhancer(_config())
    samples = torch.randn(1, 16000) * 0.001

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.0}))

    assert result.metrics["remaster_speech_active"] is False
    assert abs(result.metrics["remaster_level_gain_db"]) < 1e-6
    assert abs(result.metrics["remaster_exciter_mix"]) < 1e-6


def test_output_length_shape_dtype_and_finiteness_are_preserved(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetRemasterEnhancer(_config())
    samples = _speech_like_signal(513).repeat(2, 1).to(dtype=torch.float64)

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert result.chunk.samples.shape == samples.shape
    assert result.chunk.samples.shape[-1] == samples.shape[-1]
    assert result.chunk.samples.dtype == samples.dtype
    assert torch.isfinite(result.chunk.samples).all()


def test_reset_clears_denoiser_and_remaster_state(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetRemasterEnhancer(_config())
    samples = _speech_like_signal()
    processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95}))

    assert processor.body is not None
    assert processor.level_gain_db > 0.0
    assert processor.exciter_mix > 0.0

    processor.reset()

    assert processor.body is None
    assert processor.exciter_source is None
    assert processor.noise_state is None
    assert processor.level_gain_db == 0.0
    assert processor.exciter_mix == 0.0
    assert processor.body_boost_db == 0.0
    assert isinstance(processor.denoiser, FakeDpdfnetDenoiser)
    assert processor.denoiser.reset_calls == 1


def test_metrics_keys_exist(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetRemasterEnhancer(_config())
    samples = _speech_like_signal()

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95}))

    expected = {
        "dpdfnet_remaster_enabled",
        "remaster_speech_active",
        "remaster_input_rms_dbfs",
        "remaster_output_rms_dbfs",
        "remaster_level_gain_db",
        "remaster_compressor_cut_db",
        "remaster_body_boost_db",
        "remaster_lower_presence_boost_db",
        "remaster_phone_cut_db",
        "remaster_deess_cut_db",
        "remaster_exciter_gain_db",
        "remaster_exciter_mix",
        "remaster_density_mix",
        "remaster_high_before_dbfs",
        "remaster_output_peak",
        "dpdfnet_model",
    }
    assert expected.issubset(result.metrics.keys())


def test_create_method_works(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)

    processor = create_method("enhance", "dpdfnet_remaster", _config())

    assert isinstance(processor, DpdfnetRemasterEnhancer)
    assert isinstance(processor.denoiser, FakeDpdfnetDenoiser)


def test_pipeline_config_uses_remaster_without_agc() -> None:
    cfg = compose_config(["pipeline=dpdfnet_remaster", "enhance=dpdfnet_remaster"])
    stages = OmegaConf.to_container(cfg.pipeline.stages, resolve=True)

    assert any(stage["type"] == "enhance" and stage["name"] == "dpdfnet_remaster" for stage in stages)
    assert not any(stage["type"] == "leveler" for stage in stages)
