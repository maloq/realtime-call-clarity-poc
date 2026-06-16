from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import signal

from callclarity.config import compose_config
from callclarity.methods.enhance import dpdfnet_detail_rescue as rescue_module
from callclarity.methods.enhance.dpdfnet_detail_rescue import DpdfnetDetailRescueEnhancer
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
        if x.ndim == 1:
            x_2d = x.unsqueeze(0)
        else:
            x_2d = x.reshape(-1, x.shape[-1])

        wet_np = []
        for channel in x_2d.cpu().numpy().astype(np.float32, copy=False):
            freqs = np.fft.rfftfreq(channel.shape[-1], d=1.0 / float(chunk.sample_rate))
            spectrum = np.fft.rfft(channel)
            consonant = (freqs >= 2200.0) & (freqs <= 6800.0)
            spectrum[consonant] *= 0.12
            wet_np.append(np.fft.irfft(spectrum, n=channel.shape[-1]).astype(np.float32))

        wet = torch.from_numpy(np.stack(wet_np, axis=0)).to(
            device=chunk.samples.device,
            dtype=chunk.samples.dtype,
        )
        wet = wet.reshape(original_shape).contiguous()
        self.last_wet = wet.detach().clone()
        metadata = dict(chunk.metadata)
        metadata.update(denoised=True, neural_denoiser="DPDFNet")
        return ProcessResult(
            chunk=AudioChunk(
                wet,
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics={
                "neural_denoiser": "DPDFNet",
                "dpdfnet_model": "fake",
                "pending_samples": 0,
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )


def _patch_fake_denoiser(monkeypatch) -> None:
    monkeypatch.setattr(rescue_module, "DpdfnetDenoiser", FakeDpdfnetDenoiser)


def _config(**updates: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "enabled": True,
        "denoise": {
            "enabled": True,
            "model": "dpdfnet2",
            "algorithmic_latency_ms": 20,
            "lookahead_ms": 10,
        },
        "rescue": {
            "enabled": True,
            "speech_prob_threshold": 0.35,
            "min_input_rms_dbfs": -58.0,
            "consonant_band_hz": [2200.0, 6800.0],
            "presence_band_hz": [1200.0, 4200.0],
            "sibilance_guard_band_hz": [5600.0, 7600.0],
            "max_consonant_restore_mix": 0.42,
            "max_presence_restore_mix": 0.20,
            "restore_when_wet_loses_db": 3.0,
            "onset_boost_mix": 0.18,
            "noise_guard_high_band_dbfs": -45.0,
            "sibilance_guard_min_dbfs": -92.0,
            "sibilance_guard_loss_db": 1.5,
            "sibilance_guard_max_reduction": 0.75,
            "attack_ms": 6.0,
            "release_ms": 80.0,
            "dry_delay_compensation_samples": 0,
        },
        "final": {"ceiling": 0.98},
    }
    config.update(updates)
    return config


def _speech_like_signal(num_samples: int = 16000, sample_rate: int = 16000) -> torch.Tensor:
    t = torch.arange(num_samples, dtype=torch.float32) / float(sample_rate)
    x = 0.08 * torch.sin(2.0 * torch.pi * 180.0 * t)
    x += 0.04 * torch.sin(2.0 * torch.pi * 720.0 * t)
    burst_len = max(12, int(0.018 * sample_rate))
    window = torch.hann_window(burst_len, periodic=False)
    for start in range(sample_rate // 10, num_samples - burst_len, sample_rate // 5):
        burst_t = torch.arange(burst_len, dtype=torch.float32) / float(sample_rate)
        burst = 0.34 * torch.sin(2.0 * torch.pi * 3600.0 * burst_t) * window
        x[start : start + burst_len] += burst
    return x.unsqueeze(0)


def _band_rms(
    samples: torch.Tensor,
    sample_rate: int,
    band: tuple[float, float] = (2200.0, 6800.0),
) -> float:
    sos = signal.butter(4, band, btype="bandpass", fs=sample_rate, output="sos")
    filtered = signal.sosfilt(sos, samples.detach().cpu().float().numpy(), axis=-1)
    return float(np.sqrt(np.mean(np.square(filtered)) + 1e-12))


def test_disabled_returns_passthrough_exactly() -> None:
    processor = create_method("enhance", "dpdfnet_detail_rescue", {"enabled": False})
    samples = torch.randn(1, 160)
    chunk = AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9})

    result = processor.process(chunk)

    assert result.chunk is chunk
    assert result.chunk.samples is samples
    assert result.algorithmic_latency_ms == 0.0


def test_detail_rescue_restores_consonant_band_without_full_dry_blend(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetDetailRescueEnhancer(_config())
    dry = _speech_like_signal()
    chunk = AudioChunk(dry, 16000, 0.0, metadata={"speech_prob": 0.95})

    result = processor.process(chunk)
    assert isinstance(processor.denoiser, FakeDpdfnetDenoiser)
    assert processor.denoiser.last_wet is not None
    wet = processor.denoiser.last_wet

    assert _band_rms(result.chunk.samples, 16000) > _band_rms(wet, 16000) * 1.2
    assert not torch.allclose(result.chunk.samples, dry, atol=1e-3)
    assert result.chunk.metadata["neural_denoiser"] == "DPDFNet+detail_rescue"
    assert result.chunk.metadata["dpdfnet_detail_rescue"] is True


def test_output_length_equals_input_for_several_chunk_sizes(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetDetailRescueEnhancer(_config())

    for size in [80, 160, 257, 511]:
        samples = _speech_like_signal(size)
        result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))
        assert result.chunk.samples.shape[-1] == size


def test_output_shape_and_dtype_are_preserved(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetDetailRescueEnhancer(_config())
    samples = _speech_like_signal(512).to(dtype=torch.float64).repeat(2, 1)

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert result.chunk.samples.shape == samples.shape
    assert result.chunk.samples.dtype == samples.dtype


def test_output_has_no_nan_or_inf(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetDetailRescueEnhancer(_config())
    samples = _speech_like_signal(1024)

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    assert torch.isfinite(result.chunk.samples).all()


def test_reset_clears_filter_states_and_restore_mix(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetDetailRescueEnhancer(_config())
    samples = _speech_like_signal()

    processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.95}))
    assert processor._dry_consonant_state is not None
    assert processor.consonant_restore_mix > 0.0

    processor.reset()

    assert processor._dry_consonant_state is None
    assert processor._wet_consonant_state is None
    assert processor._dry_presence_state is None
    assert processor._wet_presence_state is None
    assert processor._dry_sibilance_state is None
    assert processor._wet_sibilance_state is None
    assert processor.consonant_restore_mix == 0.0
    assert processor.presence_restore_mix == 0.0
    assert processor.onset_restore_mix == 0.0
    assert processor.sibilance_guard == 1.0
    assert processor._prev_dry_consonant_dbfs is None
    assert processor._prev_wet_consonant_dbfs is None
    assert isinstance(processor.denoiser, FakeDpdfnetDenoiser)
    assert processor.denoiser.reset_calls == 1


def test_metrics_keys_exist(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)
    processor = DpdfnetDetailRescueEnhancer(_config())
    samples = _speech_like_signal()

    result = processor.process(AudioChunk(samples, 16000, 0.0, metadata={"speech_prob": 0.9}))

    expected = {
        "dpdfnet_detail_rescue_enabled",
        "speech_prob",
        "input_rms_dbfs",
        "dry_consonant_dbfs",
        "wet_consonant_dbfs",
        "consonant_loss_db",
        "consonant_restore_mix",
        "dry_presence_dbfs",
        "wet_presence_dbfs",
        "presence_loss_db",
        "presence_restore_mix",
        "dry_sibilance_dbfs",
        "wet_sibilance_dbfs",
        "sibilance_loss_db",
        "sibilance_guard",
        "onset_restore_mix",
        "output_peak",
        "dpdfnet_model",
        "dpdfnet_pending_samples",
    }
    assert expected.issubset(result.metrics.keys())


def test_create_method_works(monkeypatch) -> None:
    _patch_fake_denoiser(monkeypatch)

    processor = create_method("enhance", "dpdfnet_detail_rescue", _config())

    assert isinstance(processor, DpdfnetDetailRescueEnhancer)
    assert isinstance(processor.denoiser, FakeDpdfnetDenoiser)


def test_default_pipeline_stays_dpdfnet_centered_without_agc() -> None:
    cfg = compose_config(["pipeline=dpdfnet_detail_rescue", "enhance=dpdfnet_detail_rescue"])
    stages = OmegaConf.to_container(cfg.pipeline.stages, resolve=True)

    assert any(
        stage["type"] == "enhance" and stage["name"] == "dpdfnet_detail_rescue"
        for stage in stages
    )
    assert not any(stage["type"] == "leveler" for stage in stages)
