from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from callclarity.metrics.operational import _spectral_metrics
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("bandwidth", "guarded_exciter")
class GuardedBandwidthExciter(BaseStreamingProcessor):
    name = "guarded_exciter"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.sos: np.ndarray | None = None
        self.state: np.ndarray | None = None

    def reset(self) -> None:
        self.sample_rate = None
        self.channels = None
        self.sos = None
        self.state = None

    def _configure(self, sample_rate: int, channels: int) -> None:
        if self.sample_rate == sample_rate and self.channels == channels:
            return
        self.sample_rate = sample_rate
        self.channels = channels
        cutoff = min(float(self.config.get("synth_highpass_hz", 3600.0)), sample_rate / 2.0 - 10.0)
        self.sos = signal.butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos")
        zi = signal.sosfilt_zi(self.sos)
        self.state = zi[:, None, :].repeat(channels, axis=1)

    def process(self, chunk: AudioChunk) -> ProcessResult:
        spectral = _spectral_metrics(chunk.samples, chunk.sample_rate)
        enabled = bool(self.config.get("enabled", False))
        threshold = float(self.config.get("narrowband_threshold", 0.65))
        narrowband_score = float(spectral["narrowband_likelihood"])
        should_apply = enabled and chunk.sample_rate >= 12000 and narrowband_score >= threshold
        if not should_apply:
            metadata = merge_metadata(
                chunk,
                narrowband_score=narrowband_score,
                bandwidth_extension_applied=False,
            )
            return ProcessResult(
                chunk=AudioChunk(
                    chunk.samples,
                    chunk.sample_rate,
                    chunk.start_time_sec,
                    chunk.stream_id,
                    metadata,
                ),
                metrics={
                    "bandwidth_extension_enabled": enabled,
                    "bandwidth_extension_applied": False,
                    "narrowband_score": narrowband_score,
                    "high_frequency_energy_ratio": spectral["high_frequency_energy_ratio"],
                },
                algorithmic_latency_ms=0.0,
            )

        x = chunk.samples.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self._configure(chunk.sample_rate, int(x.shape[0]))
        assert self.sos is not None
        assert self.state is not None
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        harmonic = np.tanh(float(self.config.get("drive", 1.8)) * x_np) - x_np
        high, self.state = signal.sosfilt(self.sos, harmonic, axis=-1, zi=self.state)
        strength = float(self.config.get("strength", 0.06))
        y_np = x_np + strength * high
        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32)).to(
            device=chunk.samples.device,
            dtype=chunk.samples.dtype,
        )
        y = y.clamp(-1.0, 1.0).contiguous()
        metadata = merge_metadata(chunk, narrowband_score=narrowband_score, bandwidth_extension_applied=True)
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "bandwidth_extension_enabled": enabled,
                "bandwidth_extension_applied": True,
                "narrowband_score": narrowband_score,
                "bandwidth_extension_strength": strength,
                "high_frequency_energy_ratio": spectral["high_frequency_energy_ratio"],
            },
            algorithmic_latency_ms=0.0,
        )
