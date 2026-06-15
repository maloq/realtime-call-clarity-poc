from __future__ import annotations

import numpy as np
import torch

from callclarity.dsp.envelope import linear_to_db
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("denoise", "spectral_gate")
class SpectralGateDenoiser(BaseStreamingProcessor):
    name = "spectral_gate"
    realtime_safe = True

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.fft_size = int(self.config.get("fft_size", 512))
        gain_cfg = self.config.get("gain", {})
        self.min_gain = 10.0 ** (float(gain_cfg.get("min_gain_db", -18.0)) / 20.0)
        self.temporal_smoothing = float(gain_cfg.get("temporal_smoothing", 0.85))
        update_cfg = self.config.get("noise_update", {})
        self.noise_alpha = float(update_cfg.get("alpha", 0.98))
        self.vad_threshold = float(update_cfg.get("vad_threshold", 0.35))
        self.oversubtraction = float(self.config.get("oversubtraction", 1.25))
        self.noise_mag: torch.Tensor | None = None
        self.prev_gain: torch.Tensor | None = None

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(self.config.get("hop_ms", 10.0))

    def reset(self) -> None:
        self.noise_mag = None
        self.prev_gain = None

    def process(self, chunk: AudioChunk) -> ProcessResult:
        x = chunk.samples.float()
        channels, n = x.shape[0], x.shape[-1]
        pad = max(0, self.fft_size - n)
        padded = torch.nn.functional.pad(x, (0, pad))
        spec = torch.fft.rfft(padded, n=self.fft_size, dim=-1)
        mag = spec.abs().mean(dim=0)
        speech_prob = float(chunk.metadata.get("speech_prob", 0.0))
        if self.noise_mag is None:
            self.noise_mag = mag.detach().clone()
        if speech_prob < self.vad_threshold:
            self.noise_mag = self.noise_alpha * self.noise_mag + (1.0 - self.noise_alpha) * mag.detach()
        assert self.noise_mag is not None
        gain = (mag - self.oversubtraction * self.noise_mag).clamp_min(0.0) / (mag + 1e-8)
        gain = gain.clamp_min(self.min_gain).clamp_max(1.0)
        if self.prev_gain is not None:
            gain = self.temporal_smoothing * self.prev_gain + (1.0 - self.temporal_smoothing) * gain
        self.prev_gain = gain.detach()
        enhanced_spec = spec * gain[None, :]
        enhanced = torch.fft.irfft(enhanced_spec, n=self.fft_size, dim=-1)[..., :n].contiguous()
        noise_floor = float(self.noise_mag.mean().item())
        metadata = merge_metadata(chunk, denoised=True)
        return ProcessResult(
            chunk=AudioChunk(enhanced, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "noise_mag_mean": noise_floor,
                "noise_floor_db": linear_to_db(noise_floor),
                "mean_gain_db": linear_to_db(float(gain.mean().item())),
                "realtime_safe": True,
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
