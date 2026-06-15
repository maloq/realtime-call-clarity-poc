from __future__ import annotations

from dataclasses import asdict

import torch

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult, SpeechRateEstimate


@register_method("rate_detector", "onset_flux")
class OnsetFluxRateDetector(BaseStreamingProcessor):
    name = "onset_flux"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.window_sec = float(self.config.get("window_sec", 1.8))
        self.fast_threshold = float(self.config.get("fast_rate_threshold_syllables_per_sec", 5.5))
        self.min_speech_prob = float(self.config.get("min_speech_prob", 0.55))
        self.flux_threshold = float(self.config.get("flux_threshold", 0.08))
        self.prev_mag: torch.Tensor | None = None
        self.events: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.prev_mag = None
        self.events.clear()

    def process(self, chunk: AudioChunk) -> ProcessResult:
        n_fft = min(512, max(32, 2 ** int((chunk.num_samples - 1).bit_length())))
        mag = torch.fft.rfft(torch.nn.functional.pad(chunk.samples.mean(dim=0), (0, n_fft - chunk.num_samples)), n=n_fft).abs()
        flux = 0.0
        if self.prev_mag is not None:
            delta = (mag - self.prev_mag).clamp_min(0.0)
            flux = float(delta.mean().item())
        self.prev_mag = mag.detach()
        t = chunk.start_time_sec + chunk.duration_sec
        speech_prob = float(chunk.metadata.get("speech_prob", 0.0))
        if speech_prob >= self.min_speech_prob and flux >= self.flux_threshold:
            if not self.events or t - self.events[-1][0] >= 0.12:
                self.events.append((t, flux))
        cutoff = t - self.window_sec
        self.events = [e for e in self.events if e[0] >= cutoff]
        voiced_fraction = min(1.0, sum(1 for e in self.events if e[0] >= cutoff) * 0.15 / self.window_sec)
        voiced_sec = max(voiced_fraction * self.window_sec, chunk.duration_sec)
        rate = len(self.events) / voiced_sec
        confidence = min(1.0, speech_prob * (0.4 + len(self.events) / 8.0))
        est = SpeechRateEstimate(
            timestamp_sec=t,
            window_sec=self.window_sec,
            is_fast=rate >= self.fast_threshold and confidence >= 0.4,
            confidence=confidence,
            syllables_per_sec=rate,
            words_per_sec=None,
            chars_per_sec=None,
            speech_fraction=voiced_fraction,
            method=self.name,
            debug={"flux": flux, "peak_count": len(self.events)},
        )
        return ProcessResult(
            chunk=AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, merge_metadata(chunk, speech_rate=asdict(est))),
            metrics=asdict(est),
            events=[{"type": "speech_rate", **asdict(est)}],
            algorithmic_latency_ms=40.0,
        )
