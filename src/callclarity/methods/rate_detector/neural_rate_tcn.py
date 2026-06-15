from __future__ import annotations

from dataclasses import asdict

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult, SpeechRateEstimate


@register_method("rate_detector", "neural_rate_tcn")
class NeuralRateTcnProcessor(BaseStreamingProcessor):
    name = "neural_rate_tcn"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        # A checkpoint-backed implementation can replace this neutral fallback.
        speech_prob = float(chunk.metadata.get("speech_prob", 0.0))
        est = SpeechRateEstimate(
            timestamp_sec=chunk.start_time_sec + chunk.duration_sec,
            window_sec=float(self.config.get("context_sec", 2.0)),
            is_fast=False,
            confidence=0.0,
            syllables_per_sec=None,
            words_per_sec=None,
            chars_per_sec=None,
            speech_fraction=speech_prob,
            method=self.name,
            debug={"checkpoint_path": self.config.get("checkpoint_path")},
        )
        return ProcessResult(
            chunk=AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, merge_metadata(chunk, speech_rate=asdict(est))),
            metrics=asdict(est),
            events=[{"type": "speech_rate", **asdict(est)}],
            algorithmic_latency_ms=100.0,
        )
