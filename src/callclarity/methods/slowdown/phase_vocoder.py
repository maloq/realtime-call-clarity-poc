from __future__ import annotations

from dataclasses import asdict

from callclarity.dsp.phase_vocoder import phase_vocoder_time_scale
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.methods.slowdown.latency_controller import SlowdownLatencyController
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("slowdown", "phase_vocoder")
class PhaseVocoderSlowdown(BaseStreamingProcessor):
    name = "phase_vocoder"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.controller = SlowdownLatencyController(config or {})

    @property
    def algorithmic_latency_ms(self) -> float:
        return 96.0

    def reset(self) -> None:
        self.controller.reset()

    def process(self, chunk: AudioChunk) -> ProcessResult:
        speech_prob = float(chunk.metadata.get("speech_prob", 0.0))
        decision = self.controller.decide(
            chunk.start_time_sec,
            chunk.duration_sec * 1000.0,
            speech_prob,
            chunk.metadata.get("speech_rate"),
            allow_speech_slowdown=True,
        )
        y = chunk.samples if abs(decision.tempo - 1.0) <= 1e-3 else phase_vocoder_time_scale(chunk.samples, chunk.sample_rate, decision.tempo)
        metadata = merge_metadata(chunk, dynamic_buffer_ms=decision.buffer_ms, tempo=decision.tempo)
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={"tempo": decision.tempo, "dynamic_buffer_ms": decision.buffer_ms},
            events=[{"type": "tempo_decision", **asdict(decision)}],
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
