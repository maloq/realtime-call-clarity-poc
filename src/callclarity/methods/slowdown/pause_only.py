from __future__ import annotations

from dataclasses import asdict

from callclarity.dsp.wsola import wsola_time_scale
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.methods.slowdown.latency_controller import SlowdownLatencyController
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("slowdown", "pause_only")
class PauseOnlySlowdown(BaseStreamingProcessor):
    name = "pause_only"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.controller = SlowdownLatencyController(config or {})

    def reset(self) -> None:
        self.controller.reset()

    def process(self, chunk: AudioChunk) -> ProcessResult:
        speech_prob = float(chunk.metadata.get("speech_prob", 0.0))
        decision = self.controller.decide(
            chunk.start_time_sec,
            chunk.duration_sec * 1000.0,
            speech_prob,
            chunk.metadata.get("speech_rate"),
            allow_speech_slowdown=False,
        )
        y = chunk.samples
        if decision.tempo > 1.0 and speech_prob < self.controller.min_speech_prob:
            y = wsola_time_scale(y, chunk.sample_rate, decision.tempo)
        metadata = merge_metadata(chunk, dynamic_buffer_ms=decision.buffer_ms, tempo=decision.tempo)
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "tempo": decision.tempo,
                "dynamic_buffer_ms": decision.buffer_ms,
                "slowdown_active": False,
                "catchup_active": decision.tempo > 1.0,
            },
            events=[{"type": "tempo_decision", **asdict(decision)}],
        )
