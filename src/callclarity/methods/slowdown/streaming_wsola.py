from __future__ import annotations

from dataclasses import asdict

from callclarity.dsp.wsola import wsola_time_scale
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.methods.slowdown.latency_controller import SlowdownLatencyController
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("slowdown", "streaming_wsola")
class StreamingWsolaSlowdown(BaseStreamingProcessor):
    name = "streaming_wsola"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.controller = SlowdownLatencyController(config or {})
        self.wsola_cfg = self.config.get("wsola", {})

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(self.wsola_cfg.get("frame_ms", 40.0))

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
        y = chunk.samples
        if abs(decision.tempo - 1.0) > 1e-3:
            y = wsola_time_scale(
                y,
                chunk.sample_rate,
                decision.tempo,
                frame_ms=float(self.wsola_cfg.get("frame_ms", 40.0)),
                analysis_hop_ms=float(self.wsola_cfg.get("analysis_hop_ms", 10.0)),
                search_ms=float(self.wsola_cfg.get("search_ms", 15.0)),
                crossfade_ms=float(self.wsola_cfg.get("crossfade_ms", 8.0)),
            )
        metadata = merge_metadata(chunk, dynamic_buffer_ms=decision.buffer_ms, tempo=decision.tempo)
        return ProcessResult(
            chunk=AudioChunk(y.contiguous(), chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "tempo": decision.tempo,
                "dynamic_buffer_ms": decision.buffer_ms,
                "slowdown_active": decision.tempo < 0.999,
                "catchup_active": decision.tempo > 1.001,
                "hard_latency_limit_hit_count": self.controller.hard_limit_hit_count,
            },
            events=[{"type": "tempo_decision", **asdict(decision)}],
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
