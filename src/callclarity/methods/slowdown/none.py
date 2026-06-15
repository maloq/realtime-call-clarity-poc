from __future__ import annotations

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("slowdown", "none")
class NoSlowdown(BaseStreamingProcessor):
    name = "slowdown_none"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        metadata = merge_metadata(chunk, dynamic_buffer_ms=0.0, tempo=1.0)
        return ProcessResult(
            chunk=AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={"tempo": 1.0, "dynamic_buffer_ms": 0.0, "slowdown_active": False},
        )
