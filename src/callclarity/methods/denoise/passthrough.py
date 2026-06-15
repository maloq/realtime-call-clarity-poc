from __future__ import annotations

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("denoise", "passthrough")
class PassthroughDenoiser(BaseStreamingProcessor):
    name = "passthrough"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        return ProcessResult(
            chunk=chunk,
            metrics={"realtime_safe": True},
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
