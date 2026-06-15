from __future__ import annotations

from typing import Any

from callclarity.types import AudioChunk, ProcessResult


class BaseStreamingProcessor:
    name = "base"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def reset(self) -> None:
        pass

    def warmup(self, sample_rate: int) -> None:
        del sample_rate

    def process(self, chunk: AudioChunk) -> ProcessResult:
        return ProcessResult(chunk=chunk, algorithmic_latency_ms=self.algorithmic_latency_ms)

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(self.config.get("algorithmic_latency_ms", 0.0))

    @property
    def lookahead_ms(self) -> float:
        return float(self.config.get("lookahead_ms", 0.0))


def merge_metadata(chunk: AudioChunk, **updates: Any) -> dict[str, Any]:
    metadata = dict(chunk.metadata)
    metadata.update(updates)
    return metadata
