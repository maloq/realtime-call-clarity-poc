import torch

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk, ProcessResult


class LatencyProbe(BaseStreamingProcessor):
    name = "latency_probe"

    @property
    def algorithmic_latency_ms(self):
        return 25.0

    def process(self, chunk: AudioChunk) -> ProcessResult:
        metadata = dict(chunk.metadata)
        metadata["dynamic_buffer_ms"] = 13.0
        return ProcessResult(
            AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )


def test_latency_accounting_includes_algorithmic_and_dynamic():
    result = RealtimeSimulator(Pipeline([LatencyProbe({})]), max_added_latency_ms=200).run(torch.zeros(1, 320), 16000)
    totals = result.latency_summary["total_added_latency_ms"]
    assert totals["max"] == 38.0
    assert result.latency_summary["declared_algorithmic_latency_ms"] == 25.0
