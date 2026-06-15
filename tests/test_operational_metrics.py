import torch

from callclarity.metrics.operational import OperationalMetricsTracker
from callclarity.types import AudioChunk


def test_operational_metrics_detect_zero_and_repeated_frames():
    tracker = OperationalMetricsTracker()
    chunk = AudioChunk(torch.zeros(1, 160), 16000, 0.0)
    tracker.add_chunk(chunk, chunk, processing_time_ms=0.1, added_latency_ms=0.0, dynamic_buffer_ms=0.0)
    repeated = AudioChunk(torch.zeros(1, 160), 16000, 0.01)
    tracker.add_chunk(repeated, repeated, processing_time_ms=0.1, added_latency_ms=0.0, dynamic_buffer_ms=0.0)
    summary = tracker.summary()
    assert summary["zero_frame_count"] == 2
    assert summary["repeated_frame_count"] == 1
    assert summary["processing_rtf"] >= 0.0
