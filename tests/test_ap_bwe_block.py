import numpy as np
import torch

from callclarity.methods.bandwidth import ap_bwe
from callclarity.registry import create_method
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator


class FakeApBweBackend:
    task = "fake"
    device_requested = "cpu"
    device = "cpu"
    model_sample_rate = 1000
    low_sample_rate = 500

    def __init__(self, config):
        self.config = config
        self.calls = 0

    def process_window(self, window, sample_rate):
        self.calls += 1
        assert sample_rate == 1000
        return np.asarray(window, dtype=np.float32) + 0.25


def test_ap_bwe_block_buffers_context_and_preserves_duration(monkeypatch):
    monkeypatch.setattr(ap_bwe, "_ApBweBackend", FakeApBweBackend)
    pipeline = Pipeline.from_config(
        {
            "name": "ap_bwe_block_test",
            "stages": [
                {
                    "type": "bandwidth",
                    "name": "ap_bwe_block",
                    "config": {
                        "hop_ms": 20,
                        "left_context_ms": 10,
                        "right_context_ms": 10,
                        "algorithmic_latency_ms": 30,
                    },
                }
            ],
        }
    )
    waveform = torch.ones(1, 80)

    result = RealtimeSimulator(pipeline, chunk_ms=10).run(waveform, 1000)

    assert result.output.shape == waveform.shape
    assert torch.allclose(result.output[:, :20], torch.zeros(1, 20))
    assert torch.allclose(result.output[:, 20:60], torch.full((1, 40), 1.25))
    assert result.latency_summary["declared_algorithmic_latency_ms"] == 30.0


def test_ap_bwe_block_disabled_does_not_load_backend(monkeypatch):
    def fail_backend(config):
        raise AssertionError("backend should not load")

    monkeypatch.setattr(ap_bwe, "_ApBweBackend", fail_backend)
    processor = create_method("bandwidth", "ap_bwe_block", {"enabled": False})
    samples = torch.randn(1, 160)

    result = processor.process(ap_bwe.AudioChunk(samples, 16000, 0.0))

    assert result.chunk.samples is samples
    assert result.algorithmic_latency_ms == 0.0
