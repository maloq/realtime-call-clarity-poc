import numpy as np
import torch

from callclarity.methods.bandwidth import flashsr
from callclarity.registry import create_method
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk


class FakeFlashSrBackend:
    variant = "fake"
    provider = "CPUExecutionProvider"
    device = "cpu"

    def __init__(self, config):
        self.config = config
        self.calls = 0

    def run(self, x):
        self.calls += 1
        return np.repeat(np.asarray(x, dtype=np.float32), 3)


def test_flashsr_onnx_changes_sample_rate_and_preserves_duration(monkeypatch):
    monkeypatch.setattr(flashsr, "_FlashSrOnnxBackend", FakeFlashSrBackend)
    pipeline = Pipeline.from_config(
        {
            "name": "flashsr_test",
            "stages": [
                {
                    "type": "bandwidth",
                    "name": "flashsr_onnx",
                    "config": {
                        "input_sample_rate": 16000,
                        "output_sample_rate": 48000,
                        "algorithmic_latency_ms": 10,
                    },
                }
            ],
        }
    )
    waveform = torch.randn(1, 1600) * 0.01

    result = RealtimeSimulator(pipeline, chunk_ms=10, max_added_latency_ms=50).run(waveform, 16000)

    assert result.sample_rate == 48000
    assert result.output.shape[-1] == waveform.shape[-1] * 3
    assert result.output.shape[-1] / result.sample_rate == waveform.shape[-1] / 16000
    assert result.latency_summary["declared_algorithmic_latency_ms"] == 10.0


def test_flashsr_disabled_does_not_load_backend(monkeypatch):
    def fail_backend(config):
        raise AssertionError("backend should not load")

    monkeypatch.setattr(flashsr, "_FlashSrOnnxBackend", fail_backend)
    processor = create_method("bandwidth", "flashsr_onnx", {"enabled": False})
    samples = torch.randn(1, 160)

    result = processor.process(AudioChunk(samples, 16000, 0.0))

    assert result.chunk.samples is samples
    assert result.chunk.sample_rate == 16000
    assert result.algorithmic_latency_ms == 0.0
