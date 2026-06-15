import sys
import types

import numpy as np
import torch

from callclarity.registry import create_method
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk


class FakeStreamEnhancer:
    def __init__(self, model="dpdfnet2", onnx_path=None, verbose=False):
        self.model = model
        self.onnx_path = onnx_path
        self.verbose = verbose
        self._model_sr = 16000
        self._win_len = 320
        self._hop_size = 160
        self.calls = 0

    def reset(self):
        self.calls = 0

    def process(self, chunk, sample_rate=None):
        self.calls += 1
        return np.asarray(chunk, dtype=np.float32) * 0.5


class DelayedFakeStreamEnhancer(FakeStreamEnhancer):
    def process(self, chunk, sample_rate=None):
        self.calls += 1
        if self.calls == 1:
            return np.zeros(0, dtype=np.float32)
        return np.ones_like(np.asarray(chunk, dtype=np.float32))


def _install_fake_dpdfnet(monkeypatch, enhancer_cls):
    monkeypatch.setitem(
        sys.modules,
        "dpdfnet",
        types.SimpleNamespace(StreamEnhancer=enhancer_cls),
    )


def test_dpdfnet_registered_wrapper_preserves_shape_and_metadata(monkeypatch):
    _install_fake_dpdfnet(monkeypatch, FakeStreamEnhancer)
    processor = create_method("denoise", "dpdfnet", {"model": "dpdfnet4"})
    samples = torch.ones(1, 160)

    result = processor.process(AudioChunk(samples, 16000, 0.0))

    assert result.chunk.samples.shape == samples.shape
    assert torch.allclose(result.chunk.samples, samples * 0.5)
    assert result.chunk.metadata["denoised"] is True
    assert result.metrics["neural_denoiser"] == "DPDFNet"
    assert result.metrics["dpdfnet_model"] == "dpdfnet4"
    assert result.algorithmic_latency_ms == 20.0


def test_dpdfnet_pipeline_keeps_duration_when_stream_output_is_delayed(monkeypatch):
    _install_fake_dpdfnet(monkeypatch, DelayedFakeStreamEnhancer)
    pipeline = Pipeline.from_config(
        {
            "name": "dpdfnet_test",
            "stages": [
                {
                    "type": "denoise",
                    "name": "dpdfnet",
                    "config": {"model": "dpdfnet2"},
                }
            ],
        }
    )
    waveform = torch.randn(1, 480)

    result = RealtimeSimulator(pipeline, chunk_ms=10).run(waveform, 16000)

    assert result.output.shape == waveform.shape
    assert torch.allclose(result.output[:, :160], torch.zeros(1, 160))
    assert torch.isfinite(result.output).all()
