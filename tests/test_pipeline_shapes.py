import torch

from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator


def test_pipeline_preserves_valid_tensor_shapes():
    pipeline = Pipeline.from_config({"name": "baseline", "stages": [{"type": "denoise", "name": "passthrough", "config": {}}]})
    waveform = torch.randn(1, 1000)
    result = RealtimeSimulator(pipeline).run(waveform, 16000)
    assert result.output.ndim == 2
    assert result.output.shape[0] == 1
    assert result.output.shape[-1] == waveform.shape[-1]
