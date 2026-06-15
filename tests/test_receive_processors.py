import torch

from callclarity.methods.filter.dc_highpass import DcHighpassFilterProcessor
from callclarity.methods.repair.dropout_click import DropoutClickRepairProcessor
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk


def test_dc_highpass_reduces_dc_offset():
    processor = DcHighpassFilterProcessor(
        {
            "dc_blocker": {"enabled": True, "radius": 0.98},
            "highpass": {"enabled": True, "cutoff_hz": 90.0, "order": 2},
            "presence": {"enabled": False},
        }
    )
    processor.warmup(16000)
    means = []
    for idx in range(80):
        chunk = AudioChunk(torch.ones(1, 160) * 0.25, 16000, idx * 0.01)
        out = processor.process(chunk).chunk.samples
        means.append(float(out.mean()))
    assert abs(means[-1]) < abs(means[0])
    assert abs(means[-1]) < 0.05


def test_click_and_tiny_gap_repair_preserve_shape():
    processor = DropoutClickRepairProcessor({})
    samples = torch.sin(2 * torch.pi * 220 * torch.arange(160).float() / 16000).unsqueeze(0) * 0.05
    samples[0, 40] = 1.0
    samples[0, 80:83] = 0.0
    result = processor.process(AudioChunk(samples, 16000, 0.0))
    assert result.chunk.samples.shape == samples.shape
    assert result.metrics["click_repair_count"] >= 1
    assert result.metrics["tiny_gap_repair_count"] >= 1
    assert torch.isfinite(result.chunk.samples).all()


def test_receive_baseline_streaming_preserves_duration_and_finiteness():
    pipeline = Pipeline.from_config(
        {
            "name": "receive_baseline_test",
            "stages": [
                {"type": "preprocess", "name": "audio_validation", "config": {"enabled": True, "mono": True}},
                {"type": "repair", "name": "dropout_click", "config": {"enabled": True}},
                {
                    "type": "filter",
                    "name": "dc_highpass",
                    "config": {
                        "enabled": True,
                        "dc_blocker": {"enabled": True},
                        "highpass": {"enabled": True, "cutoff_hz": 90.0},
                    },
                },
                {"type": "limiter", "name": "limiter", "config": {"ceiling_dbfs": -1.5}},
            ],
        }
    )
    waveform = torch.randn(2, 1000) * 0.05
    result = RealtimeSimulator(pipeline, chunk_ms=10).run(waveform, 16000)
    assert result.output.shape[0] == 1
    assert result.output.shape[-1] == waveform.shape[-1]
    assert torch.isfinite(result.output).all()
