import sys
import types

import numpy as np
import torch

from callclarity.registry import create_method
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk


class FakeAudioProcessingModule:
    instances = []

    def __init__(self, aec_type=0, enable_ns=False, agc_type=0, enable_vad=False):
        self.aec_type = aec_type
        self.enable_ns = enable_ns
        self.agc_type = agc_type
        self.enable_vad = enable_vad
        self.sample_rate = None
        self.channels = None
        self.out_sample_rate = None
        self.out_channels = None
        self.ns_level = None
        self.agc_target = None
        self.vad_level = None
        self.calls = 0
        self.__class__.instances.append(self)

    def set_stream_format(self, sample_rate, channels, out_sample_rate=16000, out_channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.out_sample_rate = out_sample_rate
        self.out_channels = out_channels

    def set_ns_level(self, level):
        self.ns_level = level

    def set_agc_target(self, dbfs):
        self.agc_target = dbfs

    def set_vad_level(self, level):
        self.vad_level = level

    def process_stream(self, payload):
        self.calls += 1
        pcm = np.frombuffer(payload, dtype="<i2").copy()
        return np.round(pcm.astype(np.float32) * 0.5).astype("<i2").tobytes()

    def has_voice(self):
        return True

    def agc_level(self):
        return 42


def _install_fake_webrtc(monkeypatch):
    FakeAudioProcessingModule.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "webrtc_audio_processing",
        types.SimpleNamespace(AudioProcessingModule=FakeAudioProcessingModule),
    )


def test_webrtc_apm_registered_wrapper_preserves_shape_and_metadata(monkeypatch):
    _install_fake_webrtc(monkeypatch)
    processor = create_method(
        "denoise",
        "webrtc_apm",
        {
            "noise_suppression": {"enabled": True, "level": "high"},
            "digital_agc": {"enabled": True, "target_level_dbfs": -16},
            "vad": {"enabled": True, "level": "moderate"},
        },
    )
    samples = torch.ones(1, 160) * 0.5

    result = processor.process(AudioChunk(samples, 16000, 0.0))

    assert result.chunk.samples.shape == samples.shape
    assert torch.allclose(result.chunk.samples, samples * 0.5, atol=1e-4)
    assert result.chunk.metadata["denoised"] is True
    assert result.chunk.metadata["webrtc_apm"] is True
    assert result.metrics["webrtc_apm_ns_level"] == 2
    assert result.metrics["webrtc_apm_agc_target_dbfs"] == 16
    assert result.metrics["webrtc_apm_voice_detected"] is True
    assert result.algorithmic_latency_ms == 10.0

    instance = FakeAudioProcessingModule.instances[-1]
    assert instance.aec_type == 0
    assert instance.enable_ns is True
    assert instance.agc_type == 1
    assert instance.enable_vad is True
    assert instance.sample_rate == 16000
    assert instance.ns_level == 2
    assert instance.agc_target == 16
    assert instance.vad_level == 1


def test_webrtc_apm_pipeline_keeps_duration_with_subframe_chunks(monkeypatch):
    _install_fake_webrtc(monkeypatch)
    pipeline = Pipeline.from_config(
        {
            "name": "webrtc_apm_test",
            "stages": [
                {
                    "type": "denoise",
                    "name": "webrtc_apm",
                    "config": {"noise_suppression": {"level": "moderate"}},
                }
            ],
        }
    )
    waveform = torch.ones(1, 240)

    result = RealtimeSimulator(pipeline, chunk_ms=5).run(waveform, 16000)

    assert result.output.shape == waveform.shape
    assert torch.allclose(result.output[:, :80], torch.zeros(1, 80))
    assert torch.allclose(result.output[:, 80:], torch.full((1, 160), 0.5), atol=1e-4)
    assert torch.isfinite(result.output).all()


def test_webrtc_apm_disabled_does_not_require_backend(monkeypatch):
    monkeypatch.delitem(sys.modules, "webrtc_audio_processing", raising=False)
    processor = create_method("denoise", "webrtc_apm", {"enabled": False})
    samples = torch.randn(1, 160)

    result = processor.process(AudioChunk(samples, 16000, 0.0))

    assert result.chunk.samples is samples
    assert result.algorithmic_latency_ms == 0.0
