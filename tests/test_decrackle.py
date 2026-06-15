import time
from pathlib import Path

import torch

from callclarity.io.audio_io import load_audio
from callclarity.methods.repair.decrackle import DecrackleProcessor
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator
from callclarity.types import AudioChunk


def _speech_like(sample_rate: int = 16000, seconds: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(sample_rate * seconds)).float() / float(sample_rate)
    envelope = 0.55 + 0.45 * torch.sin(2 * torch.pi * 3.0 * t).abs()
    carrier = (
        0.08 * torch.sin(2 * torch.pi * 180.0 * t)
        + 0.035 * torch.sin(2 * torch.pi * 620.0 * t)
        + 0.018 * torch.sin(2 * torch.pi * 2100.0 * t)
    )
    return (carrier * envelope).unsqueeze(0)


def _run_streaming(waveform: torch.Tensor, config: dict | None = None, chunk_ms: float = 10.0):
    processor = DecrackleProcessor(config or {})
    chunks = []
    events = []
    metrics = []
    sample_rate = 16000
    chunk_samples = int(round(sample_rate * chunk_ms / 1000.0))
    for idx, start in enumerate(range(0, waveform.shape[-1], chunk_samples)):
        chunk = AudioChunk(waveform[..., start : start + chunk_samples], sample_rate, idx * chunk_ms / 1000.0)
        result = processor.process(chunk)
        chunks.append(result.chunk.samples)
        events.extend(result.events)
        metrics.append(result.metrics)
    return torch.cat(chunks, dim=-1), events, metrics


def test_decrackle_reduces_synthetic_impulses():
    clean = _speech_like()
    noisy = clean.clone()
    positions = [600, 1700, 4100, 6200]
    for pos in positions:
        noisy[0, pos] = 0.95 if pos % 2 else -0.95
    output, events, metrics = _run_streaming(noisy, {"strength": "medium"})
    before = torch.mean(torch.abs(noisy[0, positions] - clean[0, positions]))
    after = torch.mean(torch.abs(output[0, positions] - clean[0, positions]))
    assert after < before * 0.45
    assert sum(int(row["decrackle_repaired_click_count"]) for row in metrics) >= len(positions)
    assert events
    assert torch.isfinite(output).all()
    assert float(output.abs().max()) <= 1.0


def test_decrackle_keeps_clean_speech_mostly_unchanged():
    clean = _speech_like(seconds=0.7)
    output, _events, metrics = _run_streaming(clean, {"strength": "mild"})
    diff = torch.mean(torch.abs(output - clean)).item()
    assert diff < 1e-4
    assert sum(int(row["decrackle_repaired_click_count"]) for row in metrics) == 0


def test_decrackle_repairs_click_at_frame_boundary():
    clean = _speech_like(seconds=0.08)
    noisy = clean.clone()
    boundary = 160
    noisy[0, boundary] = 1.0
    output, _events, metrics = _run_streaming(noisy, {"strength": "medium"}, chunk_ms=10.0)
    assert abs(float(output[0, boundary] - clean[0, boundary])) < abs(float(noisy[0, boundary] - clean[0, boundary])) * 0.5
    assert sum(int(row["decrackle_repaired_click_count"]) for row in metrics) >= 1


def test_decrackle_pipeline_preserves_shape_and_budget():
    pipeline = Pipeline.from_config(
        {
            "name": "decrackle_perf_test",
            "stages": [
                {"type": "repair", "name": "decrackle", "config": {"enabled": True, "strength": "mild"}},
            ],
        }
    )
    waveform = _speech_like(seconds=1.0)
    start = time.perf_counter()
    result = RealtimeSimulator(pipeline, chunk_ms=10).run(waveform, 16000)
    elapsed = time.perf_counter() - start
    assert result.output.shape == waveform.shape
    assert torch.isfinite(result.output).all()
    assert elapsed / 1.0 < 0.5


def test_decrackle_fixture_crackle_repairs_more_than_clean():
    clean_path = Path("data/test_data_samples/clean/clean1.wav")
    crackle_path = Path("data/test_data_samples/crackle_examples/crackle1.wav")
    if not (clean_path.exists() and crackle_path.exists()):
        return
    clean, _ = load_audio(clean_path, 16000, "mono")
    crackle, _ = load_audio(crackle_path, 16000, "mono")
    clean = clean[..., : 16000]
    crackle = crackle[..., : 16000]
    _, _clean_events, clean_metrics = _run_streaming(clean, {"strength": "mild"})
    _, _crackle_events, crackle_metrics = _run_streaming(crackle, {"strength": "mild"})
    clean_repairs = sum(int(row["decrackle_repaired_click_count"]) for row in clean_metrics)
    crackle_repairs = sum(int(row["decrackle_repaired_click_count"]) for row in crackle_metrics)
    assert crackle_repairs > clean_repairs
