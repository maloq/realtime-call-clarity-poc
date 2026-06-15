from __future__ import annotations

from collections.abc import Iterator

import torch

from callclarity.types import AudioChunk


def iter_audio_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_ms: float = 10.0,
    stream_id: str = "default",
    pad_final: bool = False,
    metadata: dict | None = None,
) -> Iterator[AudioChunk]:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    chunk_samples = max(1, int(round(sample_rate * chunk_ms / 1000.0)))
    total = int(waveform.shape[-1])
    start = 0
    while start < total:
        end = min(total, start + chunk_samples)
        samples = waveform[..., start:end]
        if pad_final and samples.shape[-1] < chunk_samples:
            samples = torch.nn.functional.pad(samples, (0, chunk_samples - samples.shape[-1]))
        yield AudioChunk(
            samples=samples.contiguous(),
            sample_rate=sample_rate,
            start_time_sec=start / float(sample_rate),
            stream_id=stream_id,
            metadata=dict(metadata or {}),
        )
        start = end


def reconstruct_chunks(chunks: list[AudioChunk]) -> torch.Tensor:
    if not chunks:
        return torch.zeros(1, 0)
    return torch.cat([chunk.samples for chunk in chunks if chunk.samples.numel() > 0], dim=-1)
