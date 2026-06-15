from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass
class AudioChunk:
    samples: torch.Tensor
    sample_rate: int
    start_time_sec: float
    stream_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[-1])

    @property
    def duration_sec(self) -> float:
        return self.num_samples / float(self.sample_rate)


@dataclass
class ProcessResult:
    chunk: AudioChunk
    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    algorithmic_latency_ms: float = 0.0
    processing_time_ms: float = 0.0


class StreamingProcessor(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def warmup(self, sample_rate: int) -> None:
        ...

    def process(self, chunk: AudioChunk) -> ProcessResult:
        ...

    @property
    def algorithmic_latency_ms(self) -> float:
        ...

    @property
    def lookahead_ms(self) -> float:
        ...


@dataclass
class DatasetItem:
    recording_id: str
    audio_path: Path
    transcript_path: Path | None
    transcript: str
    sample_rate: int | None = None
    waveform: torch.Tensor | None = None
    clean_reference_path: Path | None = None
    speaker_id: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechRateEstimate:
    timestamp_sec: float
    window_sec: float
    is_fast: bool
    confidence: float
    syllables_per_sec: float | None
    words_per_sec: float | None
    chars_per_sec: float | None
    speech_fraction: float
    method: str
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class TempoDecision:
    timestamp_sec: float
    tempo: float
    reason: str
    buffer_ms: float
    fast_speech_confidence: float
    hard_limit_active: bool


class MethodUnavailable(RuntimeError):
    """Raised when an optional method backend is missing or not configured."""
