from __future__ import annotations

from dataclasses import asdict

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult, SpeechRateEstimate


@register_method("rate_detector", "transcript_duration")
class TranscriptDurationRateDetector(BaseStreamingProcessor):
    name = "transcript_duration"
    realtime_safe = False

    def process(self, chunk: AudioChunk) -> ProcessResult:
        transcript = str(chunk.metadata.get("transcript", ""))
        words = [w for w in transcript.split() if w.strip()]
        active_sec = float(chunk.metadata.get("active_speech_sec", max(chunk.start_time_sec + chunk.duration_sec, 1e-3)))
        words_per_sec = len(words) / max(active_sec, 1e-6)
        chars_per_sec = len(transcript.replace(" ", "")) / max(active_sec, 1e-6)
        is_fast = words_per_sec >= float(self.config.get("fast_words_per_sec", 3.2)) or chars_per_sec >= float(
            self.config.get("fast_chars_per_sec", 16.0)
        )
        est = SpeechRateEstimate(
            timestamp_sec=chunk.start_time_sec,
            window_sec=active_sec,
            is_fast=is_fast,
            confidence=0.5 if transcript else 0.0,
            syllables_per_sec=None,
            words_per_sec=words_per_sec if transcript else None,
            chars_per_sec=chars_per_sec if transcript else None,
            speech_fraction=float(chunk.metadata.get("speech_prob", 0.0)),
            method=self.name,
            debug={"realtime_safe": False},
        )
        metadata = merge_metadata(chunk, speech_rate=asdict(est))
        return ProcessResult(
            chunk=AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics=asdict(est),
            events=[{"type": "speech_rate", **asdict(est)}],
            algorithmic_latency_ms=0.0,
        )
