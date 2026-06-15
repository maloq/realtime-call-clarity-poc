from __future__ import annotations

from dataclasses import asdict

import numpy as np

from callclarity.dsp.envelope import rms_dbfs, smooth_1d
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult, SpeechRateEstimate


@register_method("rate_detector", "syllable_nuclei")
class SyllableNucleiRateDetector(BaseStreamingProcessor):
    name = "syllable_nuclei"
    realtime_safe = True

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.window_sec = float(self.config.get("window_sec", 1.8))
        self.update_ms = float(self.config.get("update_ms", 100.0))
        self.fast_threshold = float(self.config.get("fast_rate_threshold_syllables_per_sec", 5.5))
        self.min_speech_prob = float(self.config.get("min_speech_prob", 0.55))
        self.prominence_db = float(self.config.get("peak_prominence_db", 2.5))
        self.history: list[tuple[float, float, float]] = []
        self.peak_times: list[float] = []
        self._last_update = -1e9

    def reset(self) -> None:
        self.history.clear()
        self.peak_times.clear()
        self._last_update = -1e9

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_sec
        self.history = [h for h in self.history if h[0] >= cutoff]
        self.peak_times = [t for t in self.peak_times if t >= cutoff]

    def _maybe_peak(self) -> None:
        if len(self.history) < 3:
            return
        a, b, c = self.history[-3], self.history[-2], self.history[-1]
        if b[2] < self.min_speech_prob:
            return
        local_floor = min(a[1], c[1])
        if b[1] > a[1] and b[1] >= c[1] and (b[1] - local_floor) >= self.prominence_db:
            if not self.peak_times or (b[0] - self.peak_times[-1]) >= 0.12:
                self.peak_times.append(b[0])

    def process(self, chunk: AudioChunk) -> ProcessResult:
        mid = chunk.start_time_sec + 0.5 * chunk.duration_sec
        energy = rms_dbfs(chunk.samples)
        speech_prob = float(chunk.metadata.get("speech_prob", 0.0))
        self.history.append((mid, energy, speech_prob))
        self._maybe_peak()
        self._trim(mid)
        speech_frames = [h for h in self.history if h[2] >= self.min_speech_prob]
        frame_sec = chunk.duration_sec
        voiced_sec = max(frame_sec * len(speech_frames), 1e-6)
        speech_fraction = min(1.0, voiced_sec / max(self.window_sec, frame_sec))
        syllables_per_sec = len(self.peak_times) / voiced_sec if speech_frames else 0.0
        if len(self.peak_times) >= 2:
            intervals = np.diff(np.asarray(self.peak_times))
            regularity = 1.0 / (1.0 + float(np.std(intervals)) * 4.0)
        else:
            regularity = 0.5
        confidence = float(min(1.0, speech_fraction * 1.5) * regularity)
        is_fast = syllables_per_sec >= self.fast_threshold and confidence >= 0.4
        est = SpeechRateEstimate(
            timestamp_sec=mid,
            window_sec=self.window_sec,
            is_fast=is_fast,
            confidence=confidence,
            syllables_per_sec=syllables_per_sec,
            words_per_sec=None,
            chars_per_sec=None,
            speech_fraction=speech_fraction,
            method=self.name,
            debug={"peak_count": len(self.peak_times), "voiced_sec": voiced_sec},
        )
        metadata = merge_metadata(chunk, speech_rate=asdict(est))
        events = []
        if mid - self._last_update >= self.update_ms / 1000.0:
            events.append({"type": "speech_rate", **asdict(est)})
            self._last_update = mid
        return ProcessResult(
            chunk=AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics=asdict(est),
            events=events,
            algorithmic_latency_ms=float(self.config.get("smooth_ms", 70.0)),
        )
