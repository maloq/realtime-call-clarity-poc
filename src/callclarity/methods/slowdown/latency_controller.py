from __future__ import annotations

from dataclasses import asdict
from typing import Any

from callclarity.types import SpeechRateEstimate, TempoDecision


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SlowdownLatencyController:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        target = cfg.get("target", {})
        tempo = cfg.get("tempo", {})
        decision = cfg.get("decision", {})
        self.max_added_latency_ms = float(target.get("max_added_latency_ms", 160.0))
        self.preferred_buffer_ms = float(target.get("preferred_buffer_ms", 70.0))
        self.hard_buffer_ms = float(target.get("hard_buffer_ms", 180.0))
        self.min_tempo = float(tempo.get("min_tempo", 0.90))
        self.burst_min_tempo = float(tempo.get("burst_min_tempo", 0.85))
        self.max_speech_catchup_tempo = float(tempo.get("max_speech_catchup_tempo", 1.06))
        self.max_silence_catchup_tempo = float(tempo.get("max_silence_catchup_tempo", 3.0))
        self.smoothing_per_100ms = float(tempo.get("smoothing_per_100ms", 0.02))
        self.fast_threshold = float(decision.get("fast_rate_threshold_syllables_per_sec", 5.5))
        self.min_confidence = float(decision.get("min_confidence", 0.65))
        self.min_speech_prob = float(decision.get("min_speech_prob", 0.6))
        self.buffer_ms = 0.0
        self.previous_tempo = 1.0
        self.hard_limit_hit_count = 0

    def reset(self) -> None:
        self.buffer_ms = 0.0
        self.previous_tempo = 1.0
        self.hard_limit_hit_count = 0

    def _smooth(self, desired: float, input_ms: float) -> float:
        max_delta = self.smoothing_per_100ms * max(input_ms / 100.0, 0.1)
        if abs(desired - self.previous_tempo) <= max_delta:
            return desired
        return self.previous_tempo + max_delta * (1.0 if desired > self.previous_tempo else -1.0)

    def decide(
        self,
        timestamp_sec: float,
        input_ms: float,
        speech_prob: float,
        rate: SpeechRateEstimate | dict[str, Any] | None,
        allow_speech_slowdown: bool = True,
    ) -> TempoDecision:
        if isinstance(rate, dict):
            measured = rate.get("syllables_per_sec")
            is_fast = bool(rate.get("is_fast", False))
            confidence = float(rate.get("confidence", 0.0))
        elif rate is not None:
            measured = rate.syllables_per_sec
            is_fast = rate.is_fast
            confidence = rate.confidence
        else:
            measured = None
            is_fast = False
            confidence = 0.0
        silence = speech_prob < self.min_speech_prob
        hard = self.buffer_ms >= self.hard_buffer_ms - 1e-6
        if hard:
            desired = self.max_speech_catchup_tempo
            reason = "hard_buffer_limit"
            self.hard_limit_hit_count += 1
        elif silence and self.buffer_ms > self.preferred_buffer_ms:
            desired = self.max_silence_catchup_tempo
            reason = "silence_catchup"
        elif (
            allow_speech_slowdown
            and is_fast
            and confidence >= self.min_confidence
            and speech_prob >= self.min_speech_prob
            and self.buffer_ms < self.max_added_latency_ms - 30.0
        ):
            desired = self.fast_threshold / max(float(measured or self.fast_threshold), 1e-6)
            desired = _clamp(desired, self.min_tempo, 1.0)
            reason = "fast_speech_slowdown"
        elif self.buffer_ms > self.preferred_buffer_ms + 40.0:
            desired = self.max_speech_catchup_tempo
            reason = "soft_catchup"
        else:
            desired = 1.0
            reason = "normal"

        tempo = self._smooth(desired, input_ms)
        if tempo < 1.0:
            available = max(0.0, self.hard_buffer_ms - self.buffer_ms)
            min_safe_tempo = input_ms / max(input_ms + available, 1e-6)
            tempo = max(tempo, min_safe_tempo)
        tempo = _clamp(tempo, self.burst_min_tempo, self.max_silence_catchup_tempo)
        projected_delta = input_ms / tempo - input_ms
        self.buffer_ms = _clamp(self.buffer_ms + projected_delta, 0.0, self.hard_buffer_ms)
        self.previous_tempo = tempo
        decision = TempoDecision(
            timestamp_sec=timestamp_sec,
            tempo=float(tempo),
            reason=reason,
            buffer_ms=float(self.buffer_ms),
            fast_speech_confidence=confidence,
            hard_limit_active=hard,
        )
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer_ms": self.buffer_ms,
            "previous_tempo": self.previous_tempo,
            "hard_limit_hit_count": self.hard_limit_hit_count,
        }

    @staticmethod
    def decision_to_event(decision: TempoDecision) -> dict[str, Any]:
        return {"type": "tempo_decision", **asdict(decision)}
