from __future__ import annotations

from callclarity.dsp.compressor import apply_gain_db, static_compress
from callclarity.dsp.envelope import rms_dbfs
from callclarity.dsp.limiter import limit_peak
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("leveler", "speech_aware_agc")
class SpeechAwareAgc(BaseStreamingProcessor):
    name = "speech_aware_agc"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.target = float(self.config.get("target_speech_rms_dbfs", -20.0))
        self.max_boost = float(self.config.get("max_boost_db", 15.0))
        self.max_cut = float(self.config.get("max_cut_db", 18.0))
        self.attack = float(self.config.get("attack_db_per_sec", 10.0))
        self.release = float(self.config.get("release_db_per_sec", 2.0))
        self.vad_threshold = float(self.config.get("vad_threshold", 0.55))
        self.freeze_on_silence = bool(self.config.get("freeze_gain_on_silence", True))
        self.current_gain_db = 0.0
        self.last_speech_rms_dbfs: float | None = None
        self.max_silence_gain_db = 0.0

    def reset(self) -> None:
        self.current_gain_db = 0.0
        self.last_speech_rms_dbfs = None
        self.max_silence_gain_db = 0.0

    def _slew(self, target_gain: float, dt: float) -> None:
        delta = target_gain - self.current_gain_db
        rate = self.attack if delta > 0 else self.release
        step = rate * dt
        if abs(delta) <= step:
            self.current_gain_db = target_gain
        else:
            self.current_gain_db += step if delta > 0 else -step

    def process(self, chunk: AudioChunk) -> ProcessResult:
        speech_prob = float(chunk.metadata.get("speech_prob", 1.0 if not self.config.get("vad_required", True) else 0.0))
        is_speech = speech_prob >= self.vad_threshold
        dt = chunk.duration_sec
        speech_rms = rms_dbfs(chunk.samples)
        target_error = 0.0
        if is_speech:
            self.last_speech_rms_dbfs = speech_rms
            desired = self.target - speech_rms
            desired = min(self.max_boost, max(-self.max_cut, desired))
            target_error = self.target - speech_rms
            self._slew(desired, dt)
        elif not self.freeze_on_silence and self.last_speech_rms_dbfs is not None:
            desired = min(self.max_boost, max(-self.max_cut, self.target - self.last_speech_rms_dbfs))
            self._slew(desired, dt)
        else:
            self.max_silence_gain_db = max(self.max_silence_gain_db, self.current_gain_db)

        y = apply_gain_db(chunk.samples, self.current_gain_db)
        comp_cfg = self.config.get("compressor", {})
        comp_metrics = {"compressor_gain_reduction_db": 0.0}
        if comp_cfg.get("enabled", True):
            y, comp_metrics = static_compress(
                y,
                threshold_dbfs=float(comp_cfg.get("threshold_dbfs", -22.0)),
                ratio=float(comp_cfg.get("ratio", 2.5)),
            )
        lim_cfg = self.config.get("limiter", {})
        limiter_metrics = {"clipping_count": 0, "limiter_gain_reduction_db": 0.0}
        if lim_cfg.get("enabled", True):
            y, limiter_metrics = limit_peak(y, float(lim_cfg.get("ceiling_dbfs", -1.5)))
        metadata = merge_metadata(chunk, agc_gain_db=self.current_gain_db)
        silence_noise_boost_db = max(0.0, self.current_gain_db if not is_speech else 0.0)
        return ProcessResult(
            chunk=AudioChunk(y.contiguous(), chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "current_gain_db": self.current_gain_db,
                "speech_rms_dbfs": speech_rms if is_speech else None,
                "target_error_db": target_error if is_speech else None,
                "silence_noise_boost_db": silence_noise_boost_db,
                **comp_metrics,
                **limiter_metrics,
            },
            algorithmic_latency_ms=0.0,
        )
