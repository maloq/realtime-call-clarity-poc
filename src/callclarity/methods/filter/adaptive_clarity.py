from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


_EPS = 1e-12


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _linear_to_db(value: float) -> float:
    return 20.0 * float(np.log10(max(float(value), _EPS)))


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -240.0
    finite = np.nan_to_num(
        x.astype(np.float64, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return _linear_to_db(float(np.sqrt(np.mean(np.square(finite)) + _EPS)))


def _clamped_freq(freq_hz: float, sample_rate: int) -> float:
    nyquist = float(sample_rate) / 2.0
    margin = max(1.0, nyquist * 0.01)
    return float(min(max(float(freq_hz), margin), max(margin, nyquist - margin)))


def _smooth_gain_db(
    current: float,
    target: float,
    dt: float,
    attack_ms: float,
    release_ms: float,
) -> float:
    tau_ms = float(attack_ms) if abs(target) > abs(current) else float(release_ms)
    if tau_ms <= 0.0 or dt <= 0.0:
        return float(target)
    alpha = 1.0 - float(np.exp(-float(dt) / (tau_ms / 1000.0)))
    alpha = max(0.0, min(1.0, alpha))
    return float(current + alpha * (target - current))


def _peaking_eq(center_hz: float, q: float, gain_db: float, sample_rate: int) -> np.ndarray:
    center = _clamped_freq(center_hz, sample_rate)
    q = max(float(q), 0.05)
    gain_db = float(gain_db)
    if abs(gain_db) < 1e-6:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    a = _db_to_linear(gain_db / 2.0)
    omega = 2.0 * np.pi * center / float(sample_rate)
    alpha = np.sin(omega) / (2.0 * q)
    cos_omega = np.cos(omega)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / a

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float32,
    )


def _empty_iir_state(channels: int, length: int) -> np.ndarray:
    return np.zeros((int(length), int(channels), 2), dtype=np.float32)


def _as_2d(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = samples.shape
    if samples.ndim == 1:
        return samples.unsqueeze(0), original_shape
    if samples.ndim == 2:
        return samples, original_shape
    return samples.reshape(-1, samples.shape[-1]), original_shape


def _restore_shape(samples: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    return samples.reshape(original_shape)


def _clamped_band(
    band_hz: list[float] | tuple[float, float],
    sample_rate: int,
) -> tuple[float, float]:
    low = _clamped_freq(float(band_hz[0]), sample_rate)
    high = _clamped_freq(float(band_hz[1]), sample_rate)
    if high <= low:
        high = _clamped_freq(low + max(20.0, float(sample_rate) * 0.02), sample_rate)
    if high <= low:
        low = max(1.0, high * 0.5)
    return low, high


def _bandpass_sos(
    band_hz: list[float] | tuple[float, float],
    sample_rate: int,
    order: int = 3,
) -> np.ndarray:
    low, high = _clamped_band(band_hz, sample_rate)
    return signal.butter(
        int(order),
        [low, high],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float32)


def _peak_np(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


@dataclass
class _IirFilter:
    label: str
    sos: np.ndarray
    state: np.ndarray

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.state = signal.sosfilt(self.sos, x, axis=-1, zi=self.state)
        return y.astype(np.float32, copy=False)


@dataclass
class _BandState:
    label: str
    sos: np.ndarray
    state: np.ndarray
    current_gain_db: float = 0.0
    current_reduction_db: float = 0.0
    previous_level_dbfs: float | None = None

    def band(self, x: np.ndarray) -> np.ndarray:
        y, self.state = signal.sosfilt(self.sos, x, axis=-1, zi=self.state)
        return y.astype(np.float32, copy=False)


@register_method("filter", "adaptive_clarity")
class AdaptiveClarityProcessor(BaseStreamingProcessor):
    name = "adaptive_clarity"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.analysis_cfg = self.config.get("analysis", {})
        self.filters_cfg = self.config.get("filters", {})
        self.final_cfg = self.config.get("final", {})

        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.highpass: _IirFilter | None = None
        self.static_filters: list[_IirFilter] = []
        self.demud: _BandState | None = None
        self.demud_compare: _BandState | None = None
        self.presence: _BandState | None = None
        self.presence_reference: _BandState | None = None
        self.consonant: _BandState | None = None
        self.harshness: _BandState | None = None
        self.deess: _BandState | None = None

        self.demud_cut_db = 0.0
        self.presence_boost_db = 0.0
        self.consonant_boost_db = 0.0
        self.harshness_cut_db = 0.0
        self.deess_cut_db = 0.0
        self.saturation_mix = 0.0

    @property
    def algorithmic_latency_ms(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.sample_rate = None
        self.channels = None
        self.highpass = None
        self.static_filters = []
        self.demud = None
        self.demud_compare = None
        self.presence = None
        self.presence_reference = None
        self.consonant = None
        self.harshness = None
        self.deess = None
        self.demud_cut_db = 0.0
        self.presence_boost_db = 0.0
        self.consonant_boost_db = 0.0
        self.harshness_cut_db = 0.0
        self.deess_cut_db = 0.0
        self.saturation_mix = 0.0

    def warmup(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)

    def _filter_from_sos(self, label: str, sos: np.ndarray, channels: int) -> _IirFilter:
        return _IirFilter(label=label, sos=sos, state=_empty_iir_state(channels, sos.shape[0]))

    def _band_state(self, label: str, sos: np.ndarray, channels: int) -> _BandState:
        return _BandState(label=label, sos=sos, state=_empty_iir_state(channels, sos.shape[0]))

    def _configure(self, sample_rate: int, channels: int) -> None:
        if self.sample_rate == sample_rate and self.channels == channels and self.demud is not None:
            return

        self.sample_rate = sample_rate
        self.channels = channels
        self.static_filters = []

        hp_cfg = self.filters_cfg.get("highpass", {})
        if bool(hp_cfg.get("enabled", True)):
            cutoff = _clamped_freq(float(hp_cfg.get("cutoff_hz", 80.0)), sample_rate)
            order = max(1, int(hp_cfg.get("order", 2)))
            sos = signal.butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")
            self.highpass = self._filter_from_sos("highpass", sos.astype(np.float32), channels)
        else:
            self.highpass = None

        static_cfg = self.filters_cfg.get("static_eq", {})
        if bool(static_cfg.get("enabled", True)):
            mud_sos = _peaking_eq(
                float(static_cfg.get("mud_center_hz", 320.0)),
                float(static_cfg.get("mud_q", 0.9)),
                float(static_cfg.get("mud_gain_db", -1.0)),
                sample_rate,
            )
            presence_sos = _peaking_eq(
                float(static_cfg.get("presence_center_hz", 2400.0)),
                float(static_cfg.get("presence_q", 0.85)),
                float(static_cfg.get("presence_gain_db", 0.9)),
                sample_rate,
            )
            self.static_filters.append(self._filter_from_sos("static_mud", mud_sos, channels))
            self.static_filters.append(
                self._filter_from_sos("static_presence", presence_sos, channels)
            )

        demud_cfg = self.filters_cfg.get("dynamic_demud", {})
        self.demud = self._band_state(
            "demud",
            _bandpass_sos(demud_cfg.get("band_hz", [180.0, 560.0]), sample_rate),
            channels,
        )
        self.demud_compare = self._band_state(
            "demud_compare",
            _bandpass_sos(demud_cfg.get("compare_band_hz", [1200.0, 4200.0]), sample_rate),
            channels,
        )

        presence_cfg = self.filters_cfg.get("dynamic_presence", {})
        self.presence = self._band_state(
            "presence",
            _bandpass_sos(presence_cfg.get("band_hz", [1600.0, 3600.0]), sample_rate),
            channels,
        )
        self.presence_reference = self._band_state(
            "presence_reference",
            _bandpass_sos(presence_cfg.get("reference_band_hz", [250.0, 1200.0]), sample_rate),
            channels,
        )

        consonant_cfg = self.filters_cfg.get("consonant_lift", {})
        self.consonant = self._band_state(
            "consonant",
            _bandpass_sos(consonant_cfg.get("band_hz", [3200.0, 5200.0]), sample_rate),
            channels,
        )

        harsh_cfg = self.filters_cfg.get("harshness_guard", {})
        self.harshness = self._band_state(
            "harshness",
            _bandpass_sos(harsh_cfg.get("band_hz", [2800.0, 5200.0]), sample_rate),
            channels,
        )

        deess_cfg = self.filters_cfg.get("deesser", {})
        self.deess = self._band_state(
            "deess",
            _bandpass_sos(deess_cfg.get("band_hz", [5000.0, 7600.0]), sample_rate),
            channels,
        )

    def _speech_active(self, chunk: AudioChunk, input_rms_dbfs: float) -> bool:
        threshold = float(self.analysis_cfg.get("speech_prob_threshold", 0.35))
        if "speech_prob" in chunk.metadata:
            return float(chunk.metadata["speech_prob"]) >= threshold
        min_rms = float(self.analysis_cfg.get("min_active_rms_dbfs", -58.0))
        return input_rms_dbfs >= min_rms

    def _apply_demud(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float, float]:
        cfg = self.filters_cfg.get("dynamic_demud", {})
        assert self.demud is not None
        assert self.demud_compare is not None
        mud_band = self.demud.band(y)
        compare_band = self.demud_compare.band(y)
        mud_dbfs = _rms_db(mud_band)
        compare_dbfs = _rms_db(compare_band)

        target_cut = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            dominance = mud_dbfs - compare_dbfs
            threshold = float(cfg.get("ratio_threshold_db", 5.0))
            if dominance > threshold:
                target_cut = min(float(cfg.get("max_cut_db", 3.0)), dominance - threshold)

        self.demud_cut_db = _smooth_gain_db(
            self.demud_cut_db,
            target_cut,
            dt,
            float(cfg.get("attack_ms", 15.0)),
            float(cfg.get("release_ms", 180.0)),
        )
        self.demud.current_reduction_db = self.demud_cut_db
        gain = _db_to_linear(-self.demud_cut_db)
        y = y + (gain - 1.0) * mud_band
        return y.astype(np.float32, copy=False), mud_dbfs, compare_dbfs

    def _apply_presence(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        cfg = self.filters_cfg.get("dynamic_presence", {})
        assert self.presence is not None
        assert self.presence_reference is not None
        presence_band = self.presence.band(y)
        reference_band = self.presence_reference.band(y)
        presence_dbfs = _rms_db(presence_band)
        reference_dbfs = _rms_db(reference_band)

        target_boost = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            missing = reference_dbfs - presence_dbfs
            threshold = float(cfg.get("missing_threshold_db", 4.0))
            noise_guard = float(cfg.get("noise_guard_high_band_dbfs", -48.0))
            if missing > threshold and presence_dbfs < noise_guard:
                target_boost = min(float(cfg.get("max_boost_db", 2.0)), missing - threshold)
                if self.harshness_cut_db > 1.0:
                    target_boost *= max(0.0, 1.0 - self.harshness_cut_db / 4.0)

        self.presence_boost_db = _smooth_gain_db(
            self.presence_boost_db,
            target_boost,
            dt,
            float(cfg.get("attack_ms", 20.0)),
            float(cfg.get("release_ms", 220.0)),
        )
        self.presence.current_gain_db = self.presence_boost_db
        gain = _db_to_linear(self.presence_boost_db)
        y = y + (gain - 1.0) * presence_band
        return y.astype(np.float32, copy=False), presence_dbfs

    def _apply_consonant_lift(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        cfg = self.filters_cfg.get("consonant_lift", {})
        assert self.consonant is not None
        consonant_band = self.consonant.band(y)
        consonant_dbfs = _rms_db(consonant_band)

        allow_for_activity = speech_active or not bool(cfg.get("only_when_speech", True))
        target_boost = 0.0
        if bool(cfg.get("enabled", True)) and allow_for_activity:
            max_boost = float(cfg.get("max_boost_db", 0.9))
            target_boost = max_boost
            if consonant_dbfs > -34.0:
                target_boost *= max(0.0, min(1.0, (-30.0 - consonant_dbfs) / 4.0))
            if self.consonant.previous_level_dbfs is not None:
                jump_db = consonant_dbfs - self.consonant.previous_level_dbfs
                if jump_db > 2.0:
                    extra = float(cfg.get("transient_extra_db", 0.35))
                    target_boost += extra * max(0.0, min(1.0, (jump_db - 2.0) / 8.0))
            target_boost = min(
                target_boost,
                max_boost + float(cfg.get("transient_extra_db", 0.35)),
            )

        self.consonant_boost_db = _smooth_gain_db(
            self.consonant_boost_db,
            target_boost,
            dt,
            float(cfg.get("attack_ms", 6.0)),
            float(cfg.get("release_ms", 100.0)),
        )
        self.consonant.current_gain_db = self.consonant_boost_db
        self.consonant.previous_level_dbfs = consonant_dbfs
        gain = _db_to_linear(self.consonant_boost_db)
        y = y + (gain - 1.0) * consonant_band
        return y.astype(np.float32, copy=False), consonant_dbfs

    def _apply_reduction_band(
        self,
        y: np.ndarray,
        band: _BandState,
        cfg: dict[str, Any],
        current: float,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float, float]:
        band_signal = band.band(y)
        band_dbfs = _rms_db(band_signal)

        target_cut = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            default_threshold = -39.0 if band.label == "harshness" else -52.0
            default_max_cut = 5.0 if band.label == "harshness" else 6.0
            threshold = float(cfg.get("threshold_dbfs", default_threshold))
            if band_dbfs > threshold:
                target_cut = min(float(cfg.get("max_cut_db", default_max_cut)), band_dbfs - threshold)

        smoothed_cut = _smooth_gain_db(
            current,
            target_cut,
            dt,
            float(cfg.get("attack_ms", 4.0)),
            float(cfg.get("release_ms", 120.0)),
        )
        band.current_reduction_db = smoothed_cut
        gain = _db_to_linear(-smoothed_cut)
        y = y + (gain - 1.0) * band_signal
        return y.astype(np.float32, copy=False), band_dbfs, smoothed_cut

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not self.enabled:
            return super().process(chunk)

        x_float, original_shape = _as_2d(chunk.samples.detach().float())
        x_np = x_float.cpu().numpy().astype(np.float32, copy=False)
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        dry_np = x_np.copy()
        input_rms_dbfs = _rms_db(x_np)
        speech_active = self._speech_active(chunk, input_rms_dbfs)

        self._configure(chunk.sample_rate, int(x_np.shape[0]))
        y = x_np

        if self.highpass is not None:
            y = self.highpass.process(y)

        for static_filter in self.static_filters:
            y = static_filter.process(y)

        y, mud_band_dbfs, _compare_dbfs = self._apply_demud(
            y,
            speech_active,
            chunk.duration_sec,
        )
        y, presence_band_dbfs = self._apply_presence(
            y,
            speech_active,
            chunk.duration_sec,
        )
        y, consonant_band_dbfs = self._apply_consonant_lift(
            y,
            speech_active,
            chunk.duration_sec,
        )

        assert self.harshness is not None
        harsh_cfg = self.filters_cfg.get("harshness_guard", {})
        y, _harshness_dbfs, self.harshness_cut_db = self._apply_reduction_band(
            y,
            self.harshness,
            harsh_cfg,
            self.harshness_cut_db,
            speech_active,
            chunk.duration_sec,
        )

        assert self.deess is not None
        deess_cfg = self.filters_cfg.get("deesser", {})
        y, deess_band_dbfs, self.deess_cut_db = self._apply_reduction_band(
            y,
            self.deess,
            deess_cfg,
            self.deess_cut_db,
            speech_active,
            chunk.duration_sec,
        )

        sat_cfg = self.filters_cfg.get("saturation", {})
        self.saturation_mix = 0.0
        if bool(sat_cfg.get("enabled", False)) and speech_active:
            peak = _peak_np(y)
            disable_above = float(sat_cfg.get("disable_above_peak", 0.92))
            if peak < disable_above:
                drive = max(1e-3, float(sat_cfg.get("drive", 1.25)))
                mix = max(0.0, min(1.0, float(sat_cfg.get("mix", 0.0))))
                wet = np.tanh(drive * y) / max(float(np.tanh(drive)), _EPS)
                y = y * (1.0 - mix) + wet.astype(np.float32, copy=False) * mix
                self.saturation_mix = mix

        dry_wet = max(0.0, min(1.0, float(self.final_cfg.get("dry_wet", 1.0))))
        if dry_wet < 1.0:
            y = dry_np * (1.0 - dry_wet) + y * dry_wet

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        ceiling = abs(float(self.final_cfg.get("ceiling", 0.98)))
        if ceiling > 0.0:
            y = np.clip(y, -ceiling, ceiling)

        output_rms_dbfs = _rms_db(y)
        output_peak = _peak_np(y)
        out = torch.from_numpy(y).to(device=chunk.samples.device, dtype=chunk.samples.dtype)
        out = _restore_shape(out, original_shape).contiguous()

        metadata = merge_metadata(
            chunk,
            adaptive_clarity=True,
            clarity_speech_active=speech_active,
            demud_cut_db=self.demud_cut_db,
            presence_boost_db=self.presence_boost_db,
            consonant_boost_db=self.consonant_boost_db,
        )
        metrics = {
            "adaptive_clarity_enabled": True,
            "clarity_speech_active": speech_active,
            "clarity_input_rms_dbfs": input_rms_dbfs,
            "clarity_output_rms_dbfs": output_rms_dbfs,
            "demud_cut_db": self.demud_cut_db,
            "presence_boost_db": self.presence_boost_db,
            "consonant_boost_db": self.consonant_boost_db,
            "harshness_cut_db": self.harshness_cut_db,
            "deess_cut_db": self.deess_cut_db,
            "saturation_mix": self.saturation_mix,
            "clarity_output_peak": output_peak,
            "mud_band_dbfs": mud_band_dbfs,
            "presence_band_dbfs": presence_band_dbfs,
            "consonant_band_dbfs": consonant_band_dbfs,
            "deess_band_dbfs": deess_band_dbfs,
            "realtime_safe": True,
        }
        return ProcessResult(
            chunk=AudioChunk(
                out,
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics=metrics,
            algorithmic_latency_ms=0.0,
        )
