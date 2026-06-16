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


def _peak_np(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _clamped_freq(freq_hz: float, sample_rate: int) -> float:
    nyquist = float(sample_rate) / 2.0
    margin = max(1.0, nyquist * 0.01)
    return float(min(max(float(freq_hz), margin), max(margin, nyquist - margin)))


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


def _smooth_db(
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
    return float(current + _clamp(alpha, 0.0, 1.0) * (target - current))


def _empty_iir_state(channels: int, sections: int) -> np.ndarray:
    return np.zeros((int(sections), int(channels), 2), dtype=np.float32)


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


def _lowpass_sos(cutoff_hz: float, sample_rate: int, order: int = 3) -> np.ndarray:
    cutoff = _clamped_freq(cutoff_hz, sample_rate)
    return signal.butter(
        int(order),
        cutoff,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float32)


def _highpass_sos(cutoff_hz: float, sample_rate: int, order: int = 2) -> np.ndarray:
    cutoff = _clamped_freq(cutoff_hz, sample_rate)
    return signal.butter(
        int(order),
        cutoff,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float32)


def _as_2d(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = samples.shape
    if samples.ndim == 1:
        return samples.unsqueeze(0), original_shape
    if samples.ndim == 2:
        return samples, original_shape
    return samples.reshape(-1, samples.shape[-1]), original_shape


def _restore_shape(samples: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    return samples.reshape(original_shape)


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

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.state = signal.sosfilt(self.sos, x, axis=-1, zi=self.state)
        return y.astype(np.float32, copy=False)


@register_method("filter", "dpdfnet_naturalize")
class DpdfnetNaturalizeProcessor(BaseStreamingProcessor):
    name = "dpdfnet_naturalize"
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
        self.body: _BandState | None = None
        self.body_reference: _BandState | None = None
        self.lower_presence: _BandState | None = None
        self.presence_reference: _BandState | None = None
        self.phone: _BandState | None = None
        self.phone_reference: _BandState | None = None
        self.metal: _BandState | None = None
        self.deess: _BandState | None = None
        self.density_band: _BandState | None = None
        self.density_smoother: _IirFilter | None = None

        self.body_boost_db = 0.0
        self.lower_presence_boost_db = 0.0
        self.phone_cut_db = 0.0
        self.metal_cut_db = 0.0
        self.deess_cut_db = 0.0
        self.density_mix = 0.0

    @property
    def algorithmic_latency_ms(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.sample_rate = None
        self.channels = None
        self.highpass = None
        self.body = None
        self.body_reference = None
        self.lower_presence = None
        self.presence_reference = None
        self.phone = None
        self.phone_reference = None
        self.metal = None
        self.deess = None
        self.density_band = None
        self.density_smoother = None
        self.body_boost_db = 0.0
        self.lower_presence_boost_db = 0.0
        self.phone_cut_db = 0.0
        self.metal_cut_db = 0.0
        self.deess_cut_db = 0.0
        self.density_mix = 0.0

    def warmup(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)

    def _iir(self, label: str, sos: np.ndarray, channels: int) -> _IirFilter:
        return _IirFilter(label=label, sos=sos, state=_empty_iir_state(channels, sos.shape[0]))

    def _band(self, label: str, sos: np.ndarray, channels: int) -> _BandState:
        return _BandState(label=label, sos=sos, state=_empty_iir_state(channels, sos.shape[0]))

    def _configure(self, sample_rate: int, channels: int) -> None:
        if self.sample_rate == sample_rate and self.channels == channels and self.body is not None:
            return

        self.sample_rate = sample_rate
        self.channels = channels
        hp_cfg = self.filters_cfg.get("highpass", {})
        if bool(hp_cfg.get("enabled", True)):
            self.highpass = self._iir(
                "highpass",
                _highpass_sos(
                    float(hp_cfg.get("cutoff_hz", 65.0)),
                    sample_rate,
                    int(hp_cfg.get("order", 2)),
                ),
                channels,
            )
        else:
            self.highpass = None

        body_cfg = self.filters_cfg.get("body", {})
        self.body = self._band(
            "body",
            _bandpass_sos(body_cfg.get("band_hz", [140.0, 420.0]), sample_rate),
            channels,
        )
        self.body_reference = self._band(
            "body_reference",
            _bandpass_sos(body_cfg.get("reference_band_hz", [900.0, 2400.0]), sample_rate),
            channels,
        )

        presence_cfg = self.filters_cfg.get("lower_presence", {})
        self.lower_presence = self._band(
            "lower_presence",
            _bandpass_sos(presence_cfg.get("band_hz", [850.0, 2200.0]), sample_rate),
            channels,
        )
        self.presence_reference = self._band(
            "presence_reference",
            _bandpass_sos(presence_cfg.get("reference_band_hz", [180.0, 900.0]), sample_rate),
            channels,
        )

        phone_cfg = self.filters_cfg.get("phone_soften", {})
        self.phone = self._band(
            "phone",
            _bandpass_sos(phone_cfg.get("band_hz", [2400.0, 4200.0]), sample_rate),
            channels,
        )
        self.phone_reference = self._band(
            "phone_reference",
            _bandpass_sos(phone_cfg.get("reference_band_hz", [140.0, 1200.0]), sample_rate),
            channels,
        )

        metal_cfg = self.filters_cfg.get("metal_smoother", {})
        self.metal = self._band(
            "metal",
            _bandpass_sos(metal_cfg.get("band_hz", [3600.0, 7600.0]), sample_rate),
            channels,
        )

        deess_cfg = self.filters_cfg.get("deesser", {})
        self.deess = self._band(
            "deess",
            _bandpass_sos(deess_cfg.get("band_hz", [5200.0, 7600.0]), sample_rate),
            channels,
        )

        density_cfg = self.filters_cfg.get("density", {})
        self.density_band = self._band(
            "density",
            _bandpass_sos(density_cfg.get("band_hz", [120.0, 2400.0]), sample_rate),
            channels,
        )
        self.density_smoother = self._iir(
            "density_smoother",
            _lowpass_sos(float(density_cfg.get("detail_lowpass_hz", 2600.0)), sample_rate),
            channels,
        )

    def _speech_active(self, chunk: AudioChunk, input_rms_dbfs: float) -> bool:
        threshold = float(self.analysis_cfg.get("speech_prob_threshold", 0.35))
        if "speech_prob" in chunk.metadata:
            return float(chunk.metadata["speech_prob"]) >= threshold
        min_rms = float(self.analysis_cfg.get("min_active_rms_dbfs", -58.0))
        return input_rms_dbfs >= min_rms

    def _smooth_control(
        self,
        current: float,
        target: float,
        dt: float,
        cfg: dict[str, Any],
        attack_default: float,
        release_default: float,
    ) -> float:
        return _smooth_db(
            current,
            target,
            dt,
            float(cfg.get("attack_ms", attack_default)),
            float(cfg.get("release_ms", release_default)),
        )

    def _parallel_gain(self, y: np.ndarray, band: np.ndarray, gain_db: float) -> np.ndarray:
        gain = _db_to_linear(gain_db)
        return (y + (gain - 1.0) * band).astype(np.float32, copy=False)

    def _parallel_cut(self, y: np.ndarray, band: np.ndarray, cut_db: float) -> np.ndarray:
        gain = _db_to_linear(-cut_db)
        return (y + (gain - 1.0) * band).astype(np.float32, copy=False)

    def _apply_body(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float, float]:
        cfg = self.filters_cfg.get("body", {})
        assert self.body is not None
        assert self.body_reference is not None
        body_band = self.body.process(y)
        reference_band = self.body_reference.process(y)
        body_dbfs = _rms_db(body_band)
        reference_dbfs = _rms_db(reference_band)

        target = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            thinness = reference_dbfs - body_dbfs
            base = float(cfg.get("base_boost_db", 1.6))
            threshold = float(cfg.get("thinness_threshold_db", 5.0))
            target = base
            if thinness > threshold:
                target += 0.35 * (thinness - threshold)
            target = min(float(cfg.get("max_boost_db", 3.2)), target)

        self.body_boost_db = self._smooth_control(self.body_boost_db, target, dt, cfg, 30.0, 240.0)
        self.body.current_gain_db = self.body_boost_db
        return self._parallel_gain(y, body_band, self.body_boost_db), body_dbfs, reference_dbfs

    def _apply_lower_presence(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float, float]:
        cfg = self.filters_cfg.get("lower_presence", {})
        assert self.lower_presence is not None
        assert self.presence_reference is not None
        presence_band = self.lower_presence.process(y)
        reference_band = self.presence_reference.process(y)
        presence_dbfs = _rms_db(presence_band)
        reference_dbfs = _rms_db(reference_band)

        target = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            missing = reference_dbfs - presence_dbfs
            base = float(cfg.get("base_boost_db", 0.6))
            threshold = float(cfg.get("missing_threshold_db", 3.0))
            target = base
            if missing > threshold:
                target += 0.25 * (missing - threshold)
            target = min(float(cfg.get("max_boost_db", 1.5)), target)

        self.lower_presence_boost_db = self._smooth_control(
            self.lower_presence_boost_db,
            target,
            dt,
            cfg,
            35.0,
            260.0,
        )
        self.lower_presence.current_gain_db = self.lower_presence_boost_db
        return (
            self._parallel_gain(y, presence_band, self.lower_presence_boost_db),
            presence_dbfs,
            reference_dbfs,
        )

    def _apply_phone_soften(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float, float]:
        cfg = self.filters_cfg.get("phone_soften", {})
        assert self.phone is not None
        assert self.phone_reference is not None
        phone_band = self.phone.process(y)
        reference_band = self.phone_reference.process(y)
        phone_dbfs = _rms_db(phone_band)
        reference_dbfs = _rms_db(reference_band)

        target = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            target = float(cfg.get("base_cut_db", 0.5))
            dominance = phone_dbfs - reference_dbfs
            threshold = float(cfg.get("dominance_threshold_db", 8.0))
            if dominance > threshold:
                target += 0.4 * (dominance - threshold)
            target = min(float(cfg.get("max_cut_db", 2.4)), target)

        self.phone_cut_db = self._smooth_control(self.phone_cut_db, target, dt, cfg, 12.0, 180.0)
        self.phone.current_reduction_db = self.phone_cut_db
        return self._parallel_cut(y, phone_band, self.phone_cut_db), phone_dbfs, reference_dbfs

    def _apply_metal_smoother(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        cfg = self.filters_cfg.get("metal_smoother", {})
        assert self.metal is not None
        metal_band = self.metal.process(y)
        metal_dbfs = _rms_db(metal_band)

        target = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            floor = float(cfg.get("floor_dbfs", -90.0))
            if metal_dbfs > floor:
                target = float(cfg.get("base_cut_db", 0.9))
                threshold = float(cfg.get("threshold_dbfs", -62.0))
                if metal_dbfs > threshold:
                    target += 0.4 * (metal_dbfs - threshold)
                target = min(float(cfg.get("max_cut_db", 3.0)), target)

        self.metal_cut_db = self._smooth_control(self.metal_cut_db, target, dt, cfg, 8.0, 160.0)
        self.metal.current_reduction_db = self.metal_cut_db
        return self._parallel_cut(y, metal_band, self.metal_cut_db), metal_dbfs

    def _apply_deesser(
        self,
        y: np.ndarray,
        speech_active: bool,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        cfg = self.filters_cfg.get("deesser", {})
        assert self.deess is not None
        deess_band = self.deess.process(y)
        deess_dbfs = _rms_db(deess_band)

        target = 0.0
        if bool(cfg.get("enabled", True)) and speech_active:
            threshold = float(cfg.get("threshold_dbfs", -58.0))
            if deess_dbfs > threshold:
                target = min(float(cfg.get("max_cut_db", 3.0)), 0.5 * (deess_dbfs - threshold))

        self.deess_cut_db = self._smooth_control(self.deess_cut_db, target, dt, cfg, 4.0, 110.0)
        self.deess.current_reduction_db = self.deess_cut_db
        return self._parallel_cut(y, deess_band, self.deess_cut_db), deess_dbfs

    def _apply_density(self, y: np.ndarray, speech_active: bool) -> np.ndarray:
        cfg = self.filters_cfg.get("density", {})
        self.density_mix = 0.0
        if not (bool(cfg.get("enabled", True)) and speech_active):
            return y
        if _peak_np(y) >= float(cfg.get("disable_above_peak", 0.92)):
            return y
        assert self.density_band is not None
        assert self.density_smoother is not None
        low_mid = self.density_band.process(y)
        drive = max(1e-3, float(cfg.get("drive", 1.15)))
        mix = _clamp(float(cfg.get("mix", 0.025)), 0.0, 1.0)
        if mix <= 0.0:
            return y
        saturated = np.tanh(drive * low_mid) / max(float(np.tanh(drive)), _EPS)
        detail = self.density_smoother.process((saturated - low_mid).astype(np.float32, copy=False))
        self.density_mix = mix
        return (y + mix * detail).astype(np.float32, copy=False)

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

        y, body_dbfs, body_reference_dbfs = self._apply_body(y, speech_active, chunk.duration_sec)
        y, lower_presence_dbfs, presence_reference_dbfs = self._apply_lower_presence(
            y,
            speech_active,
            chunk.duration_sec,
        )
        y, phone_band_dbfs, phone_reference_dbfs = self._apply_phone_soften(
            y,
            speech_active,
            chunk.duration_sec,
        )
        y, metal_band_dbfs = self._apply_metal_smoother(y, speech_active, chunk.duration_sec)
        y, deess_band_dbfs = self._apply_deesser(y, speech_active, chunk.duration_sec)
        y = self._apply_density(y, speech_active)

        dry_wet = _clamp(float(self.final_cfg.get("dry_wet", 1.0)), 0.0, 1.0)
        if dry_wet < 1.0:
            y = dry_np * (1.0 - dry_wet) + y * dry_wet
        output_gain_db = float(self.final_cfg.get("output_gain_db", 0.0))
        if abs(output_gain_db) > 1e-6:
            y = y * _db_to_linear(output_gain_db)

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
            dpdfnet_naturalize=True,
            naturalize_speech_active=speech_active,
            body_boost_db=self.body_boost_db,
            lower_presence_boost_db=self.lower_presence_boost_db,
            phone_cut_db=self.phone_cut_db,
            metal_cut_db=self.metal_cut_db,
        )
        metrics = {
            "dpdfnet_naturalize_enabled": True,
            "naturalize_speech_active": speech_active,
            "naturalize_input_rms_dbfs": input_rms_dbfs,
            "naturalize_output_rms_dbfs": output_rms_dbfs,
            "body_boost_db": self.body_boost_db,
            "lower_presence_boost_db": self.lower_presence_boost_db,
            "phone_cut_db": self.phone_cut_db,
            "metal_cut_db": self.metal_cut_db,
            "deess_cut_db": self.deess_cut_db,
            "density_mix": self.density_mix,
            "naturalize_output_peak": output_peak,
            "body_band_dbfs": body_dbfs,
            "body_reference_dbfs": body_reference_dbfs,
            "lower_presence_dbfs": lower_presence_dbfs,
            "presence_reference_dbfs": presence_reference_dbfs,
            "phone_band_dbfs": phone_band_dbfs,
            "phone_reference_dbfs": phone_reference_dbfs,
            "metal_band_dbfs": metal_band_dbfs,
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
