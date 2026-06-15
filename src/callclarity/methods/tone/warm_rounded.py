from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


_MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "safe_warm": {
        "leveler": {
            "enabled": True,
            "target_rms_dbfs": -25.0,
            "max_boost_db": 5.0,
            "max_cut_db": 6.0,
            "attack_db_per_sec": 8.0,
            "release_db_per_sec": 4.0,
            "silence_release_db_per_sec": 8.0,
            "vad_threshold": 0.45,
            "min_active_dbfs": -54.0,
        },
        "eq": {
            "warmth": {"enabled": True, "center_hz": 170.0, "q": 0.8, "gain_db": 1.0},
            "mud": {"enabled": True, "center_hz": 320.0, "q": 1.0, "gain_db": -1.5},
            "nasal": {"enabled": False, "center_hz": 900.0, "q": 1.0, "gain_db": -1.0},
            "clarity": {"enabled": True, "center_hz": 2500.0, "q": 0.9, "gain_db": 0.5},
        },
        "dynamic_eq": {
            "harsh": {
                "enabled": True,
                "low_hz": 2800.0,
                "high_hz": 4200.0,
                "threshold_dbfs": -43.0,
                "max_reduction_db": 2.0,
                "slope": 0.45,
                "attack_ms": 4.0,
                "release_ms": 100.0,
            },
            "deess": {
                "enabled": True,
                "low_hz": 5000.0,
                "high_hz": 7200.0,
                "threshold_dbfs": -47.0,
                "max_reduction_db": 1.5,
                "slope": 0.45,
                "attack_ms": 3.0,
                "release_ms": 90.0,
            },
        },
        "saturation": {
            "enabled": True,
            "drive": 1.35,
            "mix": 0.05,
            "reduce_if_peak_above": 0.92,
            "reduce_if_decrackled": True,
        },
        "compressor": {
            "enabled": True,
            "threshold_dbfs": -24.0,
            "ratio": 2.0,
            "attack_ms": 20.0,
            "release_ms": 170.0,
            "knee_db": 6.0,
            "makeup_gain_db": 0.5,
        },
        "final_eq": {
            "warmth": {"enabled": False, "center_hz": 160.0, "q": 0.8, "gain_db": 0.0},
            "mud": {"enabled": False, "center_hz": 300.0, "q": 1.0, "gain_db": 0.0},
            "sharp": {"enabled": False, "center_hz": 3500.0, "q": 1.0, "gain_db": 0.0},
            "clarity": {"enabled": False, "center_hz": 2500.0, "q": 0.9, "gain_db": 0.0},
        },
    },
    "warm_smooth": {
        "leveler": {
            "enabled": True,
            "target_rms_dbfs": -24.0,
            "max_boost_db": 6.0,
            "max_cut_db": 7.0,
            "attack_db_per_sec": 10.0,
            "release_db_per_sec": 4.0,
            "silence_release_db_per_sec": 9.0,
            "vad_threshold": 0.45,
            "min_active_dbfs": -54.0,
        },
        "eq": {
            "warmth": {"enabled": True, "center_hz": 170.0, "q": 0.8, "gain_db": 2.0},
            "mud": {"enabled": True, "center_hz": 320.0, "q": 1.0, "gain_db": -2.5},
            "nasal": {"enabled": False, "center_hz": 900.0, "q": 1.0, "gain_db": -1.5},
            "clarity": {"enabled": True, "center_hz": 2500.0, "q": 0.9, "gain_db": 0.75},
        },
        "dynamic_eq": {
            "harsh": {
                "enabled": True,
                "low_hz": 2800.0,
                "high_hz": 4200.0,
                "threshold_dbfs": -45.0,
                "max_reduction_db": 3.0,
                "slope": 0.55,
                "attack_ms": 3.0,
                "release_ms": 90.0,
            },
            "deess": {
                "enabled": True,
                "low_hz": 5000.0,
                "high_hz": 7200.0,
                "threshold_dbfs": -49.0,
                "max_reduction_db": 3.0,
                "slope": 0.55,
                "attack_ms": 2.5,
                "release_ms": 90.0,
            },
        },
        "saturation": {
            "enabled": True,
            "drive": 1.45,
            "mix": 0.09,
            "reduce_if_peak_above": 0.92,
            "reduce_if_decrackled": True,
        },
        "compressor": {
            "enabled": True,
            "threshold_dbfs": -24.0,
            "ratio": 2.5,
            "attack_ms": 20.0,
            "release_ms": 160.0,
            "knee_db": 6.0,
            "makeup_gain_db": 1.2,
        },
        "final_eq": {
            "warmth": {"enabled": False, "center_hz": 160.0, "q": 0.8, "gain_db": 0.0},
            "mud": {"enabled": False, "center_hz": 300.0, "q": 1.0, "gain_db": 0.0},
            "sharp": {"enabled": False, "center_hz": 3500.0, "q": 1.0, "gain_db": 0.0},
            "clarity": {"enabled": False, "center_hz": 2500.0, "q": 0.9, "gain_db": 0.0},
        },
    },
    "radio_smooth": {
        "leveler": {
            "enabled": True,
            "target_rms_dbfs": -23.0,
            "max_boost_db": 7.5,
            "max_cut_db": 8.0,
            "attack_db_per_sec": 12.0,
            "release_db_per_sec": 5.0,
            "silence_release_db_per_sec": 10.0,
            "vad_threshold": 0.45,
            "min_active_dbfs": -54.0,
        },
        "eq": {
            "warmth": {"enabled": True, "center_hz": 170.0, "q": 0.75, "gain_db": 3.0},
            "mud": {"enabled": True, "center_hz": 320.0, "q": 1.0, "gain_db": -3.5},
            "nasal": {"enabled": False, "center_hz": 900.0, "q": 1.0, "gain_db": -2.0},
            "clarity": {"enabled": True, "center_hz": 2500.0, "q": 0.9, "gain_db": 1.0},
        },
        "dynamic_eq": {
            "harsh": {
                "enabled": True,
                "low_hz": 2800.0,
                "high_hz": 4200.0,
                "threshold_dbfs": -47.0,
                "max_reduction_db": 4.0,
                "slope": 0.65,
                "attack_ms": 2.0,
                "release_ms": 85.0,
            },
            "deess": {
                "enabled": True,
                "low_hz": 5000.0,
                "high_hz": 7200.0,
                "threshold_dbfs": -50.0,
                "max_reduction_db": 3.5,
                "slope": 0.6,
                "attack_ms": 2.0,
                "release_ms": 80.0,
            },
        },
        "saturation": {
            "enabled": True,
            "drive": 1.55,
            "mix": 0.13,
            "reduce_if_peak_above": 0.90,
            "reduce_if_decrackled": True,
        },
        "compressor": {
            "enabled": True,
            "threshold_dbfs": -25.0,
            "ratio": 3.0,
            "attack_ms": 18.0,
            "release_ms": 150.0,
            "knee_db": 7.0,
            "makeup_gain_db": 1.8,
        },
        "final_eq": {
            "warmth": {"enabled": False, "center_hz": 160.0, "q": 0.8, "gain_db": 0.0},
            "mud": {"enabled": False, "center_hz": 300.0, "q": 1.0, "gain_db": 0.0},
            "sharp": {"enabled": False, "center_hz": 3500.0, "q": 1.0, "gain_db": 0.0},
            "clarity": {"enabled": False, "center_hz": 2500.0, "q": 0.9, "gain_db": 0.0},
        },
    },
}


@dataclass
class _IirFilter:
    label: str
    b: np.ndarray
    a: np.ndarray
    state: np.ndarray
    gain_db: float


@dataclass
class _DynamicBand:
    label: str
    low_hz: float
    high_hz: float
    threshold_dbfs: float
    max_reduction_db: float
    slope: float
    attack_ms: float
    release_ms: float
    sos: np.ndarray
    state: np.ndarray
    reduction_db: float = 0.0


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "mode":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    mode = str(config.get("mode", "warm_smooth")).lower().replace("-", "_")
    if mode == "safe":
        mode = "safe_warm"
    elif mode in {"smooth", "default", "warm"}:
        mode = "warm_smooth"
    elif mode == "radio":
        mode = "radio_smooth"
    base = _MODE_DEFAULTS.get(mode, _MODE_DEFAULTS["warm_smooth"])
    out = _merge_dicts(base, config)
    out["mode"] = mode
    return out


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _linear_to_db(value: float) -> float:
    return 20.0 * float(np.log10(max(abs(float(value)), 1e-12)))


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -240.0
    return _linear_to_db(float(np.sqrt(np.mean(x * x) + 1e-12)))


def _clamped_freq(freq_hz: float, sample_rate: int) -> float:
    nyquist = sample_rate / 2.0
    return float(min(max(float(freq_hz), 10.0), max(20.0, nyquist - 50.0)))


def _peaking_eq(center_hz: float, q: float, gain_db: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    f0 = _clamped_freq(center_hz, sample_rate)
    q = max(0.1, float(q))
    gain_db = float(gain_db)
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / float(sample_rate)
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * a_gain
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_gain
    a0 = 1.0 + alpha / a_gain
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_gain
    b = np.asarray([b0, b1, b2], dtype=np.float64) / a0
    a = np.asarray([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def _empty_iir_state(channels: int, length: int) -> np.ndarray:
    return np.zeros((channels, max(1, int(length) - 1)), dtype=np.float64)


def _smooth_step(current: float, target: float, dt: float, attack_ms: float, release_ms: float) -> float:
    tau_ms = float(attack_ms if target > current else release_ms)
    if tau_ms <= 0.0:
        return float(target)
    alpha = float(np.exp(-dt / max(tau_ms / 1000.0, 1e-6)))
    return float(alpha * current + (1.0 - alpha) * target)


def _smooth_gain_db(current: float, target: float, dt: float, attack_ms: float, release_ms: float) -> float:
    tau_ms = float(attack_ms if target < current else release_ms)
    if tau_ms <= 0.0:
        return float(target)
    alpha = float(np.exp(-dt / max(tau_ms / 1000.0, 1e-6)))
    return float(alpha * current + (1.0 - alpha) * target)


def _slew_db(current: float, target: float, dt: float, up_rate: float, down_rate: float) -> float:
    delta = float(target) - float(current)
    rate = float(up_rate if delta > 0.0 else down_rate)
    step = max(0.0, rate) * float(dt)
    if abs(delta) <= step:
        return float(target)
    return float(current + (step if delta > 0.0 else -step))


def _soft_knee_compressor_gain_db(
    level_dbfs: float,
    threshold_dbfs: float,
    ratio: float,
    knee_db: float,
) -> float:
    ratio = max(1.0, float(ratio))
    threshold = float(threshold_dbfs)
    knee = max(0.0, float(knee_db))
    level = float(level_dbfs)
    if ratio <= 1.0:
        return 0.0
    if knee <= 0.0:
        if level <= threshold:
            return 0.0
        return (threshold + (level - threshold) / ratio) - level
    lower = threshold - knee / 2.0
    upper = threshold + knee / 2.0
    if level <= lower:
        return 0.0
    if level >= upper:
        return (threshold + (level - threshold) / ratio) - level
    x = level - lower
    return (1.0 / ratio - 1.0) * x * x / (2.0 * knee)


@register_method("tone", "warm_rounded_voice")
class WarmRoundedVoiceProcessor(BaseStreamingProcessor):
    name = "warm_rounded_voice"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.cfg = _resolved_config(self.config)
        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.pre_filters: list[_IirFilter] = []
        self.post_filters: list[_IirFilter] = []
        self.dynamic_bands: list[_DynamicBand] = []
        self.level_gain_db = 0.0
        self.compressor_gain_db = 0.0

    @property
    def algorithmic_latency_ms(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.sample_rate = None
        self.channels = None
        self.pre_filters = []
        self.post_filters = []
        self.dynamic_bands = []
        self.level_gain_db = 0.0
        self.compressor_gain_db = 0.0

    def _add_peaking_filter(
        self,
        filters: list[_IirFilter],
        label: str,
        spec: dict[str, Any],
        sample_rate: int,
        channels: int,
    ) -> None:
        if not bool(spec.get("enabled", True)):
            return
        gain_db = float(spec.get("gain_db", 0.0))
        if abs(gain_db) < 0.01:
            return
        b, a = _peaking_eq(
            float(spec.get("center_hz", 1000.0)),
            float(spec.get("q", 1.0)),
            gain_db,
            sample_rate,
        )
        filters.append(
            _IirFilter(
                label=label,
                b=b,
                a=a,
                state=_empty_iir_state(channels, max(len(a), len(b))),
                gain_db=gain_db,
            )
        )

    def _add_dynamic_band(
        self,
        label: str,
        spec: dict[str, Any],
        sample_rate: int,
        channels: int,
    ) -> None:
        if not bool(spec.get("enabled", True)):
            return
        low = _clamped_freq(float(spec.get("low_hz", 2500.0)), sample_rate)
        high = _clamped_freq(float(spec.get("high_hz", 5000.0)), sample_rate)
        if high <= low + 20.0:
            return
        sos = signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        state = np.zeros((sos.shape[0], channels, 2), dtype=np.float64)
        self.dynamic_bands.append(
            _DynamicBand(
                label=label,
                low_hz=low,
                high_hz=high,
                threshold_dbfs=float(spec.get("threshold_dbfs", -45.0)),
                max_reduction_db=float(spec.get("max_reduction_db", 3.0)),
                slope=float(spec.get("slope", 0.55)),
                attack_ms=float(spec.get("attack_ms", 3.0)),
                release_ms=float(spec.get("release_ms", 90.0)),
                sos=sos,
                state=state,
            )
        )

    def _configure(self, sample_rate: int, channels: int) -> None:
        if self.sample_rate == sample_rate and self.channels == channels:
            return
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.pre_filters = []
        self.post_filters = []
        self.dynamic_bands = []

        eq_cfg = self.cfg.get("eq", {})
        for label in ("warmth", "mud", "nasal", "clarity"):
            self._add_peaking_filter(
                self.pre_filters,
                label,
                dict(eq_cfg.get(label, {})),
                self.sample_rate,
                self.channels,
            )
        final_eq_cfg = self.cfg.get("final_eq", {})
        for label in ("warmth", "mud", "sharp", "clarity"):
            self._add_peaking_filter(
                self.post_filters,
                f"final_{label}",
                dict(final_eq_cfg.get(label, {})),
                self.sample_rate,
                self.channels,
            )
        dyn_cfg = self.cfg.get("dynamic_eq", {})
        self._add_dynamic_band("harsh", dict(dyn_cfg.get("harsh", {})), self.sample_rate, self.channels)
        self._add_dynamic_band("deess", dict(dyn_cfg.get("deess", {})), self.sample_rate, self.channels)

    def _apply_iir_filters(self, y: np.ndarray, filters: list[_IirFilter]) -> np.ndarray:
        for filt in filters:
            y, filt.state = signal.lfilter(filt.b, filt.a, y, axis=-1, zi=filt.state)
        return np.asarray(y, dtype=np.float32)

    def _apply_leveler(
        self,
        y: np.ndarray,
        chunk: AudioChunk,
        dt: float,
        level_dbfs: float,
    ) -> tuple[np.ndarray, bool]:
        cfg = self.cfg.get("leveler", {})
        if not bool(cfg.get("enabled", True)):
            return y, False
        speech_prob = chunk.metadata.get("speech_prob")
        if speech_prob is None:
            active = level_dbfs >= float(cfg.get("min_active_dbfs", -54.0))
        else:
            active = float(speech_prob) >= float(cfg.get("vad_threshold", 0.45))
            active = active and level_dbfs >= float(cfg.get("min_active_dbfs", -54.0))
        if active:
            desired = float(cfg.get("target_rms_dbfs", -24.0)) - level_dbfs
            desired = min(float(cfg.get("max_boost_db", 6.0)), desired)
            desired = max(-float(cfg.get("max_cut_db", 7.0)), desired)
            self.level_gain_db = _slew_db(
                self.level_gain_db,
                desired,
                dt,
                float(cfg.get("attack_db_per_sec", 10.0)),
                float(cfg.get("release_db_per_sec", 4.0)),
            )
        else:
            target = min(self.level_gain_db, 0.0)
            self.level_gain_db = _slew_db(
                self.level_gain_db,
                target,
                dt,
                float(cfg.get("silence_release_db_per_sec", 9.0)),
                float(cfg.get("silence_release_db_per_sec", 9.0)),
            )
        return np.asarray(y * _db_to_linear(self.level_gain_db), dtype=np.float32), active

    def _apply_dynamic_bands(self, y: np.ndarray, dt: float) -> tuple[np.ndarray, dict[str, float]]:
        metrics: dict[str, float] = {}
        for band in self.dynamic_bands:
            band_signal, band.state = signal.sosfilt(band.sos, y, axis=-1, zi=band.state)
            band_level_db = _rms_db(np.asarray(band_signal, dtype=np.float32))
            over_db = max(0.0, band_level_db - band.threshold_dbfs)
            target_reduction = min(band.max_reduction_db, over_db * band.slope)
            band.reduction_db = _smooth_step(
                band.reduction_db,
                target_reduction,
                dt,
                band.attack_ms,
                band.release_ms,
            )
            gain = _db_to_linear(-band.reduction_db)
            y = y + (gain - 1.0) * band_signal
            metrics[f"{band.label}_band_level_dbfs"] = band_level_db
            metrics[f"{band.label}_reduction_db"] = band.reduction_db
        return np.asarray(y, dtype=np.float32), metrics

    def _apply_saturation(self, y: np.ndarray, chunk: AudioChunk) -> tuple[np.ndarray, float]:
        cfg = self.cfg.get("saturation", {})
        if not bool(cfg.get("enabled", True)):
            return y, 0.0
        mix = float(cfg.get("mix", 0.08))
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak >= float(cfg.get("reduce_if_peak_above", 0.92)):
            mix *= 0.35
        repaired_clicks = int(chunk.metadata.get("decrackle_repaired_clicks", 0) or 0)
        if bool(cfg.get("reduce_if_decrackled", True)) and repaired_clicks > 0:
            mix *= 0.5
        mix = float(np.clip(mix, 0.0, 0.25))
        if mix <= 0.0:
            return y, 0.0
        drive = max(1.0, float(cfg.get("drive", 1.45)))
        wet = np.tanh(drive * y) / max(np.tanh(drive), 1e-6)
        return np.asarray(y * (1.0 - mix) + wet * mix, dtype=np.float32), mix

    def _apply_compressor(self, y: np.ndarray, dt: float) -> tuple[np.ndarray, float, float]:
        cfg = self.cfg.get("compressor", {})
        if not bool(cfg.get("enabled", True)):
            self.compressor_gain_db = 0.0
            return y, 0.0, 0.0
        level_db = _rms_db(y)
        target_gain = _soft_knee_compressor_gain_db(
            level_db,
            float(cfg.get("threshold_dbfs", -24.0)),
            float(cfg.get("ratio", 2.5)),
            float(cfg.get("knee_db", 6.0)),
        )
        self.compressor_gain_db = _smooth_gain_db(
            self.compressor_gain_db,
            target_gain,
            dt,
            float(cfg.get("attack_ms", 20.0)),
            float(cfg.get("release_ms", 160.0)),
        )
        makeup = float(cfg.get("makeup_gain_db", 1.2))
        total_gain_db = self.compressor_gain_db + makeup
        return np.asarray(y * _db_to_linear(total_gain_db), dtype=np.float32), level_db, self.compressor_gain_db

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not bool(self.config.get("enabled", True)):
            return super().process(chunk)
        x = chunk.samples.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self._configure(chunk.sample_rate, int(x.shape[0]))
        y = x.detach().cpu().numpy().astype(np.float32, copy=False)
        dt = chunk.duration_sec
        input_level_db = _rms_db(y)

        y, speech_active = self._apply_leveler(y, chunk, dt, input_level_db)
        y = self._apply_iir_filters(y, self.pre_filters)
        y, dynamic_metrics = self._apply_dynamic_bands(y, dt)
        y, saturation_mix = self._apply_saturation(y, chunk)
        y, compressor_input_level_db, compressor_reduction_db = self._apply_compressor(y, dt)
        y = self._apply_iir_filters(y, self.post_filters)

        output = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(
            device=chunk.samples.device,
            dtype=chunk.samples.dtype,
        )
        metadata = merge_metadata(
            chunk,
            warm_rounded_voice=True,
            warm_rounded_mode=self.cfg.get("mode", "warm_smooth"),
            tone_level_gain_db=self.level_gain_db,
        )
        eq_cfg = self.cfg.get("eq", {})
        return ProcessResult(
            chunk=AudioChunk(
                output.contiguous(),
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics={
                "warm_rounded_mode": str(self.cfg.get("mode", "warm_smooth")),
                "tone_speech_active": bool(speech_active),
                "tone_input_level_dbfs": input_level_db,
                "tone_level_gain_db": self.level_gain_db,
                "warmth_gain_db": float(eq_cfg.get("warmth", {}).get("gain_db", 0.0)),
                "mud_gain_db": float(eq_cfg.get("mud", {}).get("gain_db", 0.0)),
                "saturation_mix": saturation_mix,
                "compressor_input_level_dbfs": compressor_input_level_db,
                "compressor_gain_reduction_db": compressor_reduction_db,
                "realtime_safe": True,
                **dynamic_metrics,
            },
            algorithmic_latency_ms=0.0,
        )
