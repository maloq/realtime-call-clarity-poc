from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.methods.denoise.dpdfnet import DpdfnetDenoiser
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


def _lowpass_sos(cutoff_hz: float, sample_rate: int, order: int = 2) -> np.ndarray:
    return signal.butter(
        int(order),
        _clamped_freq(cutoff_hz, sample_rate),
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float32)


def _highpass_sos(cutoff_hz: float, sample_rate: int, order: int = 2) -> np.ndarray:
    return signal.butter(
        int(order),
        _clamped_freq(cutoff_hz, sample_rate),
        btype="highpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float32)


def _empty_state(channels: int, sections: int) -> np.ndarray:
    return np.zeros((int(sections), int(channels), 2), dtype=np.float32)


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
    current_db: float = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.state = signal.sosfilt(self.sos, x, axis=-1, zi=self.state)
        return y.astype(np.float32, copy=False)


@register_method("enhance", "dpdfnet_remaster")
class DpdfnetRemasterEnhancer(BaseStreamingProcessor):
    name = "dpdfnet_remaster"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.denoiser = DpdfnetDenoiser(self.config.get("denoise", {})) if self.enabled else None
        self.analysis_cfg = self.config.get("analysis", {})
        self.level_cfg = self.config.get("leveler", {})
        self.tone_cfg = self.config.get("tone", {})
        self.exciter_cfg = self.config.get("exciter", {})
        self.density_cfg = self.config.get("density", {})
        self.final_cfg = self.config.get("final", {})

        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.highpass: _IirFilter | None = None
        self.body: _BandState | None = None
        self.lower_presence: _BandState | None = None
        self.phone: _BandState | None = None
        self.deess: _BandState | None = None
        self.exciter_source: _BandState | None = None
        self.exciter_shape: _BandState | None = None
        self.exciter_env: _IirFilter | None = None
        self.noise_shape: _BandState | None = None
        self.density_band: _BandState | None = None
        self.density_smoother: _IirFilter | None = None
        self.noise_state: np.ndarray | None = None

        self.level_gain_db = 0.0
        self.compressor_cut_db = 0.0
        self.body_boost_db = 0.0
        self.lower_presence_boost_db = 0.0
        self.phone_cut_db = 0.0
        self.deess_cut_db = 0.0
        self.exciter_gain_db = 0.0
        self.exciter_mix = 0.0
        self.density_mix = 0.0

    @property
    def algorithmic_latency_ms(self) -> float:
        if not self.enabled or self.denoiser is None:
            return 0.0
        return self.denoiser.algorithmic_latency_ms + float(
            self.config.get("algorithmic_latency_ms", 0.0)
        )

    @property
    def lookahead_ms(self) -> float:
        if not self.enabled or self.denoiser is None:
            return 0.0
        return self.denoiser.lookahead_ms

    def reset(self) -> None:
        if self.denoiser is not None:
            self.denoiser.reset()
        self.sample_rate = None
        self.channels = None
        self.highpass = None
        self.body = None
        self.lower_presence = None
        self.phone = None
        self.deess = None
        self.exciter_source = None
        self.exciter_shape = None
        self.exciter_env = None
        self.noise_shape = None
        self.density_band = None
        self.density_smoother = None
        self.noise_state = None
        self.level_gain_db = 0.0
        self.compressor_cut_db = 0.0
        self.body_boost_db = 0.0
        self.lower_presence_boost_db = 0.0
        self.phone_cut_db = 0.0
        self.deess_cut_db = 0.0
        self.exciter_gain_db = 0.0
        self.exciter_mix = 0.0
        self.density_mix = 0.0

    def warmup(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        if self.denoiser is not None:
            self.denoiser.warmup(sample_rate)

    def _iir(self, label: str, sos: np.ndarray, channels: int) -> _IirFilter:
        return _IirFilter(label=label, sos=sos, state=_empty_state(channels, sos.shape[0]))

    def _band(self, label: str, sos: np.ndarray, channels: int) -> _BandState:
        return _BandState(label=label, sos=sos, state=_empty_state(channels, sos.shape[0]))

    def _configure(self, sample_rate: int, channels: int) -> None:
        if self.sample_rate == sample_rate and self.channels == channels and self.body is not None:
            return

        self.sample_rate = sample_rate
        self.channels = channels
        self.highpass = self._iir(
            "highpass",
            _highpass_sos(float(self.tone_cfg.get("highpass_hz", 55.0)), sample_rate, 2),
            channels,
        )
        self.body = self._band(
            "body",
            _bandpass_sos(self.tone_cfg.get("body_band_hz", [120.0, 720.0]), sample_rate),
            channels,
        )
        self.lower_presence = self._band(
            "lower_presence",
            _bandpass_sos(
                self.tone_cfg.get("lower_presence_band_hz", [850.0, 2400.0]),
                sample_rate,
            ),
            channels,
        )
        self.phone = self._band(
            "phone",
            _bandpass_sos(self.tone_cfg.get("phone_band_hz", [2000.0, 4800.0]), sample_rate),
            channels,
        )
        self.deess = self._band(
            "deess",
            _bandpass_sos(self.tone_cfg.get("deess_band_hz", [5200.0, 7600.0]), sample_rate),
            channels,
        )
        self.exciter_source = self._band(
            "exciter_source",
            _bandpass_sos(self.exciter_cfg.get("source_band_hz", [1000.0, 3400.0]), sample_rate),
            channels,
        )
        self.exciter_shape = self._band(
            "exciter_shape",
            _bandpass_sos(self.exciter_cfg.get("target_band_hz", [4200.0, 7600.0]), sample_rate),
            channels,
        )
        self.noise_shape = self._band(
            "noise_shape",
            _bandpass_sos(self.exciter_cfg.get("target_band_hz", [4200.0, 7600.0]), sample_rate),
            channels,
        )
        self.exciter_env = self._iir(
            "exciter_env",
            _lowpass_sos(float(self.exciter_cfg.get("envelope_lowpass_hz", 65.0)), sample_rate),
            channels,
        )
        self.density_band = self._band(
            "density_band",
            _bandpass_sos(self.density_cfg.get("band_hz", [120.0, 2600.0]), sample_rate),
            channels,
        )
        self.density_smoother = self._iir(
            "density_smoother",
            _lowpass_sos(float(self.density_cfg.get("detail_lowpass_hz", 2600.0)), sample_rate),
            channels,
        )
        self.noise_state = np.arange(1, channels + 1, dtype=np.uint32) * np.uint32(747796405)

    def _align_wet_channels(self, wet: torch.Tensor, dry_channels: int) -> torch.Tensor:
        wet_2d, _ = _as_2d(wet)
        if wet_2d.shape[0] == dry_channels:
            return wet_2d
        if wet_2d.shape[0] == 1:
            return wet_2d.repeat(dry_channels, 1)
        if dry_channels == 1:
            return wet_2d.mean(dim=0, keepdim=True)
        return wet_2d.mean(dim=0, keepdim=True).repeat(dry_channels, 1)

    def _speech_active(self, chunk: AudioChunk, denoise_result: ProcessResult, rms_dbfs: float) -> bool:
        threshold = float(self.analysis_cfg.get("speech_prob_threshold", 0.35))
        if "speech_prob" in denoise_result.chunk.metadata:
            return float(denoise_result.chunk.metadata["speech_prob"]) >= threshold
        if "speech_prob" in chunk.metadata:
            return float(chunk.metadata["speech_prob"]) >= threshold
        return rms_dbfs >= float(self.analysis_cfg.get("min_active_rms_dbfs", -58.0))

    def _process_denoiser(self, chunk: AudioChunk, dry_channels: int, target_samples: int) -> ProcessResult:
        assert self.denoiser is not None
        denoise_result = self.denoiser.process(chunk)
        wet = self._align_wet_channels(denoise_result.chunk.samples.detach().float(), dry_channels)
        if wet.shape[-1] > target_samples:
            wet = wet[..., :target_samples]
        elif wet.shape[-1] < target_samples:
            wet = torch.nn.functional.pad(wet, (0, target_samples - wet.shape[-1]))
        denoise_result.chunk = AudioChunk(
            wet.contiguous(),
            denoise_result.chunk.sample_rate,
            denoise_result.chunk.start_time_sec,
            denoise_result.chunk.stream_id,
            denoise_result.chunk.metadata,
        )
        return denoise_result

    def _apply_leveler(self, y: np.ndarray, speech_active: bool, dt: float) -> np.ndarray:
        if not speech_active or not bool(self.level_cfg.get("enabled", True)):
            self.level_gain_db = _smooth_db(
                self.level_gain_db,
                0.0,
                dt,
                float(self.level_cfg.get("attack_ms", 35.0)),
                float(self.level_cfg.get("release_ms", 260.0)),
            )
            return (y * _db_to_linear(self.level_gain_db)).astype(np.float32, copy=False)

        target_rms = float(self.level_cfg.get("target_rms_dbfs", -22.4))
        desired = target_rms - _rms_db(y)
        desired = _clamp(
            desired,
            -float(self.level_cfg.get("max_cut_db", 8.0)),
            float(self.level_cfg.get("max_boost_db", 14.0)),
        )
        self.level_gain_db = _smooth_db(
            self.level_gain_db,
            desired,
            dt,
            float(self.level_cfg.get("attack_ms", 35.0)),
            float(self.level_cfg.get("release_ms", 260.0)),
        )
        y = y * _db_to_linear(self.level_gain_db)

        comp_cfg = self.level_cfg.get("compressor", {})
        target_cut = 0.0
        if bool(comp_cfg.get("enabled", True)):
            over = _rms_db(y) - float(comp_cfg.get("threshold_dbfs", -22.0))
            if over > 0.0:
                ratio = max(float(comp_cfg.get("ratio", 2.2)), 1.0)
                target_cut = over * (1.0 - 1.0 / ratio)
        self.compressor_cut_db = _smooth_db(
            self.compressor_cut_db,
            target_cut,
            dt,
            float(comp_cfg.get("attack_ms", 10.0)),
            float(comp_cfg.get("release_ms", 90.0)),
        )
        return (y * _db_to_linear(-self.compressor_cut_db)).astype(np.float32, copy=False)

    def _parallel_gain(self, y: np.ndarray, band: np.ndarray, gain_db: float) -> np.ndarray:
        return (y + (_db_to_linear(gain_db) - 1.0) * band).astype(np.float32, copy=False)

    def _parallel_cut(self, y: np.ndarray, band: np.ndarray, cut_db: float) -> np.ndarray:
        return (y + (_db_to_linear(-cut_db) - 1.0) * band).astype(np.float32, copy=False)

    def _apply_tone(self, y: np.ndarray, speech_active: bool, dt: float) -> tuple[np.ndarray, dict[str, float]]:
        assert self.body is not None
        assert self.lower_presence is not None
        assert self.phone is not None
        assert self.deess is not None
        body_band = self.body.process(y)
        lower_presence_band = self.lower_presence.process(y)
        phone_band = self.phone.process(y)

        body_db = _rms_db(body_band)
        lower_presence_db = _rms_db(lower_presence_band)
        phone_db = _rms_db(phone_band)
        low_ref_db = _rms_db(y)

        body_target = 0.0
        lower_presence_target = 0.0
        phone_target = 0.0
        if speech_active:
            body_target = float(self.tone_cfg.get("body_boost_db", 5.2))
            if body_db > float(self.tone_cfg.get("mud_guard_dbfs", -18.0)):
                body_target *= 0.35
            lower_presence_target = float(self.tone_cfg.get("lower_presence_boost_db", -0.8))
            phone_target = float(self.tone_cfg.get("phone_cut_db", 7.0))
            if phone_db - body_db > float(self.tone_cfg.get("phone_dominance_db", 11.0)):
                phone_target = min(
                    float(self.tone_cfg.get("max_phone_cut_db", 10.0)),
                    phone_target + 0.45 * ((phone_db - body_db) - 11.0),
                )

        self.body_boost_db = _smooth_db(
            self.body_boost_db,
            body_target,
            dt,
            float(self.tone_cfg.get("attack_ms", 25.0)),
            float(self.tone_cfg.get("release_ms", 220.0)),
        )
        self.lower_presence_boost_db = _smooth_db(
            self.lower_presence_boost_db,
            lower_presence_target,
            dt,
            float(self.tone_cfg.get("attack_ms", 25.0)),
            float(self.tone_cfg.get("release_ms", 220.0)),
        )
        self.phone_cut_db = _smooth_db(
            self.phone_cut_db,
            phone_target,
            dt,
            float(self.tone_cfg.get("phone_attack_ms", 8.0)),
            float(self.tone_cfg.get("phone_release_ms", 160.0)),
        )
        y = self._parallel_gain(y, body_band, self.body_boost_db)
        y = self._parallel_gain(y, lower_presence_band, self.lower_presence_boost_db)
        y = self._parallel_cut(y, phone_band, self.phone_cut_db)
        return y, {
            "body_band_dbfs": body_db,
            "lower_presence_band_dbfs": lower_presence_db,
            "phone_band_dbfs": phone_db,
            "level_reference_dbfs": low_ref_db,
        }

    def _noise(self, channels: int, samples: int) -> np.ndarray:
        if self.noise_state is None or self.noise_state.shape[0] != channels:
            self.noise_state = np.arange(1, channels + 1, dtype=np.uint32) * np.uint32(747796405)
        out = np.empty((channels, samples), dtype=np.float32)
        for idx in range(samples):
            self.noise_state = self.noise_state * np.uint32(1664525) + np.uint32(1013904223)
            out[:, idx] = (
                ((self.noise_state >> np.uint32(8)).astype(np.float32) / float(1 << 24)) * 2.0
                - 1.0
            )
        return out

    def _apply_exciter(self, y: np.ndarray, speech_active: bool, dt: float) -> tuple[np.ndarray, float]:
        cfg = self.exciter_cfg
        assert self.exciter_source is not None
        assert self.exciter_shape is not None
        assert self.exciter_env is not None
        assert self.noise_shape is not None

        high_before = self.exciter_shape.process(y)
        high_before_db = _rms_db(high_before)
        if not (bool(cfg.get("enabled", True)) and speech_active):
            self.exciter_mix = _smooth_db(
                self.exciter_mix,
                0.0,
                dt,
                float(cfg.get("attack_ms", 8.0)),
                float(cfg.get("release_ms", 120.0)),
            )
            return y, high_before_db

        source = self.exciter_source.process(y)
        source_db = _rms_db(source)
        env = self.exciter_env.process(np.abs(source).astype(np.float32, copy=False))
        env_rms = max(float(np.sqrt(np.mean(env * env) + _EPS)), _EPS)
        env = np.clip(env / (env_rms * 2.5), 0.0, 1.0).astype(np.float32, copy=False)

        harmonic = np.tanh(float(cfg.get("drive", 2.4)) * source)
        harmonic = (harmonic - 0.45 * source).astype(np.float32, copy=False)
        harmonic = self.exciter_shape.process(harmonic)

        noise = self._noise(y.shape[0], y.shape[-1])
        noise = self.noise_shape.process(noise) * env
        excitation = (
            float(cfg.get("harmonic_mix", 0.72)) * harmonic
            + float(cfg.get("noise_mix", 0.28)) * noise
        ).astype(np.float32, copy=False)
        excitation_db = _rms_db(excitation)

        target_high = source_db + float(cfg.get("target_high_relative_to_source_db", -16.0))
        target_high = min(target_high, float(cfg.get("max_high_band_dbfs", -46.0)))
        gap = target_high - high_before_db
        mix_target = 0.0
        gain_target = 0.0
        if gap > float(cfg.get("min_gap_db", 2.0)) and excitation_db > -120.0:
            mix_target = _clamp(gap / max(float(cfg.get("full_mix_gap_db", 12.0)), 1.0), 0.0, 1.0)
            gain_target = _clamp(
                target_high - excitation_db,
                -12.0,
                float(cfg.get("max_exciter_gain_db", 34.0)),
            )

        self.exciter_mix = _smooth_db(
            self.exciter_mix,
            mix_target,
            dt,
            float(cfg.get("attack_ms", 8.0)),
            float(cfg.get("release_ms", 120.0)),
        )
        self.exciter_gain_db = _smooth_db(
            self.exciter_gain_db,
            gain_target,
            dt,
            float(cfg.get("attack_ms", 8.0)),
            float(cfg.get("release_ms", 120.0)),
        )
        return (
            y + self.exciter_mix * excitation * _db_to_linear(self.exciter_gain_db)
        ).astype(np.float32, copy=False), high_before_db

    def _apply_deesser(self, y: np.ndarray, speech_active: bool, dt: float) -> tuple[np.ndarray, float]:
        cfg = self.tone_cfg
        assert self.deess is not None
        deess_band = self.deess.process(y)
        deess_db = _rms_db(deess_band)
        target = 0.0
        if speech_active and deess_db > float(cfg.get("deess_threshold_dbfs", -42.0)):
            target = min(
                float(cfg.get("max_deess_cut_db", 5.0)),
                0.65 * (deess_db - float(cfg.get("deess_threshold_dbfs", -42.0))),
            )
        self.deess_cut_db = _smooth_db(
            self.deess_cut_db,
            target,
            dt,
            float(cfg.get("deess_attack_ms", 3.0)),
            float(cfg.get("deess_release_ms", 90.0)),
        )
        return self._parallel_cut(y, deess_band, self.deess_cut_db), deess_db

    def _apply_density(self, y: np.ndarray, speech_active: bool) -> np.ndarray:
        cfg = self.density_cfg
        self.density_mix = 0.0
        if not (bool(cfg.get("enabled", True)) and speech_active):
            return y
        if _peak_np(y) >= float(cfg.get("disable_above_peak", 0.92)):
            return y
        assert self.density_band is not None
        assert self.density_smoother is not None
        low_mid = self.density_band.process(y)
        drive = max(1e-3, float(cfg.get("drive", 1.25)))
        mix = _clamp(float(cfg.get("mix", 0.055)), 0.0, 1.0)
        if mix <= 0.0:
            return y
        saturated = np.tanh(drive * low_mid) / max(float(np.tanh(drive)), _EPS)
        detail = self.density_smoother.process((saturated - low_mid).astype(np.float32, copy=False))
        self.density_mix = mix
        return (y + mix * detail).astype(np.float32, copy=False)

    def _dpdfnet_metrics(self, denoise_result: ProcessResult) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for key, value in denoise_result.metrics.items():
            prefixed = key if str(key).startswith("dpdfnet_") else f"dpdfnet_{key}"
            metrics[prefixed] = value
        return metrics

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not self.enabled:
            return super().process(chunk)
        assert self.denoiser is not None

        dry_float, original_shape = _as_2d(chunk.samples.detach().float())
        denoise_result = self._process_denoiser(chunk, int(dry_float.shape[0]), chunk.num_samples)
        wet = denoise_result.chunk.samples.cpu().numpy().astype(np.float32, copy=False)
        wet = np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0)
        input_rms_dbfs = _rms_db(wet)
        speech_active = self._speech_active(chunk, denoise_result, input_rms_dbfs)

        self._configure(chunk.sample_rate, int(wet.shape[0]))
        y = wet
        if self.highpass is not None:
            y = self.highpass.process(y)
        y = self._apply_leveler(y, speech_active, chunk.duration_sec)
        y, tone_metrics = self._apply_tone(y, speech_active, chunk.duration_sec)
        y, high_before_dbfs = self._apply_exciter(y, speech_active, chunk.duration_sec)
        y, deess_band_dbfs = self._apply_deesser(y, speech_active, chunk.duration_sec)
        y = self._apply_density(y, speech_active)

        dry_wet = _clamp(float(self.final_cfg.get("dry_wet", 1.0)), 0.0, 1.0)
        if dry_wet < 1.0:
            y = wet * (1.0 - dry_wet) + y * dry_wet
        ceiling = abs(float(self.final_cfg.get("ceiling", 0.98)))
        if ceiling > 0.0:
            y = np.clip(y, -ceiling, ceiling)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        output_rms_dbfs = _rms_db(y)
        output_peak = _peak_np(y)
        out = torch.from_numpy(y).to(device=chunk.samples.device, dtype=chunk.samples.dtype)
        out = _restore_shape(out, original_shape).contiguous()
        metadata = merge_metadata(
            denoise_result.chunk,
            denoised=True,
            dpdfnet_remaster=True,
            neural_denoiser="DPDFNet+remaster",
            remaster_speech_active=speech_active,
            remaster_level_gain_db=self.level_gain_db,
            remaster_exciter_mix=self.exciter_mix,
        )
        metrics = {
            "dpdfnet_remaster_enabled": True,
            "remaster_speech_active": speech_active,
            "remaster_input_rms_dbfs": input_rms_dbfs,
            "remaster_output_rms_dbfs": output_rms_dbfs,
            "remaster_level_gain_db": self.level_gain_db,
            "remaster_compressor_cut_db": self.compressor_cut_db,
            "remaster_body_boost_db": self.body_boost_db,
            "remaster_lower_presence_boost_db": self.lower_presence_boost_db,
            "remaster_phone_cut_db": self.phone_cut_db,
            "remaster_deess_cut_db": self.deess_cut_db,
            "remaster_exciter_gain_db": self.exciter_gain_db,
            "remaster_exciter_mix": self.exciter_mix,
            "remaster_density_mix": self.density_mix,
            "remaster_high_before_dbfs": high_before_dbfs,
            "remaster_deess_band_dbfs": deess_band_dbfs,
            "remaster_output_peak": output_peak,
            "realtime_safe": True,
            **tone_metrics,
            **self._dpdfnet_metrics(denoise_result),
        }
        return ProcessResult(
            chunk=AudioChunk(out, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics=metrics,
            events=denoise_result.events,
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
