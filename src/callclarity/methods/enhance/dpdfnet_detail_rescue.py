from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.methods.denoise.dpdfnet import DpdfnetDenoiser
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


_EPS = 1e-12


def _rms_dbfs_np(x: np.ndarray) -> float:
    if x.size == 0:
        return -240.0
    finite = np.nan_to_num(
        x.astype(np.float64, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    value = float(np.sqrt(np.mean(np.square(finite)) + _EPS))
    return 20.0 * float(np.log10(max(value, _EPS)))


def _peak_np(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _smooth_toward(
    current: float,
    target: float,
    duration_sec: float,
    attack_ms: float,
    release_ms: float,
) -> float:
    tau_ms = attack_ms if target > current else release_ms
    if tau_ms <= 0.0 or duration_sec <= 0.0:
        return float(target)
    alpha = 1.0 - float(np.exp(-duration_sec / (tau_ms / 1000.0)))
    return float(current + _clamp01(alpha) * (target - current))


def _as_2d(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = samples.shape
    if samples.ndim == 1:
        return samples.unsqueeze(0), original_shape
    if samples.ndim == 2:
        return samples, original_shape
    return samples.reshape(-1, samples.shape[-1]), original_shape


def _restore_shape(samples: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    if len(original_shape) == 1:
        return samples.reshape(original_shape)
    return samples.reshape(original_shape)


@register_method("enhance", "dpdfnet_detail_rescue")
class DpdfnetDetailRescueEnhancer(BaseStreamingProcessor):
    name = "dpdfnet_detail_rescue"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.rescue_cfg = self.config.get("rescue", {})
        self.final_cfg = self.config.get("final", {})
        self.denoiser = DpdfnetDenoiser(self.config.get("denoise", {})) if self.enabled else None

        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._consonant_sos: np.ndarray | None = None
        self._presence_sos: np.ndarray | None = None
        self._sibilance_sos: np.ndarray | None = None
        self._dry_consonant_state: np.ndarray | None = None
        self._wet_consonant_state: np.ndarray | None = None
        self._dry_presence_state: np.ndarray | None = None
        self._wet_presence_state: np.ndarray | None = None
        self._dry_sibilance_state: np.ndarray | None = None
        self._wet_sibilance_state: np.ndarray | None = None
        self._dry_delay_buffer: np.ndarray | None = None

        self.consonant_restore_mix = 0.0
        self.presence_restore_mix = 0.0
        self.onset_restore_mix = 0.0
        self.sibilance_guard = 1.0
        self._prev_dry_consonant_dbfs: float | None = None
        self._prev_wet_consonant_dbfs: float | None = None

    @property
    def algorithmic_latency_ms(self) -> float:
        if not self.enabled or self.denoiser is None:
            return 0.0
        rescue_latency = self.rescue_cfg.get("algorithmic_latency_ms")
        if rescue_latency is not None:
            return self.denoiser.algorithmic_latency_ms + float(rescue_latency)
        delay = int(self.rescue_cfg.get("dry_delay_compensation_samples", 0))
        if delay > 0 and self._sample_rate:
            delay_ms = 1000.0 * delay / float(self._sample_rate)
            return self.denoiser.algorithmic_latency_ms + delay_ms
        return self.denoiser.algorithmic_latency_ms

    @property
    def lookahead_ms(self) -> float:
        if not self.enabled or self.denoiser is None:
            return 0.0
        return self.denoiser.lookahead_ms

    def reset(self) -> None:
        if self.denoiser is not None:
            self.denoiser.reset()
        self._sample_rate = None
        self._channels = None
        self._consonant_sos = None
        self._presence_sos = None
        self._sibilance_sos = None
        self._dry_consonant_state = None
        self._wet_consonant_state = None
        self._dry_presence_state = None
        self._wet_presence_state = None
        self._dry_sibilance_state = None
        self._wet_sibilance_state = None
        self._dry_delay_buffer = None
        self.consonant_restore_mix = 0.0
        self.presence_restore_mix = 0.0
        self.onset_restore_mix = 0.0
        self.sibilance_guard = 1.0
        self._prev_dry_consonant_dbfs = None
        self._prev_wet_consonant_dbfs = None

    def warmup(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        if self.denoiser is not None:
            self.denoiser.warmup(sample_rate)

    def _clamped_band(
        self,
        band: list[float] | tuple[float, float],
        sample_rate: int,
    ) -> tuple[float, float]:
        nyquist = float(sample_rate) / 2.0
        margin = max(1.0, nyquist * 0.01)
        high_limit = max(margin * 2.0, nyquist - margin)
        requested_low = float(band[0])
        requested_high = float(band[1])
        high = min(max(requested_high, margin * 2.0), high_limit)
        low = min(max(requested_low, margin), high - margin)
        if low >= high:
            low = max(margin, high * 0.5)
        return low, high

    def _configure_filters(self, sample_rate: int, channels: int) -> None:
        if (
            self._sample_rate == sample_rate
            and self._channels == channels
            and self._consonant_sos is not None
            and self._presence_sos is not None
            and self._sibilance_sos is not None
        ):
            return

        self._sample_rate = sample_rate
        self._channels = channels
        consonant_band = self.rescue_cfg.get("consonant_band_hz", [2200.0, 5200.0])
        presence_band = self.rescue_cfg.get("presence_band_hz", [1200.0, 3600.0])
        sibilance_band = self.rescue_cfg.get("sibilance_guard_band_hz", [5200.0, 7600.0])
        consonant_low, consonant_high = self._clamped_band(consonant_band, sample_rate)
        presence_low, presence_high = self._clamped_band(presence_band, sample_rate)
        sibilance_low, sibilance_high = self._clamped_band(sibilance_band, sample_rate)

        self._consonant_sos = signal.butter(
            4,
            [consonant_low, consonant_high],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        ).astype(np.float32)
        self._presence_sos = signal.butter(
            3,
            [presence_low, presence_high],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        ).astype(np.float32)
        self._sibilance_sos = signal.butter(
            4,
            [sibilance_low, sibilance_high],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        ).astype(np.float32)

        consonant_state_shape = (self._consonant_sos.shape[0], channels, 2)
        presence_state_shape = (self._presence_sos.shape[0], channels, 2)
        sibilance_state_shape = (self._sibilance_sos.shape[0], channels, 2)
        self._dry_consonant_state = np.zeros(consonant_state_shape, dtype=np.float32)
        self._wet_consonant_state = np.zeros(consonant_state_shape, dtype=np.float32)
        self._dry_presence_state = np.zeros(presence_state_shape, dtype=np.float32)
        self._wet_presence_state = np.zeros(presence_state_shape, dtype=np.float32)
        self._dry_sibilance_state = np.zeros(sibilance_state_shape, dtype=np.float32)
        self._wet_sibilance_state = np.zeros(sibilance_state_shape, dtype=np.float32)

        delay = int(self.rescue_cfg.get("dry_delay_compensation_samples", 0))
        self._dry_delay_buffer = np.zeros((channels, max(0, delay)), dtype=np.float32)

    def _delay_dry(self, dry: np.ndarray) -> np.ndarray:
        delay = int(self.rescue_cfg.get("dry_delay_compensation_samples", 0))
        if delay <= 0:
            return dry
        if self._dry_delay_buffer is None or self._dry_delay_buffer.shape[0] != dry.shape[0]:
            self._dry_delay_buffer = np.zeros((dry.shape[0], delay), dtype=np.float32)
        combined = np.concatenate([self._dry_delay_buffer, dry], axis=-1)
        delayed = combined[:, : dry.shape[-1]]
        self._dry_delay_buffer = combined[:, dry.shape[-1] :].astype(np.float32, copy=False)
        return delayed.astype(np.float32, copy=False)

    def _filter_bands(
        self,
        dry: np.ndarray,
        wet: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._configure_filters(sample_rate, int(dry.shape[0]))
        assert self._consonant_sos is not None
        assert self._presence_sos is not None
        assert self._sibilance_sos is not None
        assert self._dry_consonant_state is not None
        assert self._wet_consonant_state is not None
        assert self._dry_presence_state is not None
        assert self._wet_presence_state is not None
        assert self._dry_sibilance_state is not None
        assert self._wet_sibilance_state is not None

        dry_consonant, self._dry_consonant_state = signal.sosfilt(
            self._consonant_sos,
            dry,
            axis=-1,
            zi=self._dry_consonant_state,
        )
        wet_consonant, self._wet_consonant_state = signal.sosfilt(
            self._consonant_sos,
            wet,
            axis=-1,
            zi=self._wet_consonant_state,
        )
        dry_presence, self._dry_presence_state = signal.sosfilt(
            self._presence_sos,
            dry,
            axis=-1,
            zi=self._dry_presence_state,
        )
        wet_presence, self._wet_presence_state = signal.sosfilt(
            self._presence_sos,
            wet,
            axis=-1,
            zi=self._wet_presence_state,
        )
        dry_sibilance, self._dry_sibilance_state = signal.sosfilt(
            self._sibilance_sos,
            dry,
            axis=-1,
            zi=self._dry_sibilance_state,
        )
        wet_sibilance, self._wet_sibilance_state = signal.sosfilt(
            self._sibilance_sos,
            wet,
            axis=-1,
            zi=self._wet_sibilance_state,
        )
        return (
            dry_consonant.astype(np.float32, copy=False),
            wet_consonant.astype(np.float32, copy=False),
            dry_presence.astype(np.float32, copy=False),
            wet_presence.astype(np.float32, copy=False),
            dry_sibilance.astype(np.float32, copy=False),
            wet_sibilance.astype(np.float32, copy=False),
        )

    def _align_wet_channels(self, wet: torch.Tensor, dry_channels: int) -> torch.Tensor:
        wet_2d, _ = _as_2d(wet)
        if wet_2d.shape[0] == dry_channels:
            return wet_2d
        if wet_2d.shape[0] == 1:
            return wet_2d.repeat(dry_channels, 1)
        if dry_channels == 1:
            return wet_2d.mean(dim=0, keepdim=True)
        return wet_2d.mean(dim=0, keepdim=True).repeat(dry_channels, 1)

    def _speech_probability(
        self,
        chunk: AudioChunk,
        denoise_result: ProcessResult,
        input_rms_dbfs: float,
    ) -> float:
        wet_metadata = denoise_result.chunk.metadata
        if "speech_prob" in wet_metadata:
            return float(wet_metadata["speech_prob"])
        if "speech_prob" in chunk.metadata:
            return float(chunk.metadata["speech_prob"])
        min_input = float(self.rescue_cfg.get("min_input_rms_dbfs", -58.0))
        return 1.0 if input_rms_dbfs > min_input else 0.0

    def _loss_factor(self, loss_db: float) -> float:
        threshold = float(self.rescue_cfg.get("restore_when_wet_loses_db", 5.0))
        if loss_db <= threshold:
            return 0.0
        return _clamp01((loss_db - threshold) / max(6.0, threshold * 2.0))

    def _high_band_guard(self, dry_consonant_dbfs: float) -> float:
        threshold = float(self.rescue_cfg.get("noise_guard_high_band_dbfs", -38.0))
        if dry_consonant_dbfs >= threshold:
            return 1.0
        return _clamp01((dry_consonant_dbfs - (threshold - 12.0)) / 12.0)

    def _sibilance_restore_guard(
        self,
        dry_sibilance_dbfs: float,
        sibilance_loss_db: float,
    ) -> float:
        floor = float(self.rescue_cfg.get("sibilance_guard_min_dbfs", -92.0))
        threshold = float(self.rescue_cfg.get("sibilance_guard_loss_db", 1.5))
        max_reduction = _clamp01(
            float(self.rescue_cfg.get("sibilance_guard_max_reduction", 0.75))
        )
        if dry_sibilance_dbfs < floor or sibilance_loss_db <= threshold:
            return 1.0
        strength = _clamp01((sibilance_loss_db - threshold) / 6.0)
        return float(1.0 - max_reduction * strength)

    def _target_mixes(
        self,
        speech_prob: float,
        input_rms_dbfs: float,
        dry_consonant_dbfs: float,
        consonant_loss_db: float,
        presence_loss_db: float,
        dry_sibilance_dbfs: float,
        sibilance_loss_db: float,
    ) -> tuple[float, float, float, float]:
        speech_threshold = float(self.rescue_cfg.get("speech_prob_threshold", 0.35))
        min_input = float(self.rescue_cfg.get("min_input_rms_dbfs", -58.0))
        if speech_prob < speech_threshold or input_rms_dbfs < min_input:
            return 0.0, 0.0, 0.0, 1.0

        high_guard = self._high_band_guard(dry_consonant_dbfs)
        sibilance_guard = self._sibilance_restore_guard(
            dry_sibilance_dbfs,
            sibilance_loss_db,
        )
        consonant_guard = high_guard * sibilance_guard
        consonant_target = (
            float(self.rescue_cfg.get("max_consonant_restore_mix", 0.22))
            * self._loss_factor(consonant_loss_db)
            * consonant_guard
        )
        presence_target = (
            float(self.rescue_cfg.get("max_presence_restore_mix", 0.10))
            * self._loss_factor(presence_loss_db)
        )

        onset_target = 0.0
        if self._prev_dry_consonant_dbfs is not None and self._prev_wet_consonant_dbfs is not None:
            dry_jump = dry_consonant_dbfs - self._prev_dry_consonant_dbfs
            wet_jump = (dry_consonant_dbfs - consonant_loss_db) - self._prev_wet_consonant_dbfs
            onset_advantage = dry_jump - wet_jump
            if dry_jump > 3.0 and onset_advantage > 3.0:
                onset_target = (
                    float(self.rescue_cfg.get("onset_boost_mix", 0.04))
                    * _clamp01((onset_advantage - 3.0) / 9.0)
                    * consonant_guard
                )

        return consonant_target, presence_target, onset_target, sibilance_guard

    def _update_smoothed_mixes(
        self,
        consonant_target: float,
        presence_target: float,
        onset_target: float,
        duration_sec: float,
    ) -> None:
        attack_ms = float(self.rescue_cfg.get("attack_ms", 6.0))
        release_ms = float(self.rescue_cfg.get("release_ms", 80.0))
        self.consonant_restore_mix = _smooth_toward(
            self.consonant_restore_mix,
            consonant_target,
            duration_sec,
            attack_ms,
            release_ms,
        )
        self.presence_restore_mix = _smooth_toward(
            self.presence_restore_mix,
            presence_target,
            duration_sec,
            attack_ms,
            release_ms,
        )
        self.onset_restore_mix = _smooth_toward(
            self.onset_restore_mix,
            onset_target,
            duration_sec,
            attack_ms,
            release_ms,
        )

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
        dry = dry_float.cpu().numpy().astype(np.float32, copy=False)
        dry = np.nan_to_num(dry, nan=0.0, posinf=0.0, neginf=0.0)
        input_rms_dbfs = _rms_dbfs_np(dry)

        denoise_result = self.denoiser.process(chunk)
        wet_aligned = self._align_wet_channels(
            denoise_result.chunk.samples.detach().float(),
            int(dry.shape[0]),
        )
        if wet_aligned.shape[-1] != dry.shape[-1]:
            if wet_aligned.shape[-1] > dry.shape[-1]:
                wet_aligned = wet_aligned[..., : dry.shape[-1]]
            else:
                pad = dry.shape[-1] - wet_aligned.shape[-1]
                wet_aligned = torch.nn.functional.pad(wet_aligned, (0, pad))
        wet = wet_aligned.cpu().numpy().astype(np.float32, copy=False)
        wet = np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0)
        speech_prob = self._speech_probability(chunk, denoise_result, input_rms_dbfs)

        if not bool(self.rescue_cfg.get("enabled", True)):
            out_np = wet
            metrics = {
                "dpdfnet_detail_rescue_enabled": False,
                "speech_prob": speech_prob,
                "input_rms_dbfs": input_rms_dbfs,
                "consonant_restore_mix": 0.0,
                "presence_restore_mix": 0.0,
                "onset_restore_mix": 0.0,
                "output_peak": _peak_np(out_np),
                **self._dpdfnet_metrics(denoise_result),
            }
            out = torch.from_numpy(out_np).to(
                device=chunk.samples.device,
                dtype=chunk.samples.dtype,
            )
            out = _restore_shape(out, original_shape).contiguous()
            metadata = merge_metadata(
                denoise_result.chunk,
                dpdfnet_detail_rescue=True,
                denoised=True,
                neural_denoiser="DPDFNet+detail_rescue",
                consonant_restore_mix=0.0,
                presence_restore_mix=0.0,
                onset_restore_mix=0.0,
            )
            return ProcessResult(
                chunk=AudioChunk(
                    out,
                    chunk.sample_rate,
                    chunk.start_time_sec,
                    chunk.stream_id,
                    metadata,
                ),
                metrics=metrics,
                events=denoise_result.events,
                algorithmic_latency_ms=self.algorithmic_latency_ms,
            )

        dry_for_rescue = self._delay_dry(dry)
        (
            dry_consonant,
            wet_consonant,
            dry_presence,
            wet_presence,
            dry_sibilance,
            wet_sibilance,
        ) = self._filter_bands(dry_for_rescue, wet, chunk.sample_rate)

        dry_consonant_dbfs = _rms_dbfs_np(dry_consonant)
        wet_consonant_dbfs = _rms_dbfs_np(wet_consonant)
        dry_presence_dbfs = _rms_dbfs_np(dry_presence)
        wet_presence_dbfs = _rms_dbfs_np(wet_presence)
        dry_sibilance_dbfs = _rms_dbfs_np(dry_sibilance)
        wet_sibilance_dbfs = _rms_dbfs_np(wet_sibilance)
        consonant_loss_db = dry_consonant_dbfs - wet_consonant_dbfs
        presence_loss_db = dry_presence_dbfs - wet_presence_dbfs
        sibilance_loss_db = dry_sibilance_dbfs - wet_sibilance_dbfs

        consonant_target, presence_target, onset_target, self.sibilance_guard = self._target_mixes(
            speech_prob,
            input_rms_dbfs,
            dry_consonant_dbfs,
            consonant_loss_db,
            presence_loss_db,
            dry_sibilance_dbfs,
            sibilance_loss_db,
        )
        self._update_smoothed_mixes(
            consonant_target,
            presence_target,
            onset_target,
            chunk.duration_sec,
        )

        detail_consonant = dry_consonant - wet_consonant
        detail_presence = dry_presence - wet_presence
        out_np = (
            wet
            + self.consonant_restore_mix * detail_consonant
            + self.presence_restore_mix * detail_presence
            + self.onset_restore_mix * detail_consonant
        )
        out_np = np.nan_to_num(out_np, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32,
            copy=False,
        )
        ceiling = abs(float(self.final_cfg.get("ceiling", 0.98)))
        if ceiling > 0.0:
            out_np = np.clip(out_np, -ceiling, ceiling)

        output_peak = _peak_np(out_np)
        out = torch.from_numpy(out_np).to(device=chunk.samples.device, dtype=chunk.samples.dtype)
        out = _restore_shape(out, original_shape).contiguous()

        metadata = merge_metadata(
            denoise_result.chunk,
            dpdfnet_detail_rescue=True,
            denoised=True,
            neural_denoiser="DPDFNet+detail_rescue",
            consonant_restore_mix=self.consonant_restore_mix,
            presence_restore_mix=self.presence_restore_mix,
            onset_restore_mix=self.onset_restore_mix,
        )
        metrics = {
            "dpdfnet_detail_rescue_enabled": True,
            "speech_prob": speech_prob,
            "input_rms_dbfs": input_rms_dbfs,
            "dry_consonant_dbfs": dry_consonant_dbfs,
            "wet_consonant_dbfs": wet_consonant_dbfs,
            "consonant_loss_db": consonant_loss_db,
            "consonant_restore_mix": self.consonant_restore_mix,
            "dry_presence_dbfs": dry_presence_dbfs,
            "wet_presence_dbfs": wet_presence_dbfs,
            "presence_loss_db": presence_loss_db,
            "presence_restore_mix": self.presence_restore_mix,
            "dry_sibilance_dbfs": dry_sibilance_dbfs,
            "wet_sibilance_dbfs": wet_sibilance_dbfs,
            "sibilance_loss_db": sibilance_loss_db,
            "sibilance_guard": self.sibilance_guard,
            "onset_restore_mix": self.onset_restore_mix,
            "output_peak": output_peak,
            **self._dpdfnet_metrics(denoise_result),
        }
        self._prev_dry_consonant_dbfs = dry_consonant_dbfs
        self._prev_wet_consonant_dbfs = wet_consonant_dbfs

        return ProcessResult(
            chunk=AudioChunk(
                out,
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics=metrics,
            events=denoise_result.events,
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
