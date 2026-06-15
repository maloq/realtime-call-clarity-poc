from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from callclarity.dsp.envelope import linear_to_db, rms_dbfs
from callclarity.types import AudioChunk


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def _mono_np(samples: torch.Tensor) -> np.ndarray:
    x = samples.detach().float().cpu()
    if x.ndim == 2:
        x = x.mean(dim=0)
    return x.numpy().astype(np.float32, copy=False)


def _zero_run_stats(x: np.ndarray, threshold: float = 1e-7) -> tuple[int, int]:
    if x.size == 0:
        return 0, 0
    silent = np.abs(x) <= threshold
    if not bool(silent.any()):
        return 0, 0
    padded = np.concatenate(([False], silent, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    lengths = ends - starts
    return int(lengths.size), int(lengths.max(initial=0))


def _spectral_metrics(samples: torch.Tensor, sample_rate: int) -> dict[str, float]:
    x = _mono_np(samples)
    if x.size < 4:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "high_frequency_energy_ratio": 0.0,
            "narrowband_likelihood": 0.0,
        }
    window = np.hanning(x.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    total = float(np.sum(spectrum) + 1e-20)
    centroid = float(np.sum(freqs * spectrum) / total)
    cdf = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cdf, 0.95 * total, side="left"))
    rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
    hf_start = min(4000.0, max(0.0, sample_rate / 2.0 - 1.0))
    hf_energy = float(np.sum(spectrum[freqs >= hf_start]))
    hf_ratio = hf_energy / total
    if sample_rate <= 8000:
        sr_score = 1.0
    elif sample_rate <= 16000:
        sr_score = 0.35
    else:
        sr_score = 0.0
    hf_score = float(np.clip((0.08 - hf_ratio) / 0.08, 0.0, 1.0))
    rolloff_score = float(np.clip((4200.0 - rolloff) / 1800.0, 0.0, 1.0))
    narrowband = max(sr_score, 0.6 * hf_score + 0.4 * rolloff_score)
    return {
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": rolloff,
        "high_frequency_energy_ratio": hf_ratio,
        "narrowband_likelihood": float(np.clip(narrowband, 0.0, 1.0)),
    }


def _count_clicks(x: np.ndarray, sample_rate: int, prev_sample: float | None) -> int:
    if x.size == 0:
        return 0
    diffs = np.diff(x, prepend=prev_sample if prev_sample is not None else x[0])
    frame_rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    # Local derivative spikes that are large relative to the frame and absolute full-scale.
    threshold = max(0.35, 8.0 * frame_rms)
    count = int(np.sum(np.abs(diffs) > threshold))
    # Do not let a single clipped packet dominate the count.
    return min(count, max(1, int(0.002 * sample_rate)))


@dataclass
class OperationalMetricsTracker:
    speech_threshold: float = 0.5
    repeat_tolerance: float = 1e-5
    zero_threshold: float = 1e-7
    noise_floor_alpha: float = 0.95
    prev_input: torch.Tensor | None = None
    prev_input_last_sample: float | None = None
    prev_expected_start_sec: float | None = None
    noise_floor_dbfs: float | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add_chunk(
        self,
        input_chunk: AudioChunk,
        output_chunk: AudioChunk,
        processing_time_ms: float,
        added_latency_ms: float,
        dynamic_buffer_ms: float,
        stage_metrics: list[dict[str, Any]] | None = None,
        queue_underrun_count: int = 0,
        queue_overrun_count: int = 0,
    ) -> dict[str, Any]:
        input_x = _mono_np(input_chunk.samples)
        output_x = _mono_np(output_chunk.samples)
        input_rms_db = rms_dbfs(input_chunk.samples)
        output_rms_db = rms_dbfs(output_chunk.samples)
        speech_prob = _safe_float(
            output_chunk.metadata.get("speech_prob", input_chunk.metadata.get("speech_prob", 0.0))
        )
        is_speech = speech_prob >= self.speech_threshold

        if not is_speech:
            if self.noise_floor_dbfs is None:
                self.noise_floor_dbfs = input_rms_db
            else:
                alpha = self.noise_floor_alpha
                self.noise_floor_dbfs = alpha * self.noise_floor_dbfs + (1.0 - alpha) * input_rms_db
        approximate_snr_db = (
            input_rms_db - self.noise_floor_dbfs if is_speech and self.noise_floor_dbfs is not None else None
        )

        zero_frame = int(input_x.size > 0 and float(np.max(np.abs(input_x))) <= self.zero_threshold)
        zero_runs, max_zero_run = _zero_run_stats(input_x, self.zero_threshold)
        repeated_frame = 0
        if self.prev_input is not None and input_chunk.samples.shape == self.prev_input.shape:
            repeated_frame = int(
                torch.max(torch.abs(input_chunk.samples.detach().cpu() - self.prev_input)).item()
                <= self.repeat_tolerance
            )

        timestamp_gap_ms = 0.0
        dropout_count = 0
        if self.prev_expected_start_sec is not None:
            delta_ms = (input_chunk.start_time_sec - self.prev_expected_start_sec) * 1000.0
            if abs(delta_ms) > 0.5:
                timestamp_gap_ms = float(delta_ms)
                dropout_count = int(delta_ms > 0.5)
        self.prev_expected_start_sec = input_chunk.start_time_sec + input_chunk.duration_sec

        input_clicks = _count_clicks(input_x, input_chunk.sample_rate, self.prev_input_last_sample)
        output_clicks = _count_clicks(output_x, output_chunk.sample_rate, None)
        self.prev_input_last_sample = float(input_x[-1]) if input_x.size else self.prev_input_last_sample
        self.prev_input = input_chunk.samples.detach().cpu().clone()

        limiter_gain_reductions = [
            _safe_float(row.get("limiter_gain_reduction_db", row.get("final_limiter_gain_reduction_db", 0.0)))
            for row in (stage_metrics or [])
            if "limiter_gain_reduction_db" in row or "final_limiter_gain_reduction_db" in row
        ]
        limiter_gain_reduction_db = min(limiter_gain_reductions) if limiter_gain_reductions else 0.0
        decrackle_repairs = int(
            sum(_safe_float(row.get("decrackle_repaired_click_count", 0.0)) for row in (stage_metrics or []))
        )
        decrackle_samples = int(
            sum(_safe_float(row.get("decrackle_repaired_samples", 0.0)) for row in (stage_metrics or []))
        )

        duration_ms = max(input_chunk.duration_sec * 1000.0, 1e-9)
        input_spectral = _spectral_metrics(input_chunk.samples, input_chunk.sample_rate)
        output_spectral = _spectral_metrics(output_chunk.samples, output_chunk.sample_rate)
        row = {
            "input_rms_dbfs": input_rms_db,
            "output_rms_dbfs": output_rms_db,
            "speech_prob": speech_prob,
            "speech_active_rms_dbfs": output_rms_db if is_speech else None,
            "input_clipping_pct": float(np.mean(np.abs(input_x) >= 1.0) * 100.0) if input_x.size else 0.0,
            "output_clipping_pct": float(np.mean(np.abs(output_x) >= 1.0) * 100.0) if output_x.size else 0.0,
            "limiter_gain_reduction_db": limiter_gain_reduction_db,
            "decrackle_repaired_click_count": decrackle_repairs,
            "decrackle_repaired_samples": decrackle_samples,
            "dropout_count": dropout_count,
            "dropout_duration_ms": max(0.0, timestamp_gap_ms),
            "timestamp_gap_ms": timestamp_gap_ms,
            "input_discontinuity_count": input_clicks,
            "output_discontinuity_count": output_clicks,
            "zero_frame_count": zero_frame,
            "zero_run_count": zero_runs,
            "max_zero_run_samples": max_zero_run,
            "repeated_frame_count": repeated_frame,
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "approx_snr_db": approximate_snr_db,
            "processing_rtf": float(processing_time_ms / duration_ms),
            "added_latency_ms": float(added_latency_ms),
            "dynamic_buffer_ms": float(dynamic_buffer_ms),
            "queue_underrun_count": int(queue_underrun_count),
            "queue_overrun_count": int(queue_overrun_count),
            **{f"input_{key}": value for key, value in input_spectral.items()},
            **{f"output_{key}": value for key, value in output_spectral.items()},
        }
        row["spectral_centroid_delta_hz"] = (
            row["output_spectral_centroid_hz"] - row["input_spectral_centroid_hz"]
        )
        row["high_frequency_energy_ratio_delta"] = (
            row["output_high_frequency_energy_ratio"] - row["input_high_frequency_energy_ratio"]
        )
        row["narrowband_score"] = row["input_narrowband_likelihood"]
        self.rows.append(row)
        return row

    def summary(self) -> dict[str, Any]:
        return aggregate_operational(self.rows)


def aggregate_operational(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [_safe_float(row.get(key)) for row in rows if row.get(key) is not None]

    input_clip = vals("input_clipping_pct")
    output_clip = vals("output_clipping_pct")
    limiter = vals("limiter_gain_reduction_db")
    snr = vals("approx_snr_db")
    return {
        "input_rms_dbfs_mean": _safe_mean(vals("input_rms_dbfs")),
        "output_rms_dbfs_mean": _safe_mean(vals("output_rms_dbfs")),
        "speech_active_rms_dbfs_mean": _safe_mean(vals("speech_active_rms_dbfs")),
        "input_clipping_pct": _safe_mean(input_clip) or 0.0,
        "output_clipping_pct": _safe_mean(output_clip) or 0.0,
        "clipping_pct": _safe_mean(output_clip) or 0.0,
        "limiter_gain_reduction_db_min": min(limiter) if limiter else 0.0,
        "decrackle_repaired_click_count": int(sum(vals("decrackle_repaired_click_count"))),
        "decrackle_repaired_samples": int(sum(vals("decrackle_repaired_samples"))),
        "dropout_count": int(sum(vals("dropout_count"))),
        "dropout_duration_ms": float(sum(vals("dropout_duration_ms"))),
        "input_discontinuity_count": int(sum(vals("input_discontinuity_count"))),
        "output_discontinuity_count": int(sum(vals("output_discontinuity_count"))),
        "discontinuity_count": int(sum(vals("output_discontinuity_count"))),
        "zero_frame_count": int(sum(vals("zero_frame_count"))),
        "repeated_frame_count": int(sum(vals("repeated_frame_count"))),
        "noise_floor_dbfs": _safe_mean(vals("noise_floor_dbfs")),
        "approx_snr_db": _safe_mean(snr),
        "spectral_centroid_hz": _safe_mean(vals("output_spectral_centroid_hz")),
        "spectral_rolloff_hz": _safe_mean(vals("output_spectral_rolloff_hz")),
        "high_frequency_energy_ratio": _safe_mean(vals("output_high_frequency_energy_ratio")),
        "narrowband_score": _safe_mean(vals("narrowband_score")) or 0.0,
        "processing_rtf": _safe_mean(vals("processing_rtf")) or 0.0,
        "processing_rtf_max": _safe_max(vals("processing_rtf")) or 0.0,
        "added_latency_ms": _safe_max(vals("added_latency_ms")) or 0.0,
        "queue_underrun_count": int(sum(vals("queue_underrun_count"))),
        "queue_overrun_count": int(sum(vals("queue_overrun_count"))),
        "input_level_dbfs": linear_to_db(
            max(_safe_mean([10 ** (v / 20.0) for v in vals("input_rms_dbfs")]) or 0.0, 1e-12)
        ),
    }
