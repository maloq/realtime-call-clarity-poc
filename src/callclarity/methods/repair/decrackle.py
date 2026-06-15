from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


_PRESETS: dict[str, dict[str, float]] = {
    "mild": {
        "median_window_ms": 0.31,
        "scale_window_ms": 2.0,
        "detection_threshold": 8.0,
        "abs_threshold": 0.16,
        "rms_threshold_multiple": 2.7,
        "max_click_duration_ms": 1.0,
        "merge_gap_ms": 0.12,
        "repair_pad_ms": 0.12,
        "repair_blend": 0.75,
        "max_repair_fraction": 0.035,
    },
    "medium": {
        "median_window_ms": 0.31,
        "scale_window_ms": 2.5,
        "detection_threshold": 6.0,
        "abs_threshold": 0.10,
        "rms_threshold_multiple": 2.2,
        "max_click_duration_ms": 1.8,
        "merge_gap_ms": 0.20,
        "repair_pad_ms": 0.18,
        "repair_blend": 0.90,
        "max_repair_fraction": 0.12,
    },
    "aggressive": {
        "median_window_ms": 0.38,
        "scale_window_ms": 3.0,
        "detection_threshold": 4.5,
        "abs_threshold": 0.07,
        "rms_threshold_multiple": 1.8,
        "max_click_duration_ms": 3.0,
        "merge_gap_ms": 0.30,
        "repair_pad_ms": 0.25,
        "repair_blend": 1.0,
        "max_repair_fraction": 0.25,
    },
}


@dataclass(frozen=True)
class _Run:
    channel: int
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def _odd_window(samples: int) -> int:
    n = max(3, int(samples))
    return n if n % 2 else n + 1


def _ms_to_samples(ms: float, sample_rate: int, minimum: int = 1) -> int:
    return max(minimum, int(round(float(ms) * sample_rate / 1000.0)))


def _float_cfg(cfg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    if x.shape[-1] == 0:
        return x.copy()
    window = _odd_window(min(window, max(3, x.shape[-1] | 1)))
    radius = window // 2
    mode = "reflect" if x.shape[-1] > 1 else "edge"
    padded = np.pad(x, ((0, 0), (radius, radius)), mode=mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=-1)
    return np.median(windows, axis=-1).astype(np.float32, copy=False)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not bool(mask.any()):
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]


def _merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _blend_weights(length: int, core_start: int, core_end: int, blend: float) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32) * float(blend)
    if core_start > 0:
        weights[:core_start] *= np.linspace(0.25, 1.0, core_start, dtype=np.float32)
    if core_end < length:
        weights[core_end:] *= np.linspace(1.0, 0.25, length - core_end, dtype=np.float32)
    return np.clip(weights, 0.0, 1.0)


@register_method("repair", "decrackle")
class DecrackleProcessor(BaseStreamingProcessor):
    """Streaming conservative impulse/crackle suppressor.

    It detects short high-confidence outlier runs against a local median predictor,
    then replaces the affected samples with a short crossfaded interpolation.
    """

    name = "decrackle"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.prev_tail: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_tail = None

    @property
    def algorithmic_latency_ms(self) -> float:
        return 0.0

    def _resolved_cfg(self) -> dict[str, Any]:
        strength = self.config.get("strength", "mild")
        if isinstance(strength, str):
            preset = dict(_PRESETS.get(strength.lower(), _PRESETS["mild"]))
        else:
            value = max(0.0, min(1.0, float(strength)))
            mild = _PRESETS["mild"]
            aggressive = _PRESETS["aggressive"]
            preset = {
                key: mild[key] + (aggressive[key] - mild[key]) * value
                for key in mild
            }
        preset.update({k: v for k, v in self.config.items() if k not in {"enabled", "strength"}})
        return preset

    def _tail_samples(self, cfg: dict[str, Any], sample_rate: int) -> int:
        scale = _ms_to_samples(_float_cfg(cfg, "scale_window_ms", 2.0), sample_rate, 3)
        max_click = _ms_to_samples(_float_cfg(cfg, "max_click_duration_ms", 1.0), sample_rate, 1)
        pad = _ms_to_samples(_float_cfg(cfg, "repair_pad_ms", 0.12), sample_rate, 0)
        return max(_odd_window(scale), max_click + 2 * pad + 4)

    def _detect_runs(
        self,
        context: np.ndarray,
        history_len: int,
        frame_len: int,
        sample_rate: int,
        cfg: dict[str, Any],
    ) -> tuple[list[_Run], np.ndarray]:
        median_window = _ms_to_samples(_float_cfg(cfg, "median_window_ms", 0.31), sample_rate, 3)
        scale_window = _ms_to_samples(_float_cfg(cfg, "scale_window_ms", 2.0), sample_rate, 5)
        median = _rolling_median(context, median_window)
        residual = np.abs(context - median)
        local_scale = 1.4826 * _rolling_median(residual, scale_window) + 1e-6
        max_click = _ms_to_samples(_float_cfg(cfg, "max_click_duration_ms", 1.0), sample_rate, 1)
        merge_gap = _ms_to_samples(_float_cfg(cfg, "merge_gap_ms", 0.12), sample_rate, 0)
        detection_threshold = _float_cfg(cfg, "detection_threshold", 8.0)
        abs_threshold = _float_cfg(cfg, "abs_threshold", 0.16)
        rms_multiple = _float_cfg(cfg, "rms_threshold_multiple", 2.7)
        runs: list[_Run] = []
        frame_start = history_len
        frame_end = history_len + frame_len
        for channel in range(context.shape[0]):
            frame = context[channel, frame_start:frame_end]
            frame_rms = float(np.sqrt(np.mean(frame * frame) + 1e-12)) if frame.size else 0.0
            threshold = np.maximum(
                np.maximum(abs_threshold, detection_threshold * local_scale[channel]),
                rms_multiple * frame_rms,
            )
            mask = residual[channel] > threshold
            for start, end in _merge_close_runs(_runs(mask), merge_gap):
                if end <= frame_start or start >= frame_end:
                    continue
                if end - start > max_click:
                    continue
                current_start = max(start, frame_start)
                current_end = min(end, frame_end)
                if current_end <= current_start:
                    continue
                score = float(np.max(residual[channel, start:end]))
                runs.append(_Run(channel, current_start, current_end, score))
        return runs, median

    def _repair_run(
        self,
        y: np.ndarray,
        median: np.ndarray,
        run: _Run,
        history_len: int,
        cfg: dict[str, Any],
        sample_rate: int,
    ) -> int:
        pad = _ms_to_samples(_float_cfg(cfg, "repair_pad_ms", 0.12), sample_rate, 0)
        blend = max(0.0, min(1.0, _float_cfg(cfg, "repair_blend", 0.75)))
        channel = run.channel
        start = max(history_len, run.start - pad)
        end = min(y.shape[-1], run.end + pad)
        if end <= start:
            return 0
        length = end - start
        left_idx = start - 1
        right_idx = end
        if left_idx >= 0 and right_idx < y.shape[-1]:
            replacement = np.linspace(
                y[channel, left_idx],
                y[channel, right_idx],
                length + 2,
                dtype=np.float32,
            )[1:-1]
        else:
            replacement = median[channel, start:end].astype(np.float32, copy=False)
        core_start = max(0, run.start - start)
        core_end = min(length, run.end - start)
        weights = _blend_weights(length, core_start, core_end, blend)
        y[channel, start:end] = y[channel, start:end] * (1.0 - weights) + replacement * weights
        return max(0, min(run.end, y.shape[-1]) - max(run.start, history_len))

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not bool(self.config.get("enabled", True)):
            return ProcessResult(chunk=chunk, algorithmic_latency_ms=0.0)
        x = chunk.samples.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=True)
        cfg = self._resolved_cfg()
        channels = int(x_np.shape[0])
        if self.prev_tail is None or self.prev_tail.shape[0] != channels:
            tail = np.zeros((channels, 0), dtype=np.float32)
        else:
            tail = self.prev_tail.astype(np.float32, copy=False)
        context = np.concatenate([tail, x_np], axis=-1)
        history_len = int(tail.shape[-1])
        runs, median = self._detect_runs(context, history_len, x_np.shape[-1], chunk.sample_rate, cfg)
        max_repair_samples = max(
            1,
            int(round(_float_cfg(cfg, "max_repair_fraction", 0.035) * max(x_np.shape[-1], 1))),
        )
        selected: list[_Run] = []
        repaired_samples = 0
        for run in sorted(runs, key=lambda r: r.score, reverse=True):
            if repaired_samples >= max_repair_samples:
                break
            if repaired_samples + run.length > max_repair_samples:
                continue
            selected.append(run)
            repaired_samples += run.length
        selected.sort(key=lambda r: (r.channel, r.start))

        y_context = context.copy()
        actual_repaired_samples = 0
        for run in selected:
            actual_repaired_samples += self._repair_run(
                y_context,
                median,
                run,
                history_len,
                cfg,
                chunk.sample_rate,
            )
        y_np = y_context[:, history_len : history_len + x_np.shape[-1]]
        tail_samples = min(self._tail_samples(cfg, chunk.sample_rate), y_context.shape[-1])
        self.prev_tail = y_context[:, -tail_samples:].copy() if tail_samples else None
        y = torch.from_numpy(y_np).to(device=chunk.samples.device, dtype=chunk.samples.dtype).contiguous()
        repaired_count = len(selected)
        peak_before = float(np.max(np.abs(x_np))) if x_np.size else 0.0
        peak_after = float(np.max(np.abs(y_np))) if y_np.size else 0.0
        events = []
        if repaired_count:
            events.append(
                {
                    "event": "decrackle_repair",
                    "count": repaired_count,
                    "samples": actual_repaired_samples,
                }
            )
        metadata = merge_metadata(
            chunk,
            decrackle_repaired_clicks=repaired_count,
            decrackle_repaired_samples=actual_repaired_samples,
        )
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "decrackle_detected_click_count": len(runs),
                "decrackle_repaired_click_count": repaired_count,
                "decrackle_repaired_samples": actual_repaired_samples,
                "decrackle_peak_before": peak_before,
                "decrackle_peak_after": peak_after,
                "decrackle_peak_reduction": max(0.0, peak_before - peak_after),
            },
            events=events,
            algorithmic_latency_ms=0.0,
        )
