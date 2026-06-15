from __future__ import annotations

import numpy as np
import torch

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


def _zero_runs(x: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    silent = np.abs(x) <= threshold
    if not bool(silent.any()):
        return []
    padded = np.concatenate(([False], silent, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]


@register_method("repair", "dropout_click")
class DropoutClickRepairProcessor(BaseStreamingProcessor):
    name = "dropout_click"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.prev_frame: torch.Tensor | None = None
        self.prev_last_sample: np.ndarray | None = None
        self.prev_expected_start_sec: float | None = None

    def reset(self) -> None:
        self.prev_frame = None
        self.prev_last_sample = None
        self.prev_expected_start_sec = None

    def _repair_tiny_gaps(self, x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
        cfg = self.config.get("tiny_gap", {})
        if not bool(cfg.get("enabled", True)):
            return x, 0, 0
        max_samples = max(1, int(round(float(cfg.get("max_ms", 2.0)) * sample_rate / 1000.0)))
        threshold = float(cfg.get("zero_threshold", 1e-7))
        repaired = x.copy()
        repaired_runs = 0
        repaired_samples = 0
        for channel in range(repaired.shape[0]):
            for start, end in _zero_runs(repaired[channel], threshold):
                length = end - start
                if length > max_samples:
                    continue
                if start == 0 or end >= repaired.shape[1]:
                    continue
                left = repaired[channel, start - 1]
                right = repaired[channel, end]
                repaired[channel, start:end] = np.linspace(left, right, length + 2, dtype=np.float32)[1:-1]
                repaired_runs += 1
                repaired_samples += length
        return repaired, repaired_runs, repaired_samples

    def _repair_clicks(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        cfg = self.config.get("clicks", {})
        if not bool(cfg.get("enabled", True)) or x.shape[-1] < 3:
            return x, 0
        repaired = x.copy()
        abs_threshold = float(cfg.get("abs_threshold", 0.35))
        rms_multiple = float(cfg.get("rms_multiple", 8.0))
        max_repairs = int(cfg.get("max_repairs_per_frame", 8))
        total = 0
        for channel in range(repaired.shape[0]):
            y = repaired[channel]
            rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
            threshold = max(abs_threshold, rms_multiple * rms)
            previous = self.prev_last_sample[channel] if self.prev_last_sample is not None else y[0]
            diffs = np.abs(np.diff(y, prepend=previous))
            candidates = np.where(diffs > threshold)[0]
            for idx in candidates[:max_repairs]:
                if idx <= 0 or idx >= y.shape[0] - 1:
                    continue
                local = 0.5 * (y[idx - 1] + y[idx + 1])
                if abs(y[idx] - local) > threshold:
                    y[idx] = local
                    total += 1
        return repaired, total

    def _conceal_zero_frame(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        cfg = self.config.get("zero_frame", {})
        if not bool(cfg.get("conceal", True)):
            return x, False
        over_threshold = x.numel() != 0 and float(x.abs().max().item()) > float(cfg.get("threshold", 1e-7))
        if self.prev_frame is None or x.numel() == 0 or over_threshold:
            return x, False
        previous = self.prev_frame.to(device=x.device, dtype=x.dtype)
        if previous.shape != x.shape:
            return x, False
        fade = torch.linspace(
            float(cfg.get("start_gain", 0.85)),
            0.0,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        return previous * fade[None, :], True

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not bool(self.config.get("enabled", True)):
            return ProcessResult(chunk=chunk, algorithmic_latency_ms=0.0)
        x = chunk.samples.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        timestamp_gap_ms = 0.0
        events: list[dict] = []
        if self.prev_expected_start_sec is not None:
            timestamp_gap_ms = (chunk.start_time_sec - self.prev_expected_start_sec) * 1000.0
            if abs(timestamp_gap_ms) > float(self.config.get("timestamp_tolerance_ms", 0.5)):
                events.append({"event": "timestamp_gap", "timestamp_gap_ms": timestamp_gap_ms})
        self.prev_expected_start_sec = chunk.start_time_sec + chunk.duration_sec

        x, zero_frame_concealed = self._conceal_zero_frame(x)
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=True)
        x_np, gap_runs, gap_samples = self._repair_tiny_gaps(x_np, chunk.sample_rate)
        x_np, click_repairs = self._repair_clicks(x_np)
        y = torch.from_numpy(x_np).to(device=chunk.samples.device, dtype=chunk.samples.dtype).contiguous()

        repeated_frame = False
        if self.prev_frame is not None and self.prev_frame.shape == chunk.samples.shape:
            repeated_frame = bool(
                torch.max(torch.abs(chunk.samples.detach().cpu() - self.prev_frame)).item() <= 1e-5
            )
            if repeated_frame:
                events.append({"event": "repeated_frame"})
        if zero_frame_concealed:
            events.append({"event": "zero_frame_concealed"})
        if click_repairs:
            events.append({"event": "click_repair", "count": click_repairs})
        if gap_runs:
            events.append({"event": "tiny_gap_repair", "count": gap_runs, "samples": gap_samples})

        self.prev_frame = chunk.samples.detach().cpu().clone()
        self.prev_last_sample = x_np[:, -1].copy() if x_np.shape[-1] else self.prev_last_sample
        metadata = merge_metadata(chunk, repaired_clicks=click_repairs, repaired_tiny_gaps=gap_runs)
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "timestamp_gap_ms": timestamp_gap_ms,
                "zero_frame_concealed": zero_frame_concealed,
                "repeated_frame": repeated_frame,
                "click_repair_count": click_repairs,
                "tiny_gap_repair_count": gap_runs,
                "tiny_gap_repair_samples": gap_samples,
            },
            events=events,
            algorithmic_latency_ms=0.0,
        )
