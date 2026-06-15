from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@register_method("filter", "dc_highpass")
class DcHighpassFilterProcessor(BaseStreamingProcessor):
    name = "dc_highpass"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.dc_x_prev: np.ndarray | None = None
        self.dc_y_prev: np.ndarray | None = None
        self.sos: np.ndarray | None = None
        self.sos_state: np.ndarray | None = None
        self.presence_ba: tuple[np.ndarray, np.ndarray] | None = None
        self.presence_state: np.ndarray | None = None

    def reset(self) -> None:
        self.sample_rate = None
        self.channels = None
        self.dc_x_prev = None
        self.dc_y_prev = None
        self.sos = None
        self.sos_state = None
        self.presence_ba = None
        self.presence_state = None

    def _configure(self, sample_rate: int, channels: int) -> None:
        if self.sample_rate == sample_rate and self.channels == channels:
            return
        self.sample_rate = sample_rate
        self.channels = channels
        self.dc_x_prev = np.zeros(channels, dtype=np.float32)
        self.dc_y_prev = np.zeros(channels, dtype=np.float32)

        hp_cfg = self.config.get("highpass", {})
        if bool(hp_cfg.get("enabled", True)):
            cutoff = min(float(hp_cfg.get("cutoff_hz", 90.0)), sample_rate / 2.0 - 10.0)
            order = int(hp_cfg.get("order", 2))
            self.sos = signal.butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")
            zi = signal.sosfilt_zi(self.sos)
            self.sos_state = zi[:, None, :].repeat(channels, axis=1)
        else:
            self.sos = None
            self.sos_state = None

        presence_cfg = self.config.get("presence", {})
        if bool(presence_cfg.get("enabled", False)):
            freq = min(float(presence_cfg.get("center_hz", 3000.0)), sample_rate / 2.0 - 10.0)
            q = float(presence_cfg.get("q", 1.0))
            self.presence_ba = signal.iirpeak(freq, q, fs=sample_rate)
            b, a = self.presence_ba
            zi2 = signal.lfilter_zi(b, a)
            self.presence_state = zi2[None, :].repeat(channels, axis=0)
        else:
            self.presence_ba = None
            self.presence_state = None

    def _dc_block(self, x: np.ndarray) -> np.ndarray:
        dc_cfg = self.config.get("dc_blocker", {})
        if not bool(dc_cfg.get("enabled", True)):
            return x
        assert self.dc_x_prev is not None
        assert self.dc_y_prev is not None
        radius = float(dc_cfg.get("radius", 0.995))
        y = np.empty_like(x)
        for idx in range(x.shape[-1]):
            y[:, idx] = x[:, idx] - self.dc_x_prev + radius * self.dc_y_prev
            self.dc_x_prev = x[:, idx]
            self.dc_y_prev = y[:, idx]
        return y

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not bool(self.config.get("enabled", True)):
            return ProcessResult(chunk=chunk, algorithmic_latency_ms=0.0)
        x = chunk.samples.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self._configure(chunk.sample_rate, int(x.shape[0]))
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        y_np = self._dc_block(x_np)

        if self.sos is not None and self.sos_state is not None:
            y_np, self.sos_state = signal.sosfilt(self.sos, y_np, axis=-1, zi=self.sos_state)

        presence_boost_db = 0.0
        if self.presence_ba is not None and self.presence_state is not None:
            b, a = self.presence_ba
            presence, self.presence_state = signal.lfilter(b, a, y_np, axis=-1, zi=self.presence_state)
            presence_boost_db = float(self.config.get("presence", {}).get("boost_db", 1.5))
            y_np = y_np + (_db_to_linear(presence_boost_db) - 1.0) * presence

        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32)).to(
            device=chunk.samples.device,
            dtype=chunk.samples.dtype,
        )
        metadata = merge_metadata(chunk, dc_highpass_applied=True)
        return ProcessResult(
            chunk=AudioChunk(
                y.contiguous(),
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics={
                "dc_blocker_enabled": bool(self.config.get("dc_blocker", {}).get("enabled", True)),
                "highpass_cutoff_hz": float(self.config.get("highpass", {}).get("cutoff_hz", 90.0)),
                "presence_boost_db": presence_boost_db,
            },
            algorithmic_latency_ms=0.0,
        )
