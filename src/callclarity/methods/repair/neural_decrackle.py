from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.models.tiny_decrackler import TinyDecrackler
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult
from callclarity.utils.device import resolve_torch_device


@register_method("repair", "neural_decrackle")
class NeuralDecrackleProcessor(BaseStreamingProcessor):
    """Checkpoint-backed lightweight causal ML de-crackle processor."""

    name = "neural_decrackle"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        ckpt = self.config.get("checkpoint_path")
        if not ckpt:
            raise MethodUnavailable("Set repair.neural_decrackle.checkpoint_path before using neural_decrackle.")
        path = Path(str(ckpt))
        if not path.exists():
            raise MethodUnavailable(f"neural_decrackle checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        model_cfg = dict(state.get("model_config", self.config.get("model", {})))
        self.model = TinyDecrackler(
            channels=int(model_cfg.get("channels", self.config.get("channels", 32))),
            kernel_size=int(model_cfg.get("kernel_size", self.config.get("kernel_size", 5))),
            dilations=tuple(model_cfg.get("dilations", self.config.get("dilations", [1, 2, 4, 8]))),
            max_correction=float(model_cfg.get("max_correction", self.config.get("max_correction", 0.75))),
        )
        self.model.load_state_dict(state.get("model", state))
        self.device_requested, self.device = resolve_torch_device(self.config.get("device", "cpu"))
        self.model.to(torch.device(self.device)).eval()
        self.blend = float(self.config.get("blend", 1.0))
        self.max_frame_correction = float(self.config.get("max_frame_correction", 0.35))
        self.tail: torch.Tensor | None = None

    @property
    def algorithmic_latency_ms(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.tail = None

    def _context(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        channels = int(x.shape[0])
        tail_len = max(0, int(self.model.receptive_field) - 1)
        if self.tail is None or self.tail.shape[0] != channels:
            tail = torch.zeros(channels, 0, dtype=x.dtype)
        else:
            tail = self.tail.to(dtype=x.dtype)
        context = torch.cat([tail.to(x.device), x], dim=-1)
        if tail_len:
            self.tail = context.detach().cpu()[:, -tail_len:].clone()
        else:
            self.tail = None
        return context, int(tail.shape[-1])

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not bool(self.config.get("enabled", True)):
            return ProcessResult(chunk=chunk, algorithmic_latency_ms=0.0)
        x = chunk.samples.detach().float()
        was_1d = x.ndim == 1
        if was_1d:
            x = x.unsqueeze(0)
        context, history_len = self._context(x.cpu())
        with torch.no_grad():
            enhanced, correction = self.model(context.to(self.device))
        enhanced = enhanced.detach().cpu()[:, history_len : history_len + x.shape[-1]]
        correction = correction.detach().cpu()[:, history_len : history_len + x.shape[-1]]
        if self.max_frame_correction > 0:
            correction = correction.clamp(-self.max_frame_correction, self.max_frame_correction)
            enhanced = (x.cpu() + correction).clamp(-1.0, 1.0)
        y = x.cpu() * (1.0 - self.blend) + enhanced * self.blend
        y = y.to(device=chunk.samples.device, dtype=chunk.samples.dtype).contiguous()
        if was_1d:
            y = y.squeeze(0)
        correction_rms = float(torch.sqrt(torch.mean(correction.float() ** 2) + 1e-12).item())
        metadata = merge_metadata(chunk, neural_decrackle=True)
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "neural_decrackle_device_requested": self.device_requested,
                "neural_decrackle_device": self.device,
                "neural_decrackle_correction_rms": correction_rms,
                "neural_decrackle_receptive_field_samples": int(self.model.receptive_field),
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
