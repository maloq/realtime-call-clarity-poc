from __future__ import annotations

from pathlib import Path

import torch

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.models.tiny_mask_gru import TinyMaskGru
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult


@register_method("denoise", "tiny_mask_gru")
class TinyMaskGruInfer(BaseStreamingProcessor):
    name = "tiny_mask_gru"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.model = TinyMaskGru(
            n_fft=int(self.config.get("n_fft", 256)),
            hop_length=int(self.config.get("hop_length", 128)),
            hidden_size=int(self.config.get("hidden_size", 96)),
            num_layers=int(self.config.get("num_layers", 1)),
        )
        ckpt = self.config.get("checkpoint_path")
        if not ckpt:
            raise MethodUnavailable("Set denoise.checkpoint_path before using tiny_mask_gru inference.")
        if not Path(ckpt).exists():
            raise MethodUnavailable(f"tiny_mask_gru checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state.get("model", state))
        self.model.eval()
        self.hidden: torch.Tensor | None = None

    @property
    def algorithmic_latency_ms(self) -> float:
        return 1000.0 * int(self.config.get("hop_length", 128)) / 16000.0

    def reset(self) -> None:
        self.hidden = None

    def process(self, chunk: AudioChunk) -> ProcessResult:
        with torch.no_grad():
            enhanced, self.hidden, _ = self.model(chunk.samples, self.hidden)
        return ProcessResult(
            chunk=AudioChunk(enhanced, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, dict(chunk.metadata)),
            metrics={"realtime_safe": True},
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
