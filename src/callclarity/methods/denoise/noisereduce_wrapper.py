from __future__ import annotations

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult


@register_method("denoise", "noisereduce")
class NoiseReduceWrapper(BaseStreamingProcessor):
    name = "noisereduce"
    realtime_safe = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        try:
            import noisereduce as nr
        except Exception as exc:
            raise MethodUnavailable("Install optional dependency with `pip install noisereduce`.") from exc
        self.nr = nr

    def process(self, chunk: AudioChunk) -> ProcessResult:
        data = chunk.samples.detach().cpu().numpy()
        out = self.nr.reduce_noise(y=data, sr=chunk.sample_rate, stationary=self.config.get("stationary", False))
        import torch

        return ProcessResult(
            chunk=AudioChunk(torch.as_tensor(out, dtype=chunk.samples.dtype), chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, dict(chunk.metadata)),
            metrics={"realtime_safe": False},
            algorithmic_latency_ms=1000.0,
        )
