from __future__ import annotations

import shutil

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult


@register_method("denoise", "rnnoise_external")
class RnnoiseExternal(BaseStreamingProcessor):
    name = "rnnoise_external"
    realtime_safe = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        binary = str(self.config.get("binary", "rnnoise_demo"))
        if shutil.which(binary) is None:
            raise MethodUnavailable(f"RNNoise binary `{binary}` was not found on PATH.")
        self.binary = binary

    def process(self, chunk: AudioChunk) -> ProcessResult:
        del chunk
        raise MethodUnavailable(
            "rnnoise_external found the RNNoise demo binary, but this wrapper is not a live-safe "
            "streaming binding. The demo tool expects raw 48 kHz mono PCM files, so invoking it per "
            "audio callback would add unbounded latency. Use spectral_gate for the current real-time "
            "baseline or add a native RNNoise stateful binding behind this interface."
        )
