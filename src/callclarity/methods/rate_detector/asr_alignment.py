from __future__ import annotations

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import MethodUnavailable


@register_method("rate_detector", "asr_alignment")
class AsrAlignmentRateDetector(BaseStreamingProcessor):
    name = "asr_alignment"
    realtime_safe = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        backend = self.config.get("backend", "none")
        if backend in {None, "none"}:
            raise MethodUnavailable(
                "asr_alignment needs an installed/configured backend such as MFA, WhisperX, or faster-whisper."
            )
