from __future__ import annotations

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import MethodUnavailable


@register_method("denoise", "deepfilternet")
class DeepFilterNetWrapper(BaseStreamingProcessor):
    name = "deepfilternet"
    realtime_safe = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        try:
            import df  # noqa: F401
        except Exception as exc:
            raise MethodUnavailable(
                "DeepFilterNet is optional. Install/configure deepfilternet before selecting denoise=deepfilternet."
            ) from exc
