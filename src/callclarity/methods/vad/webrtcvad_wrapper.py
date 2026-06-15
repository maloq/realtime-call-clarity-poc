from __future__ import annotations

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import MethodUnavailable


@register_method("vad", "webrtcvad")
class WebrtcVadProcessor(BaseStreamingProcessor):
    name = "webrtcvad"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        try:
            import webrtcvad  # noqa: F401
        except Exception as exc:
            raise MethodUnavailable("Install optional dependency with `pip install webrtcvad`.") from exc
