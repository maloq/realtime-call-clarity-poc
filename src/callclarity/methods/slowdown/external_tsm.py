from __future__ import annotations

import shutil

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import MethodUnavailable


class _ExternalTsm(BaseStreamingProcessor):
    binary_name = ""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        binary = str(self.config.get("binary", self.binary_name))
        if shutil.which(binary) is None:
            raise MethodUnavailable(f"External TSM binary `{binary}` was not found on PATH.")
        self.binary = binary


@register_method("slowdown", "rubberband_external")
class RubberbandExternal(_ExternalTsm):
    name = "rubberband_external"
    binary_name = "rubberband"


@register_method("slowdown", "soundtouch_external")
class SoundtouchExternal(_ExternalTsm):
    name = "soundtouch_external"
    binary_name = "soundstretch"


@register_method("slowdown", "signalsmith_external")
class SignalsmithExternal(_ExternalTsm):
    name = "signalsmith_external"
    binary_name = "signalsmith-stretch"
