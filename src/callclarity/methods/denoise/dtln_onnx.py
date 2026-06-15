from __future__ import annotations

from pathlib import Path

from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import MethodUnavailable


@register_method("denoise", "dtln_onnx")
class DtlnOnnxDenoiser(BaseStreamingProcessor):
    name = "dtln_onnx"
    realtime_safe = True

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        model_path = self.config.get("model_path")
        if not model_path:
            raise MethodUnavailable("Set denoise.model_path to a DTLN ONNX model before using dtln_onnx.")
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise MethodUnavailable("Install optional dependency with `pip install onnxruntime`.") from exc
        if not Path(model_path).exists():
            raise MethodUnavailable(f"DTLN ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(self.config.get("algorithmic_latency_ms", 32.0))
