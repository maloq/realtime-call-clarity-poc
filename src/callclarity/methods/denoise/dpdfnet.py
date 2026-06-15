from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult


@register_method("denoise", "dpdfnet")
class DpdfnetDenoiser(BaseStreamingProcessor):
    name = "dpdfnet"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.model_name = str(self.config.get("model", "dpdfnet2"))
        self.onnx_path = self.config.get("onnx_path")
        self.verbose = bool(self.config.get("verbose", False))
        self._pending_output = np.zeros(0, dtype=np.float32)
        self._model_sample_rate = int(self.config.get("model_sample_rate", 16000))
        self._win_len: int | None = None
        self._hop_size: int | None = None
        self._enhancer = None

        if not self.enabled:
            return

        try:
            dpdfnet = importlib.import_module("dpdfnet")
        except Exception as exc:
            raise MethodUnavailable(
                "DPDFNet is optional. Install it with `pip install dpdfnet`, then run "
                "`dpdfnet download dpdfnet2` or set denoise.onnx_path to a local ONNX model."
            ) from exc

        try:
            self._enhancer = dpdfnet.StreamEnhancer(
                model=self.model_name,
                onnx_path=self.onnx_path,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise MethodUnavailable(
                "DPDFNet could not initialize its streaming model. Install `dpdfnet`, "
                "pre-download a model with `dpdfnet download dpdfnet2`, or set "
                "denoise.onnx_path to a valid DPDFNet ONNX file."
            ) from exc

        self._model_sample_rate = int(
            getattr(self._enhancer, "_model_sr", self._model_sample_rate)
        )
        self._win_len = getattr(self._enhancer, "_win_len", None)
        self._hop_size = getattr(self._enhancer, "_hop_size", None)

    @property
    def algorithmic_latency_ms(self) -> float:
        if not self.enabled:
            return 0.0
        if "algorithmic_latency_ms" in self.config:
            return float(self.config["algorithmic_latency_ms"])
        if self._win_len is not None and self._model_sample_rate > 0:
            return 1000.0 * float(self._win_len) / float(self._model_sample_rate)
        return 20.0

    @property
    def lookahead_ms(self) -> float:
        if not self.enabled:
            return 0.0
        if "lookahead_ms" in self.config:
            return float(self.config["lookahead_ms"])
        if self._hop_size is not None and self._model_sample_rate > 0:
            return 1000.0 * float(self._hop_size) / float(self._model_sample_rate)
        return 10.0

    def reset(self) -> None:
        self._pending_output = np.zeros(0, dtype=np.float32)
        if self._enhancer is not None and hasattr(self._enhancer, "reset"):
            self._enhancer.reset()

    def _to_mono_numpy(self, samples: torch.Tensor) -> np.ndarray:
        x = samples.detach().cpu().float()
        if x.ndim == 1:
            mono = x
        elif x.shape[0] == 1:
            mono = x[0]
        else:
            mono = x.mean(dim=0)
        return np.ascontiguousarray(mono.numpy(), dtype=np.float32)

    def _consume_exact_output(self, enhanced: np.ndarray, target_samples: int) -> np.ndarray:
        enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)
        if self._pending_output.size:
            enhanced = np.concatenate([self._pending_output, enhanced])
        if enhanced.shape[0] >= target_samples:
            out = enhanced[:target_samples]
            self._pending_output = enhanced[target_samples:].astype(np.float32, copy=False)
            return out.astype(np.float32, copy=False)

        out = np.zeros(target_samples, dtype=np.float32)
        out[: enhanced.shape[0]] = enhanced
        self._pending_output = np.zeros(0, dtype=np.float32)
        return out

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not self.enabled or self._enhancer is None:
            return super().process(chunk)

        mono = self._to_mono_numpy(chunk.samples)
        enhanced = self._enhancer.process(mono, sample_rate=chunk.sample_rate)
        out_np = self._consume_exact_output(enhanced, chunk.num_samples)
        out = torch.from_numpy(out_np).to(device=chunk.samples.device).unsqueeze(0).contiguous()
        metadata = merge_metadata(
            chunk,
            denoised=True,
            neural_denoiser="DPDFNet",
            dpdfnet_model=self.model_name,
        )
        return ProcessResult(
            chunk=AudioChunk(out, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "realtime_safe": True,
                "neural_denoiser": "DPDFNet",
                "dpdfnet_model": self.model_name,
                "dpdfnet_model_sample_rate": self._model_sample_rate,
                "dpdfnet_pending_samples": int(self._pending_output.shape[0]),
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
