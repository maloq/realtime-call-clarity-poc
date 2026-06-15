from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult


_MODEL_URLS = {
    "full": "https://huggingface.co/YatharthS/FlashSR/resolve/main/onnx/model.onnx",
    "lite": "https://raw.githubusercontent.com/ysharma3501/FlashSR/master/models/model_lite.onnx",
}


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)


def _resample_np(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return np.ascontiguousarray(x, dtype=np.float32)
    import math

    gcd = math.gcd(int(source_sr), int(target_sr))
    y = signal.resample_poly(x, int(target_sr) // gcd, int(source_sr) // gcd)
    return np.ascontiguousarray(y, dtype=np.float32)


class _FlashSrOnnxBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise MethodUnavailable(
                "FlashSR ONNX requires onnxruntime. Install it in the active environment."
            ) from exc

        self.variant = str(config.get("variant", "lite")).lower()
        if self.variant not in _MODEL_URLS:
            self.variant = "lite"
        cache_dir = Path(str(config.get("cache_dir", "data/checkpoints/flashsr"))).expanduser()
        default_name = "model_lite.onnx" if self.variant == "lite" else "model.onnx"
        self.model_path = Path(str(config.get("model_path") or cache_dir / default_name))
        if bool(config.get("auto_download", True)) and not self.model_path.exists():
            _download_file(_MODEL_URLS[self.variant], self.model_path)
        if not self.model_path.exists():
            raise MethodUnavailable(
                "FlashSR ONNX model is missing. Set bandwidth model_path or enable auto_download."
            )

        provider = str(config.get("provider", "CPUExecutionProvider"))
        available = set(ort.get_available_providers())
        if provider not in available:
            provider = "CPUExecutionProvider"
        options = ort.SessionOptions()
        num_threads = int(config.get("num_threads", 1))
        if num_threads > 0:
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), options, providers=[provider])
        self.provider = provider
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_rank = len(input_meta.shape)
        self.output_name = self.session.get_outputs()[0].name

    @property
    def device(self) -> str:
        return "cpu" if self.provider == "CPUExecutionProvider" else self.provider

    def run(self, x: np.ndarray) -> np.ndarray:
        if self.input_rank == 2:
            feed = np.ascontiguousarray(x.reshape(1, -1), dtype=np.float32)
        else:
            feed = np.ascontiguousarray(x.reshape(1, 1, -1), dtype=np.float32)
        y = self.session.run([self.output_name], {self.input_name: feed})[0]
        return np.ascontiguousarray(y.reshape(-1), dtype=np.float32)


@register_method("bandwidth", "flashsr_onnx")
class FlashSrOnnxProcessor(BaseStreamingProcessor):
    name = "flashsr_onnx"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.input_sample_rate = int(self.config.get("input_sample_rate", 16000))
        self.output_sample_rate = int(self.config.get("output_sample_rate", 48000))
        self.output_gain_db = float(self.config.get("output_gain_db", 0.0))
        self.clamp_output = bool(self.config.get("clamp_output", True))
        self.processing_mode = str(self.config.get("processing_mode", "overlap")).lower()
        self.block_ms = float(self.config.get("block_ms", 30.0))
        self.hop_ms = float(self.config.get("hop_ms", 15.0))
        self.fallback_gain_db = float(self.config.get("fallback_gain_db", 0.0))
        self.startup_fallback = str(self.config.get("startup_fallback", "silence")).lower()
        self.backend: _FlashSrOnnxBackend | None = None
        self.input_history = np.zeros(0, dtype=np.float32)
        self.output_accum = np.zeros(0, dtype=np.float32)
        self.output_weight = np.zeros(0, dtype=np.float32)
        self.input_seen_samples = 0
        self.read_output_pos = 0
        self.next_block_start = 0

    @property
    def algorithmic_latency_ms(self) -> float:
        if not self.enabled:
            return 0.0
        default = self.block_ms if self.processing_mode == "overlap" else 10.0
        return float(self.config.get("algorithmic_latency_ms", default))

    def reset(self) -> None:
        self.input_history = np.zeros(0, dtype=np.float32)
        self.output_accum = np.zeros(0, dtype=np.float32)
        self.output_weight = np.zeros(0, dtype=np.float32)
        self.input_seen_samples = 0
        self.read_output_pos = 0
        self.next_block_start = 0

    def warmup(self, sample_rate: int) -> None:
        del sample_rate
        if self.enabled:
            backend = self._ensure_backend()
            warmup_ms = self.block_ms if self.processing_mode == "overlap" else 10.0
            warmup = np.zeros(max(1, int(round(self.input_sample_rate * warmup_ms / 1000.0))), dtype=np.float32)
            backend.run(warmup)

    def _ensure_backend(self) -> _FlashSrOnnxBackend:
        if self.backend is None:
            self.backend = _FlashSrOnnxBackend(self.config)
        return self.backend

    def _to_mono_numpy(self, samples: torch.Tensor) -> np.ndarray:
        x = samples.detach().cpu().float()
        if x.ndim == 1:
            mono = x
        elif x.shape[0] == 1:
            mono = x[0]
        else:
            mono = x.mean(dim=0)
        return np.ascontiguousarray(mono.numpy(), dtype=np.float32)

    def _run_backend(self, backend: _FlashSrOnnxBackend, model_input: np.ndarray) -> np.ndarray:
        enhanced = backend.run(model_input)
        native_output_sr = 3 * int(self.input_sample_rate)
        if int(self.output_sample_rate) != native_output_sr:
            enhanced = _resample_np(enhanced, native_output_sr, int(self.output_sample_rate))
        if self.output_gain_db:
            enhanced = enhanced * float(10.0 ** (self.output_gain_db / 20.0))
        if self.clamp_output:
            enhanced = np.clip(enhanced, -1.0, 1.0)
        return np.ascontiguousarray(enhanced, dtype=np.float32)

    def _ensure_output_capacity(self, end: int) -> None:
        if end <= self.output_accum.shape[0]:
            return
        pad = end - self.output_accum.shape[0]
        self.output_accum = np.pad(self.output_accum, (0, pad)).astype(np.float32, copy=False)
        self.output_weight = np.pad(self.output_weight, (0, pad)).astype(np.float32, copy=False)

    def _ola_weights(self, length: int, overlap: int, is_first: bool) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        overlap = max(0, min(int(overlap), max(0, length // 2)))
        if overlap <= 1:
            return weights
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        if not is_first:
            weights[:overlap] = np.sin(0.5 * np.pi * ramp) ** 2
        weights[-overlap:] = np.cos(0.5 * np.pi * ramp) ** 2
        return np.clip(weights, 1e-4, 1.0)

    def _add_overlap_block(self, enhanced: np.ndarray, out_start: int, is_first: bool) -> None:
        out_end = out_start + int(enhanced.shape[0])
        self._ensure_output_capacity(out_end)
        block_samples = max(1, int(round(self.input_sample_rate * self.block_ms / 1000.0)))
        hop_samples = max(1, int(round(self.input_sample_rate * self.hop_ms / 1000.0)))
        overlap_in = max(0, block_samples - hop_samples)
        overlap_out = int(round(overlap_in * self.output_sample_rate / self.input_sample_rate))
        weights = self._ola_weights(enhanced.shape[0], overlap_out, is_first)
        self.output_accum[out_start:out_end] += enhanced * weights
        self.output_weight[out_start:out_end] += weights

    def _generate_overlap_blocks(self, backend: _FlashSrOnnxBackend) -> int:
        block_samples = max(1, int(round(self.input_sample_rate * self.block_ms / 1000.0)))
        hop_samples = max(1, int(round(self.input_sample_rate * self.hop_ms / 1000.0)))
        generated = 0
        while self.next_block_start + block_samples <= self.input_seen_samples:
            start = self.next_block_start
            end = start + block_samples
            block = np.ascontiguousarray(self.input_history[start:end], dtype=np.float32)
            enhanced = self._run_backend(backend, block)
            out_start = int(round(start * self.output_sample_rate / self.input_sample_rate))
            self._add_overlap_block(enhanced, out_start, is_first=start == 0)
            self.next_block_start += hop_samples
            generated += 1
        return generated

    def _fallback_chunk(self, mono: np.ndarray, target_samples: int, chunk_sample_rate: int) -> np.ndarray:
        if self.startup_fallback != "resample":
            return np.zeros(target_samples, dtype=np.float32)
        fallback = _resample_np(mono, chunk_sample_rate, self.output_sample_rate)
        if self.fallback_gain_db:
            fallback = fallback * float(10.0 ** (self.fallback_gain_db / 20.0))
        if fallback.shape[0] < target_samples:
            fallback = np.pad(fallback, (0, target_samples - fallback.shape[0]))
        return np.ascontiguousarray(fallback[:target_samples], dtype=np.float32)

    def _consume_overlap_output(self, target_samples: int, fallback: np.ndarray) -> tuple[np.ndarray, int]:
        start = self.read_output_pos
        end = start + target_samples
        if end <= self.output_accum.shape[0]:
            weight = self.output_weight[start:end]
            if bool(np.all(weight > 1e-3)):
                out = self.output_accum[start:end] / np.maximum(weight, 1e-3)
                self.read_output_pos = end
                return np.ascontiguousarray(out, dtype=np.float32), 0
        if fallback.shape[0] < target_samples:
            fallback = np.pad(fallback, (0, target_samples - fallback.shape[0]))
        return np.ascontiguousarray(fallback[:target_samples], dtype=np.float32), target_samples

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not self.enabled:
            return super().process(chunk)
        backend = self._ensure_backend()
        mono = self._to_mono_numpy(chunk.samples)
        model_input = _resample_np(mono, chunk.sample_rate, self.input_sample_rate)
        generated_blocks = 0
        fallback_samples = 0
        if self.processing_mode == "overlap":
            previous_seen = self.input_seen_samples
            self.input_history = np.concatenate([self.input_history, model_input])
            self.input_seen_samples += int(model_input.shape[0])
            generated_blocks = self._generate_overlap_blocks(backend)
            target_end = int(round(self.input_seen_samples * self.output_sample_rate / self.input_sample_rate))
            target_start = int(round(previous_seen * self.output_sample_rate / self.input_sample_rate))
            target_samples = max(0, target_end - target_start)
            fallback = self._fallback_chunk(mono, target_samples, chunk.sample_rate)
            enhanced, fallback_samples = self._consume_overlap_output(target_samples, fallback)
        else:
            enhanced = self._run_backend(backend, model_input)
        out = torch.from_numpy(enhanced.astype(np.float32, copy=False)).to(
            device=chunk.samples.device,
            dtype=chunk.samples.dtype,
        )
        metadata = merge_metadata(
            chunk,
            bandwidth_extension_applied=True,
            flashsr_variant=backend.variant,
            flashsr_output_sample_rate=self.output_sample_rate,
        )
        return ProcessResult(
            chunk=AudioChunk(
                out.unsqueeze(0).contiguous(),
                self.output_sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics={
                "flashsr_device": backend.device,
                "flashsr_provider": backend.provider,
                "flashsr_variant": backend.variant,
                "flashsr_input_sample_rate": self.input_sample_rate,
                "flashsr_output_sample_rate": self.output_sample_rate,
                "flashsr_model_input_samples": int(model_input.shape[0]),
                "flashsr_output_samples": int(enhanced.shape[0]),
                "flashsr_processing_mode": self.processing_mode,
                "flashsr_block_ms": self.block_ms if self.processing_mode == "overlap" else 0.0,
                "flashsr_hop_ms": self.hop_ms if self.processing_mode == "overlap" else 0.0,
                "flashsr_generated_blocks": generated_blocks,
                "flashsr_fallback_samples": fallback_samples,
                "realtime_safe": True,
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
