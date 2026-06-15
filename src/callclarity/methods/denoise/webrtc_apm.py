from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

import numpy as np
import torch

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult


_LEVELS = {
    "very_low": 0,
    "low": 0,
    "moderate": 1,
    "medium": 1,
    "high": 2,
    "very_high": 3,
}

_AGC_TYPES = {
    "disabled": 0,
    "off": 0,
    "none": 0,
    "adaptive_digital": 1,
    "digital": 1,
    "adaptive_analog": 2,
    "analog": 2,
}

_AEC_TYPES = {
    "disabled": 0,
    "off": 0,
    "none": 0,
    "mobile": 1,
    "standard": 2,
    "desktop": 2,
}


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _level_value(value: Any, default: str = "moderate") -> int:
    if isinstance(value, int):
        return max(0, min(3, value))
    return _LEVELS.get(str(value or default).strip().lower(), _LEVELS[default])


def _agc_type(config: dict[str, Any]) -> int:
    value = str(config.get("mode", config.get("type", "adaptive_digital"))).strip().lower()
    return _AGC_TYPES.get(value, 1)


def _aec_type(config: dict[str, Any]) -> int:
    value = config.get("type", config.get("mode", "standard"))
    if isinstance(value, int):
        return max(0, min(2, value))
    return _AEC_TYPES.get(str(value).strip().lower(), 2)


def _pcm16_to_float32(data: bytes | str, target_samples: int) -> np.ndarray:
    if isinstance(data, str):
        data = data.encode("latin1")
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if pcm.shape[0] == target_samples:
        return pcm
    out = np.zeros(target_samples, dtype=np.float32)
    out[: min(target_samples, pcm.shape[0])] = pcm[:target_samples]
    return out


def _float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes()


def _install_legacy_imp_shim() -> None:
    if "imp" in sys.modules:
        return
    try:
        importlib.import_module("imp")
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("imp")
    shim.C_EXTENSION = 3

    def find_module(name: str, path: list[str] | None = None) -> tuple[Any, str, tuple[str, str, int]]:
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.origin is None:
            raise ImportError(f"No module named {name}")
        suffix = next(
            (
                candidate
                for candidate in importlib.machinery.EXTENSION_SUFFIXES
                if spec.origin.endswith(candidate)
            ),
            "",
        )
        return open(spec.origin, "rb"), spec.origin, (suffix, "rb", shim.C_EXTENSION)

    def load_module(name: str, file: Any, pathname: str, description: Any) -> Any:
        del file, description
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, pathname)
        if spec is None or spec.loader is None:
            raise ImportError(f"No module named {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    shim.find_module = find_module
    shim.load_module = load_module
    sys.modules["imp"] = shim


@register_method("denoise", "webrtc_apm")
class WebrtcApmDenoiser(BaseStreamingProcessor):
    name = "webrtc_apm"
    realtime_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.module_name = str(self.config.get("python_module", "webrtc_audio_processing"))
        self.frame_ms = int(self.config.get("frame_ms", 10))
        if self.frame_ms != 10:
            raise ValueError("WebRTC APM only supports 10 ms frames in this Python binding.")

        ns_cfg = _section(self.config, "noise_suppression")
        agc_cfg = _section(self.config, "digital_agc")
        aec_cfg = _section(self.config, "echo_cancellation")
        vad_cfg = _section(self.config, "vad")

        self.ns_enabled = bool(ns_cfg.get("enabled", True))
        self.ns_level = _level_value(ns_cfg.get("level", "moderate"))
        self.agc_enabled = bool(agc_cfg.get("enabled", True))
        self.agc_type = _agc_type(agc_cfg) if self.agc_enabled else 0
        self.agc_target_dbfs = int(abs(float(agc_cfg.get("target_level_dbfs", -18))))
        self.agc_initial_level = agc_cfg.get("initial_level")
        self.aec_enabled = bool(aec_cfg.get("enabled", False))
        self.aec_type = _aec_type(aec_cfg) if self.aec_enabled else 0
        self.aec_level = _level_value(aec_cfg.get("level", 0), default="low")
        self.vad_enabled = bool(vad_cfg.get("enabled", False))
        self.vad_level = _level_value(vad_cfg.get("level", "low"), default="low")

        self._apm_cls: Any | None = None
        self._apm: Any | None = None
        self._sample_rate: int | None = None
        self._frame_samples = 0
        self._input_buffer = np.zeros(0, dtype=np.float32)
        self._pending_output = np.zeros(0, dtype=np.float32)

        if not self.enabled:
            return

        try:
            _install_legacy_imp_shim()
            module = importlib.import_module(self.module_name)
            self._apm_cls = module.AudioProcessingModule
        except Exception as exc:
            raise MethodUnavailable(
                "WebRTC APM is optional. Install the native Python binding with "
                "`pip install webrtc-audio-processing` or `pip install -e \".[denoise]\"`. "
                "The binding requires a compiler toolchain and SWIG on some platforms."
            ) from exc

    @property
    def algorithmic_latency_ms(self) -> float:
        if not self.enabled:
            return 0.0
        return float(self.config.get("algorithmic_latency_ms", self.frame_ms))

    def reset(self) -> None:
        self._input_buffer = np.zeros(0, dtype=np.float32)
        self._pending_output = np.zeros(0, dtype=np.float32)
        self._apm = None
        sample_rate = self._sample_rate
        self._sample_rate = None
        if sample_rate is not None:
            self._ensure_apm(sample_rate)

    def warmup(self, sample_rate: int) -> None:
        if self.enabled:
            self._ensure_apm(sample_rate)

    def _ensure_apm(self, sample_rate: int) -> None:
        if self._apm is not None and self._sample_rate == sample_rate:
            return
        if self._apm_cls is None:
            raise MethodUnavailable(
                "WebRTC APM backend is unavailable. Install `webrtc-audio-processing`."
            )

        self._input_buffer = np.zeros(0, dtype=np.float32)
        self._pending_output = np.zeros(0, dtype=np.float32)
        self._sample_rate = int(sample_rate)
        self._frame_samples = max(1, int(round(self._sample_rate * self.frame_ms / 1000.0)))
        self._apm = self._apm_cls(
            aec_type=self.aec_type,
            enable_ns=self.ns_enabled,
            agc_type=self.agc_type,
            enable_vad=self.vad_enabled,
        )
        self._apm.set_stream_format(self._sample_rate, 1, self._sample_rate, 1)
        if self.ns_enabled and hasattr(self._apm, "set_ns_level"):
            self._apm.set_ns_level(self.ns_level)
        if self.agc_enabled and hasattr(self._apm, "set_agc_target"):
            self._apm.set_agc_target(self.agc_target_dbfs)
        if self.agc_initial_level is not None and hasattr(self._apm, "set_agc_level"):
            self._apm.set_agc_level(int(self.agc_initial_level))
        if self.aec_enabled and hasattr(self._apm, "set_aec_level"):
            self._apm.set_aec_level(self.aec_level)
        if self.vad_enabled and hasattr(self._apm, "set_vad_level"):
            self._apm.set_vad_level(self.vad_level)

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

    def _process_available_frames(self) -> np.ndarray:
        assert self._apm is not None
        frames: list[np.ndarray] = []
        while self._input_buffer.shape[0] >= self._frame_samples:
            frame = self._input_buffer[: self._frame_samples]
            self._input_buffer = self._input_buffer[self._frame_samples :]
            payload = _float32_to_pcm16(frame)
            try:
                processed = self._apm.process_stream(payload)
            except TypeError:
                processed = self._apm.process_stream(payload.decode("latin1"))
            frames.append(_pcm16_to_float32(processed, self._frame_samples))
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32, copy=False)

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not self.enabled:
            return super().process(chunk)

        self._ensure_apm(chunk.sample_rate)
        mono = self._to_mono_numpy(chunk.samples)
        if self._input_buffer.size:
            self._input_buffer = np.concatenate([self._input_buffer, mono])
        else:
            self._input_buffer = mono
        enhanced = self._process_available_frames()
        out_np = self._consume_exact_output(enhanced, chunk.num_samples)
        out = torch.from_numpy(out_np).to(device=chunk.samples.device).unsqueeze(0).contiguous()

        voice_detected = None
        if self.vad_enabled and self._apm is not None and hasattr(self._apm, "has_voice"):
            voice_detected = bool(self._apm.has_voice())
        agc_level = None
        if self.agc_enabled and self._apm is not None and hasattr(self._apm, "agc_level"):
            agc_level = int(self._apm.agc_level())

        metadata = merge_metadata(
            chunk,
            denoised=True,
            webrtc_apm=True,
            webrtc_apm_noise_suppression=self.ns_enabled,
            webrtc_apm_digital_agc=self.agc_enabled,
        )
        return ProcessResult(
            chunk=AudioChunk(
                out,
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics={
                "realtime_safe": True,
                "webrtc_apm": True,
                "webrtc_apm_module": self.module_name,
                "webrtc_apm_sample_rate": self._sample_rate,
                "webrtc_apm_frame_ms": self.frame_ms,
                "webrtc_apm_frame_samples": self._frame_samples,
                "webrtc_apm_input_buffer_samples": int(self._input_buffer.shape[0]),
                "webrtc_apm_pending_output_samples": int(self._pending_output.shape[0]),
                "webrtc_apm_ns_enabled": self.ns_enabled,
                "webrtc_apm_ns_level": self.ns_level,
                "webrtc_apm_agc_enabled": self.agc_enabled,
                "webrtc_apm_agc_type": self.agc_type,
                "webrtc_apm_agc_target_dbfs": self.agc_target_dbfs,
                "webrtc_apm_agc_level": agc_level,
                "webrtc_apm_aec_enabled": self.aec_enabled,
                "webrtc_apm_vad_enabled": self.vad_enabled,
                "webrtc_apm_voice_detected": voice_detected,
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
