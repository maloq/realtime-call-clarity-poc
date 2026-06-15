from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, MethodUnavailable, ProcessResult
from callclarity.utils.device import resolve_torch_device


_BASE_URL = "https://huggingface.co/rsxdalv/AP-BWE/resolve/main/weights"


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 3)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class _APNetBweModel(nn.Module):
    def __init__(self, h: _AttrDict) -> None:
        super().__init__()
        layer_scale = 1.0 / float(h.ConvNeXt_layers)
        bins = int(h.n_fft) // 2 + 1
        channels = int(h.ConvNeXt_channels)
        layers = int(h.ConvNeXt_layers)
        self.conv_pre_mag = nn.Conv1d(bins, channels, 7, 1, padding=3)
        self.norm_pre_mag = nn.LayerNorm(channels, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(bins, channels, 7, 1, padding=3)
        self.norm_pre_pha = nn.LayerNorm(channels, eps=1e-6)
        self.convnext_mag = nn.ModuleList(
            [_ConvNeXtBlock(channels, layer_scale) for _ in range(layers)]
        )
        self.convnext_pha = nn.ModuleList(
            [_ConvNeXtBlock(channels, layer_scale) for _ in range(layers)]
        )
        self.norm_post_mag = nn.LayerNorm(channels, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(channels, eps=1e-6)
        self.linear_post_mag = nn.Linear(channels, bins)
        self.linear_post_pha_r = nn.Linear(channels, bins)
        self.linear_post_pha_i = nn.Linear(channels, bins)

    def forward(
        self,
        mag_nb: torch.Tensor,
        pha_nb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mag = self.conv_pre_mag(mag_nb)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag)
            x_pha = conv_block_pha(x_pha)

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        mag_wb = mag_nb + self.linear_post_mag(x_mag).transpose(1, 2)
        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)
        com_wb = torch.stack(
            (torch.exp(mag_wb) * torch.cos(pha_wb), torch.exp(mag_wb) * torch.sin(pha_wb)),
            dim=-1,
        )
        return mag_wb, pha_wb, com_wb


def _amp_pha_stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(win_size, device=audio.device)
    stft_spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    log_amp = torch.log(torch.abs(stft_spec) + 1e-4)
    pha = torch.angle(stft_spec)
    return log_amp, pha


def _amp_pha_istft(
    log_amp: torch.Tensor,
    pha: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    length: int,
) -> torch.Tensor:
    amp = torch.exp(log_amp)
    com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
    window = torch.hann_window(win_size, device=com.device)
    return torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        length=length,
    )


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


def _load_torch_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class _ApBweBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.task = str(config.get("task", "8kto16k"))
        cache_dir = Path(str(config.get("cache_dir", "~/.cache/callclarity/ap_bwe"))).expanduser()
        task_dir = cache_dir / self.task
        self.config_path = Path(str(config.get("config_path") or task_dir / "config.json"))
        self.checkpoint_path = Path(
            str(config.get("checkpoint_path") or task_dir / f"g_{self.task}.ckpt")
        )
        if bool(config.get("auto_download", True)):
            self._ensure_downloaded()
        if not self.config_path.exists() or not self.checkpoint_path.exists():
            raise MethodUnavailable(
                "AP-BWE checkpoint is missing. Set bandwidth config_path/checkpoint_path "
                "or enable auto_download."
            )
        self.h = _AttrDict(json.loads(self.config_path.read_text(encoding="utf-8")))
        self.model_sample_rate = int(self.h.hr_sampling_rate)
        self.low_sample_rate = int(self.h.lr_sampling_rate)
        self.device_requested, self.device = resolve_torch_device(config.get("device", "auto"))
        self.model = _APNetBweModel(self.h).to(torch.device(self.device))
        state = _load_torch_checkpoint(self.checkpoint_path, self.device)
        self.model.load_state_dict(state.get("generator", state))
        self.model.eval()

    def _ensure_downloaded(self) -> None:
        base = f"{_BASE_URL}/{self.task}"
        if not self.config_path.exists():
            _download_file(f"{base}/config.json", self.config_path)
        if not self.checkpoint_path.exists():
            _download_file(f"{base}/g_{self.task}.ckpt", self.checkpoint_path)

    def process_window(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        if int(sample_rate) != self.model_sample_rate:
            raise ValueError(
                f"AP-BWE task {self.task} expects {self.model_sample_rate} Hz audio, "
                f"got {sample_rate} Hz."
            )
        target_len = int(window.shape[0])
        low = signal.resample_poly(window, self.low_sample_rate, self.model_sample_rate)
        low = signal.resample_poly(low, self.model_sample_rate, self.low_sample_rate)
        if low.shape[0] < target_len:
            low = np.pad(low, (0, target_len - low.shape[0]))
        low = np.ascontiguousarray(low[:target_len], dtype=np.float32)
        audio_lr = torch.from_numpy(low).float().unsqueeze(0).to(torch.device(self.device))
        with torch.no_grad():
            amp_nb, pha_nb = _amp_pha_stft(
                audio_lr,
                int(self.h.n_fft),
                int(self.h.hop_size),
                int(self.h.win_size),
            )
            amp_wb, pha_wb, _ = self.model(amp_nb, pha_nb)
            audio = _amp_pha_istft(
                amp_wb,
                pha_wb,
                int(self.h.n_fft),
                int(self.h.hop_size),
                int(self.h.win_size),
                target_len,
            )
        out = audio.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        if out.shape[0] < target_len:
            out = np.pad(out, (0, target_len - out.shape[0]))
        return np.ascontiguousarray(out[:target_len], dtype=np.float32)


@register_method("bandwidth", "ap_bwe_block")
class ApBweBlockProcessor(BaseStreamingProcessor):
    name = "ap_bwe_block"
    realtime_safe = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.hop_ms = float(self.config.get("hop_ms", 100.0))
        self.left_context_ms = float(self.config.get("left_context_ms", 250.0))
        self.right_context_ms = float(self.config.get("right_context_ms", 250.0))
        self.blend = float(self.config.get("blend", 1.0))
        self.backend: _ApBweBackend | None = None
        self.sample_rate: int | None = None
        self.hop_samples = 0
        self.left_context_samples = 0
        self.right_context_samples = 0
        self.window_samples = 0
        self.input_history = np.zeros(0, dtype=np.float32)
        self.output_queue = np.zeros(0, dtype=np.float32)
        self.input_seen_samples = 0
        self.next_output_start = 0

    @property
    def algorithmic_latency_ms(self) -> float:
        if not self.enabled:
            return 0.0
        return float(self.config.get("algorithmic_latency_ms", self.right_context_ms + self.hop_ms))

    def reset(self) -> None:
        self.input_history = np.zeros(0, dtype=np.float32)
        self.output_queue = np.zeros(0, dtype=np.float32)
        self.input_seen_samples = 0
        self.next_output_start = 0

    def warmup(self, sample_rate: int) -> None:
        if self.enabled:
            self._ensure_backend()
            self._configure_rate(sample_rate)

    def _ensure_backend(self) -> _ApBweBackend:
        if self.backend is None:
            self.backend = _ApBweBackend(self.config)
        return self.backend

    def _configure_rate(self, sample_rate: int) -> None:
        if self.sample_rate == int(sample_rate):
            return
        self.sample_rate = int(sample_rate)
        self.hop_samples = max(1, int(round(self.sample_rate * self.hop_ms / 1000.0)))
        self.left_context_samples = max(
            0,
            int(round(self.sample_rate * self.left_context_ms / 1000.0)),
        )
        self.right_context_samples = max(
            0,
            int(round(self.sample_rate * self.right_context_ms / 1000.0)),
        )
        self.window_samples = self.left_context_samples + self.hop_samples + self.right_context_samples
        self.reset()

    def _to_mono_numpy(self, samples: torch.Tensor) -> np.ndarray:
        x = samples.detach().cpu().float()
        if x.ndim == 1:
            mono = x
        elif x.shape[0] == 1:
            mono = x[0]
        else:
            mono = x.mean(dim=0)
        return np.ascontiguousarray(mono.numpy(), dtype=np.float32)

    def _slice_with_padding(self, start: int, end: int) -> np.ndarray:
        left_pad = max(0, -start)
        right_pad = max(0, end - self.input_history.shape[0])
        src_start = max(0, start)
        src_end = min(end, self.input_history.shape[0])
        parts = []
        if left_pad:
            parts.append(np.zeros(left_pad, dtype=np.float32))
        if src_end > src_start:
            parts.append(self.input_history[src_start:src_end])
        if right_pad:
            parts.append(np.zeros(right_pad, dtype=np.float32))
        if not parts:
            return np.zeros(end - start, dtype=np.float32)
        out = np.concatenate(parts).astype(np.float32, copy=False)
        if out.shape[0] != end - start:
            out = np.pad(out, (0, max(0, end - start - out.shape[0])))[: end - start]
        return np.ascontiguousarray(out, dtype=np.float32)

    def _generate_available_blocks(self, sample_rate: int) -> int:
        backend = self._ensure_backend()
        generated = 0
        ready_until = self.input_seen_samples
        while (
            self.next_output_start + self.hop_samples + self.right_context_samples
            <= ready_until
        ):
            output_start = self.next_output_start
            window_start = output_start - self.left_context_samples
            window_end = output_start + self.hop_samples + self.right_context_samples
            window = self._slice_with_padding(window_start, window_end)
            enhanced = backend.process_window(window, sample_rate)
            segment = enhanced[
                self.left_context_samples : self.left_context_samples + self.hop_samples
            ]
            if self.blend < 1.0:
                dry = self._slice_with_padding(output_start, output_start + self.hop_samples)
                segment = dry * (1.0 - self.blend) + segment * self.blend
            self.output_queue = np.concatenate([self.output_queue, segment.astype(np.float32)])
            self.next_output_start += self.hop_samples
            generated += 1
        return generated

    def _consume_output(self, samples: int) -> np.ndarray:
        if self.output_queue.shape[0] >= samples:
            out = self.output_queue[:samples]
            self.output_queue = self.output_queue[samples:].astype(np.float32, copy=False)
            return out.astype(np.float32, copy=False)
        out = np.zeros(samples, dtype=np.float32)
        if self.output_queue.size:
            out[: self.output_queue.shape[0]] = self.output_queue
        self.output_queue = np.zeros(0, dtype=np.float32)
        return out

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not self.enabled:
            return super().process(chunk)
        self._configure_rate(chunk.sample_rate)
        mono = self._to_mono_numpy(chunk.samples)
        self.input_history = np.concatenate([self.input_history, mono])
        self.input_seen_samples += int(mono.shape[0])
        generated = self._generate_available_blocks(chunk.sample_rate)
        out_np = self._consume_output(chunk.num_samples)
        out = torch.from_numpy(out_np).to(device=chunk.samples.device).unsqueeze(0).contiguous()
        backend = self._ensure_backend()
        metadata = merge_metadata(chunk, bandwidth_extension_applied=True, ap_bwe_block=True)
        return ProcessResult(
            chunk=AudioChunk(out, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "ap_bwe_device_requested": backend.device_requested,
                "ap_bwe_device": backend.device,
                "ap_bwe_task": backend.task,
                "ap_bwe_generated_blocks": generated,
                "ap_bwe_hop_ms": self.hop_ms,
                "ap_bwe_left_context_ms": self.left_context_ms,
                "ap_bwe_right_context_ms": self.right_context_ms,
                "ap_bwe_output_queue_samples": int(self.output_queue.shape[0]),
                "ap_bwe_next_output_start_samples": int(self.next_output_start),
                "ap_bwe_input_seen_samples": int(self.input_seen_samples),
            },
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
