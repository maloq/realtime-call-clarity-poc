from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.io import wavfile

from callclarity.types import MethodUnavailable


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    if array.dtype.kind in {"i", "u"}:
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), abs(info.max))
        data = array.astype(np.float32) / float(scale)
    else:
        data = array.astype(np.float32)
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data).transpose(0, 1)
    return tensor.clamp(-1.0, 1.0).contiguous()


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_if_needed(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if int(source_sr) == int(target_sr):
        return waveform
    try:
        import torchaudio.functional as F

        return F.resample(waveform, int(source_sr), int(target_sr))
    except Exception:
        from scipy.signal import resample_poly

        import math

        gcd = math.gcd(int(source_sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(source_sr) // gcd
        data = resample_poly(waveform.detach().cpu().numpy(), up, down, axis=-1).astype(np.float32)
        return torch.from_numpy(data)


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16000,
    channels: Literal["mono", "native"] = "mono",
) -> tuple[torch.Tensor, int]:
    p = Path(path)
    errors: list[str] = []

    if p.suffix.lower() == ".wav":
        try:
            sr, data = wavfile.read(p)
            wav = _as_float_tensor(data)
            if channels == "mono":
                wav = _to_mono(wav)
            wav = resample_if_needed(wav, sr, target_sample_rate)
            return wav, target_sample_rate
        except Exception as exc:  # pragma: no cover - exercised by backend matrix
            errors.append(f"scipy wavfile: {exc}")

    try:
        import torchaudio

        wav, sr = torchaudio.load(str(p))
        wav = wav.float().clamp(-1.0, 1.0)
        if channels == "mono":
            wav = _to_mono(wav)
        wav = resample_if_needed(wav, int(sr), target_sample_rate)
        return wav.contiguous(), target_sample_rate
    except Exception as exc:
        errors.append(f"torchaudio: {exc}")

    try:
        import av

        frames = []
        source_sr = None
        with av.open(str(p)) as container:
            for frame in container.decode(audio=0):
                source_sr = int(frame.sample_rate)
                arr = frame.to_ndarray().astype(np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                frames.append(arr)
        if frames and source_sr is not None:
            wav = torch.from_numpy(np.concatenate(frames, axis=-1))
            if channels == "mono":
                wav = _to_mono(wav)
            wav = resample_if_needed(wav, source_sr, target_sample_rate)
            return wav.contiguous(), target_sample_rate
    except Exception as exc:
        errors.append(f"PyAV: {exc}")

    try:
        import soundfile as sf

        data, sr = sf.read(str(p), always_2d=True, dtype="float32")
        wav = torch.from_numpy(data.T)
        if channels == "mono":
            wav = _to_mono(wav)
        wav = resample_if_needed(wav, int(sr), target_sample_rate)
        return wav.contiguous(), target_sample_rate
    except Exception as exc:
        errors.append(f"soundfile: {exc}")

    if shutil.which("ffmpeg"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav_path = Path(tmp) / "decoded.wav"
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(p),
                    "-ac",
                    "1" if channels == "mono" else "2",
                    "-ar",
                    str(target_sample_rate),
                    str(wav_path),
                ]
                subprocess.run(cmd, check=True)
                sr, data = wavfile.read(wav_path)
                wav = _as_float_tensor(data)
                if channels == "mono":
                    wav = _to_mono(wav)
                return wav, int(sr)
        except Exception as exc:
            errors.append(f"ffmpeg: {exc}")
    else:
        errors.append("ffmpeg: executable not found")

    raise MethodUnavailable(
        "Could not decode audio. Install one of: torchaudio with codec support, soundfile, "
        "PyAV, or ffmpeg. Backend errors: " + " | ".join(errors)
    )


def write_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().cpu()
    if data.ndim == 2:
        data = data.transpose(0, 1)
    data_np = data.numpy()
    data_i16 = np.clip(data_np, -1.0, 1.0)
    data_i16 = (data_i16 * 32767.0).astype(np.int16)
    wavfile.write(p, int(sample_rate), data_i16)


def peak_dbfs(waveform: torch.Tensor) -> float:
    peak = float(waveform.detach().abs().max().item()) if waveform.numel() else 0.0
    return 20.0 * float(np.log10(max(peak, 1e-12)))


def rms_dbfs(waveform: torch.Tensor) -> float:
    rms = float(torch.sqrt(torch.mean(waveform.detach().float() ** 2) + 1e-12).item())
    return 20.0 * float(np.log10(max(rms, 1e-12)))
