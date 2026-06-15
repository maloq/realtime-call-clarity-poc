from __future__ import annotations

import numpy as np
import torch


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-12)


def rms_dbfs(x: torch.Tensor) -> float:
    value = float(rms(x).mean().item())
    return 20.0 * float(np.log10(max(value, 1e-12)))


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def linear_to_db(value: float) -> float:
    return 20.0 * float(np.log10(max(abs(value), 1e-12)))


def smooth_1d(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or window <= 1:
        return arr
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="same")
