from __future__ import annotations

from typing import Any

import torch


def resolve_torch_device(requested: Any = "auto") -> tuple[str, str]:
    """Return requested and effective torch device labels.

    `auto` means CUDA when available, otherwise CPU. Explicit CUDA requests fall
    back to CPU if CUDA is not available so optional eval acceleration never
    turns into an import-time crash on CPU-only machines.
    """

    requested_label = str(requested or "auto").strip().lower()
    if requested_label in {"", "none"}:
        requested_label = "auto"
    if requested_label == "auto":
        effective = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif requested_label.startswith("cuda") and not torch.cuda.is_available():
        effective = "cpu"
    else:
        effective = requested_label
    return requested_label, effective


def is_cuda_device(device: str | None) -> bool:
    return bool(device and str(device).startswith("cuda"))
