from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, p)
