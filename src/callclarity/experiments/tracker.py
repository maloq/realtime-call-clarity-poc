from __future__ import annotations

from pathlib import Path
from typing import Any


class Tracker:
    def log_params(self, params: dict[str, Any]) -> None:
        del params

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        del metrics, step

    def log_artifact(self, path: str | Path) -> None:
        del path

    def close(self) -> None:
        pass


def create_tracker(config: dict[str, Any] | None = None) -> Tracker:
    cfg = config or {}
    backend = cfg.get("backend", "none")
    if backend != "none":
        raise RuntimeError(f"Tracking backend `{backend}` is optional and not configured in this POC runtime.")
    return Tracker()
