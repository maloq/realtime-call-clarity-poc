from __future__ import annotations

from pathlib import Path


def cache_path_for(cache_dir: str | Path, source: str | Path, suffix: str = ".wav") -> Path:
    src = Path(source)
    safe = "_".join(src.parts[-4:]).replace("/", "_").replace(".", "_")
    return Path(cache_dir) / f"{safe}{suffix}"
