from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def repo_relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
