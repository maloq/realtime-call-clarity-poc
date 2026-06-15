from __future__ import annotations

from pathlib import Path
from typing import Any

from callclarity.io.audio_io import write_wav
from callclarity.utils.files import repo_relative_path, write_json


def export_sample(
    output_dir: str | Path,
    recording_id: str,
    raw,
    processed,
    sample_rate: int,
    metadata: dict[str, Any],
    processed_sample_rate: int | None = None,
) -> dict[str, str]:
    sample_dir = Path(output_dir) / recording_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_path = sample_dir / "raw.wav"
    processed_path = sample_dir / "processed.wav"
    info_path = sample_dir / "comparison_info.json"
    write_wav(raw_path, raw, sample_rate)
    write_wav(processed_path, processed, processed_sample_rate or sample_rate)
    write_json(info_path, metadata)
    return {
        "recording_id": recording_id,
        "raw_path": repo_relative_path(raw_path),
        "processed_path": repo_relative_path(processed_path),
    }
