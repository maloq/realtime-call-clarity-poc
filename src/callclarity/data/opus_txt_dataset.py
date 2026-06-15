from __future__ import annotations

from pathlib import Path
from typing import Any

from callclarity.data.base import AudioDataset
from callclarity.data.manifests import build_manifest_from_config, read_manifest
from callclarity.io.audio_io import load_audio
from callclarity.types import DatasetItem


class OpusTxtDataset(AudioDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        sample_rate: int = 16000,
        channels: str = "mono",
        load_waveforms: bool = False,
    ) -> None:
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.load_waveforms = load_waveforms

    @classmethod
    def from_config(cls, data_cfg: Any, sample_rate: int = 16000, channels: str = "mono") -> "OpusTxtDataset":
        if data_cfg.manifest_path:
            rows = read_manifest(data_cfg.manifest_path)
        else:
            rows = build_manifest_from_config(data_cfg)
        return cls(rows, sample_rate=sample_rate, channels=channels)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.rows[idx]
        waveform = None
        sample_rate = None
        if self.load_waveforms:
            waveform, sample_rate = load_audio(
                row["audio_path"], target_sample_rate=self.sample_rate, channels=self.channels
            )
        return DatasetItem(
            recording_id=row["recording_id"],
            audio_path=Path(row["audio_path"]),
            transcript_path=Path(row["transcript_path"]) if row.get("transcript_path") else None,
            transcript=row.get("transcript", ""),
            sample_rate=sample_rate,
            waveform=waveform,
            clean_reference_path=Path(row["clean_reference_path"]) if row.get("clean_reference_path") else None,
            speaker_id=row.get("speaker_id"),
            language=row.get("language"),
            metadata=dict(row.get("metadata") or {}),
        )
