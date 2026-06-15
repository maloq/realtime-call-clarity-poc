from __future__ import annotations

import json
import random
import struct
import tarfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

from callclarity.utils.files import ensure_dir, repo_relative_path, write_jsonl


def _maybe_extract_archive(input_path: Path, cache_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    suffixes = "".join(input_path.suffixes).lower()
    if not (input_path.is_file() and suffixes.endswith((".tar.gz", ".tgz", ".tar"))):
        raise FileNotFoundError(f"Input path does not exist or is not a supported archive: {input_path}")
    extract_root = cache_dir / "extracted" / input_path.name.replace(".tar.gz", "").replace(".tgz", "")
    marker = extract_root / ".complete"
    if marker.exists():
        return extract_root
    ensure_dir(extract_root)
    with tarfile.open(input_path) as archive:
        archive.extractall(extract_root)
    marker.write_text("ok\n")
    return extract_root


def _safe_extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, root: Path) -> Path:
    dest = (root / member.name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in dest.parents and dest != root_resolved:
        raise RuntimeError(f"Refusing to extract unsafe archive path: {member.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == member.size:
        return dest
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"Could not extract archive member: {member.name}")
    with source, dest.open("wb") as handle:
        handle.write(source.read())
    return dest


def _write_archive_member_data(member: tarfile.TarInfo, data: bytes, root: Path) -> Path:
    dest = (root / member.name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in dest.parents and dest != root_resolved:
        raise RuntimeError(f"Refusing to extract unsafe archive path: {member.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size == len(data)):
        dest.write_bytes(data)
    return dest


def _opus_duration_sec(data: bytes) -> float | None:
    preskip = 0
    head = data.find(b"OpusHead")
    if head >= 0 and len(data) >= head + 12:
        preskip = struct.unpack_from("<H", data, head + 10)[0]
    pos = 0
    last_granule = None
    while True:
        pos = data.find(b"OggS", pos)
        if pos < 0 or len(data) < pos + 27:
            break
        granule = struct.unpack_from("<q", data, pos + 6)[0]
        page_segments = data[pos + 26]
        segment_table_start = pos + 27
        segment_table_end = segment_table_start + page_segments
        if segment_table_end > len(data):
            break
        if granule >= 0:
            last_granule = granule
        pos = segment_table_end + sum(data[segment_table_start:segment_table_end])
    if last_granule is None:
        return None
    return max(0.0, (last_granule - preskip) / 48000.0)


def _archive_audio_duration_sec(path: Path, data: bytes) -> float | None:
    if path.suffix.lower() == ".opus":
        return _opus_duration_sec(data)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(BytesIO(data), "rb") as handle:
                sample_rate = handle.getframerate()
                if sample_rate <= 0:
                    return None
                return handle.getnframes() / float(sample_rate)
        except (wave.Error, EOFError):
            return None
    return None


def _archive_cache_name(input_path: Path) -> str:
    return input_path.name.replace(".tar.gz", "").replace(".tgz", "").replace(".tar", "")


def _recording_id(stem: str, prefix: str | None = None) -> str:
    return f"{prefix}_{stem}" if prefix else stem


def _directory_audio_sort_key(root: Path, audio_path: Path) -> tuple[int, str]:
    relative = audio_path.relative_to(root)
    parts = [part.lower() for part in relative.parts]
    priority = 0 if "raw" in parts[:-1] else 1
    return priority, relative.as_posix().lower()


def _discover_limited_archive_pairs(
    input_path: Path,
    extensions: list[str],
    transcript_extension: str,
    cache_dir: Path,
    max_files: int,
    min_duration_sec: float | None = None,
    selection: str = "first",
    recording_id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    selection = selection.lower()
    if selection not in {"first", "longest"}:
        raise ValueError(f"Unsupported archive selection mode: {selection}")
    extract_root = cache_dir / "extracted_subset" / _archive_cache_name(input_path)
    rows: list[dict[str, Any]] = []
    with tarfile.open(input_path) as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        by_name = {m.name: m for m in members}
        audio_members = [m for m in members if Path(m.name).suffix.lower() in extensions]
        seen_audio_names: set[str] = set()
        selected: list[tuple[tarfile.TarInfo, bytes, float | None]] = []
        for audio_member in audio_members:
            if selection == "first" and len(selected) >= max_files:
                break
            if audio_member.name in seen_audio_names:
                continue
            seen_audio_names.add(audio_member.name)
            audio_file = archive.extractfile(audio_member)
            if audio_file is None:
                continue
            with audio_file:
                audio_data = audio_file.read()
            duration_sec = _archive_audio_duration_sec(Path(audio_member.name), audio_data)
            if min_duration_sec is not None:
                if duration_sec is None or duration_sec <= min_duration_sec:
                    continue
            selected.append((audio_member, audio_data, duration_sec))
            if selection == "longest":
                selected = sorted(
                    selected,
                    key=lambda item: (-(item[2] if item[2] is not None else -1.0), item[0].name),
                )[:max_files]
        if selection == "first":
            selected = selected[:max_files]
        else:
            selected = sorted(
                selected,
                key=lambda item: (-(item[2] if item[2] is not None else -1.0), item[0].name),
            )
        for audio_member, audio_data, duration_sec in selected:
            transcript_name = str(Path(audio_member.name).with_suffix(transcript_extension))
            transcript_member = by_name.get(transcript_name)
            audio_path = _write_archive_member_data(audio_member, audio_data, extract_root)
            transcript_path = None
            transcript = ""
            if transcript_member is not None:
                transcript_path = _safe_extract_member(archive, transcript_member, extract_root)
                transcript = transcript_path.read_text(encoding="utf-8").strip()
            rows.append(
                {
                    "recording_id": _recording_id(Path(audio_member.name).stem, recording_id_prefix),
                    "audio_path": repo_relative_path(audio_path),
                    "transcript_path": repo_relative_path(transcript_path),
                    "transcript": transcript,
                    "clean_reference_path": None,
                    "speaker_id": None,
                    "language": None,
                    "metadata": {
                        "relative_path": audio_member.name,
                        "duration_sec": duration_sec,
                        "source": str(input_path),
                        "selection": selection,
                    },
                }
            )
    return rows


def discover_audio_transcript_pairs(
    input_dir: str | Path,
    extensions: list[str] | None = None,
    transcript_extension: str = ".txt",
    cache_dir: str | Path = "data/cache",
    extract_archives: bool = True,
    max_files: int | None = None,
    shuffle: bool = False,
    min_duration_sec: float | None = None,
    selection: str = "first",
    recording_id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    extensions = extensions or [".opus", ".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    selection = selection.lower()
    if selection == "all":
        selection = "first"
    root = Path(input_dir)
    suffixes = "".join(root.suffixes).lower()
    if (
        extract_archives
        and root.is_file()
        and suffixes.endswith((".tar.gz", ".tgz", ".tar"))
        and max_files is not None
    ):
        rows = _discover_limited_archive_pairs(
            root,
            extensions=extensions,
            transcript_extension=transcript_extension,
            cache_dir=Path(cache_dir),
            max_files=int(max_files),
            min_duration_sec=min_duration_sec,
            selection=selection,
            recording_id_prefix=recording_id_prefix,
        )
        if shuffle:
            random.Random(1337).shuffle(rows)
        return rows
    if extract_archives and root.is_file() and suffixes.endswith((".tar.gz", ".tgz", ".tar")) and selection != "first":
        raise ValueError("Archive selection modes other than 'first' require max_files.")
    if extract_archives:
        root = _maybe_extract_archive(root, Path(cache_dir))
    rows: list[dict[str, Any]] = []
    seen_recording_ids: set[str] = set()
    audio_paths = [
        path for path in root.rglob("*") if path.suffix.lower() in extensions
    ]
    for audio_path in sorted(audio_paths, key=lambda path: _directory_audio_sort_key(root, path)):
        if audio_path.suffix.lower() not in extensions:
            continue
        transcript_path = audio_path.with_suffix(transcript_extension)
        transcript = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else ""
        recording_id = audio_path.stem
        if recording_id in seen_recording_ids:
            continue
        seen_recording_ids.add(recording_id)
        if min_duration_sec is not None:
            # Duration filtering for directory datasets is intentionally conservative:
            # rows without cheap duration metadata are left to future backend-specific
            # probes instead of decoding every file during manifest creation.
            continue
        rows.append(
            {
                "recording_id": _recording_id(recording_id, recording_id_prefix),
                "audio_path": repo_relative_path(audio_path),
                "transcript_path": repo_relative_path(transcript_path) if transcript_path.exists() else None,
                "transcript": transcript,
                "clean_reference_path": None,
                "speaker_id": None,
                "language": None,
                "metadata": {"relative_path": str(audio_path.relative_to(root)), "source": str(root)},
            }
        )
    if shuffle:
        random.Random(1337).shuffle(rows)
    if max_files is not None:
        rows = rows[: int(max_files)]
    return rows


def write_manifest(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    write_jsonl(path, rows)
    return path


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if key in cfg:
            return cfg[key]
    except (TypeError, KeyError):
        pass
    return getattr(cfg, key, default)


def _cfg_list(cfg: Any, key: str) -> list[str]:
    value = _cfg_value(cfg, key, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _filter_excluded_rows(rows: list[dict[str, Any]], data_cfg: Any) -> list[dict[str, Any]]:
    excluded_recording_ids = set(_cfg_list(data_cfg, "exclude_recording_ids"))
    if not excluded_recording_ids:
        return rows
    return [row for row in rows if str(row.get("recording_id", "")) not in excluded_recording_ids]


def _selection_prefix(input_dir: str | Path, selection_cfg: Any, use_default: bool) -> str | None:
    explicit = _cfg_value(selection_cfg, "recording_id_prefix", None)
    if explicit is False:
        return None
    if explicit not in (None, ""):
        return str(explicit)
    if not use_default:
        return None
    path = Path(input_dir)
    return _archive_cache_name(path) if path.is_file() else path.name


def build_manifest_from_config(data_cfg: Any, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    if _cfg_value(data_cfg, "manifest_path", None):
        rows = read_manifest(_cfg_value(data_cfg, "manifest_path"))
        rows = _filter_excluded_rows(rows, data_cfg)
        max_files = _cfg_value(data_cfg, "max_files", None)
        if max_files is not None:
            rows = rows[: int(max_files)]
        if output_path is not None:
            write_manifest(rows, output_path)
        return rows
    input_dir = _cfg_value(data_cfg, "input_dir", None)
    selections = _cfg_value(data_cfg, "selections", None)
    if selections and input_dir in (None, ""):
        rows: list[dict[str, Any]] = []
        use_default_prefixes = len(selections) > 1
        for selection_cfg in selections:
            input_dir = _cfg_value(selection_cfg, "input_dir")
            if input_dir is None:
                raise ValueError("Every data selection requires input_dir.")
            selection_max_files = _cfg_value(selection_cfg, "max_files", None)
            rows.extend(
                discover_audio_transcript_pairs(
                    input_dir=input_dir,
                    extensions=list(_cfg_value(selection_cfg, "extensions", _cfg_value(data_cfg, "extensions"))),
                    transcript_extension=str(
                        _cfg_value(
                            selection_cfg,
                            "transcript_extension",
                            _cfg_value(data_cfg, "transcript_extension", ".txt"),
                        )
                    ),
                    cache_dir=_cfg_value(selection_cfg, "cache_dir", _cfg_value(data_cfg, "cache_dir", "data/cache")),
                    max_files=int(selection_max_files) if selection_max_files is not None else None,
                    shuffle=bool(_cfg_value(selection_cfg, "shuffle", False)),
                    min_duration_sec=_cfg_value(
                        selection_cfg,
                        "min_duration_sec",
                        _cfg_value(data_cfg, "min_duration_sec", None),
                    ),
                    selection=str(_cfg_value(selection_cfg, "selection", "first")),
                    recording_id_prefix=_selection_prefix(input_dir, selection_cfg, use_default_prefixes),
                )
            )
        rows = _filter_excluded_rows(rows, data_cfg)
        max_files = _cfg_value(data_cfg, "max_files", None)
        if max_files is not None:
            rows = rows[: int(max_files)]
        if output_path is not None:
            write_manifest(rows, output_path)
        return rows
    rows = discover_audio_transcript_pairs(
        input_dir=input_dir,
        extensions=list(_cfg_value(data_cfg, "extensions")),
        transcript_extension=str(_cfg_value(data_cfg, "transcript_extension")),
        cache_dir=_cfg_value(data_cfg, "cache_dir"),
        max_files=_cfg_value(data_cfg, "max_files"),
        shuffle=bool(_cfg_value(data_cfg, "shuffle", False)),
        min_duration_sec=_cfg_value(data_cfg, "min_duration_sec", None),
        selection=str(_cfg_value(data_cfg, "selection", "first")),
    )
    rows = _filter_excluded_rows(rows, data_cfg)
    if output_path is not None:
        write_manifest(rows, output_path)
    return rows
