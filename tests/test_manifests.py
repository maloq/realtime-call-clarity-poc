import io
import tarfile
import wave
from pathlib import Path

from omegaconf import OmegaConf

from callclarity.data.manifests import build_manifest_from_config, discover_audio_transcript_pairs
from callclarity.utils.files import repo_relative_path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _wav_bytes(duration_sec: float, sample_rate: int = 8000) -> bytes:
    data = io.BytesIO()
    num_frames = int(duration_sec * sample_rate)
    with wave.open(data, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)
    return data.getvalue()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _make_archive(path: Path, durations: dict[str, float]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for stem, duration_sec in durations.items():
            _add_tar_bytes(archive, f"calls/{stem}.wav", _wav_bytes(duration_sec))
            _add_tar_bytes(archive, f"calls/{stem}.txt", f"{stem} transcript\n".encode("utf-8"))


def test_directory_discovery_prioritizes_raw_folder_and_skips_duplicate_ids(tmp_path):
    root = tmp_path / "samples"
    _touch(root / "clean" / "same.wav")
    _touch(root / "raw" / "same.wav")
    _touch(root / "clean" / "other.wav")

    rows = discover_audio_transcript_pairs(root, max_files=2)

    assert [row["metadata"]["relative_path"] for row in rows] == [
        "raw/same.wav",
        "clean/other.wav",
    ]


def test_archive_selection_can_take_longest_files(tmp_path):
    archive_path = tmp_path / "calls.tar.gz"
    _make_archive(archive_path, {"short": 0.1, "long": 0.5, "mid": 0.3})

    rows = discover_audio_transcript_pairs(
        archive_path,
        extensions=[".wav"],
        cache_dir=tmp_path / "cache",
        max_files=2,
        selection="longest",
        recording_id_prefix="archive",
    )

    assert [row["recording_id"] for row in rows] == ["archive_long", "archive_mid"]
    assert [row["transcript"] for row in rows] == ["long transcript", "mid transcript"]
    assert [row["metadata"]["duration_sec"] for row in rows] == [0.5, 0.3]


def test_repo_relative_path_keeps_repo_paths_relative():
    path = Path.cwd() / "outputs" / "example.wav"

    assert repo_relative_path(path) == "outputs/example.wav"


def test_config_selections_use_two_archives_and_raw_folder_only(tmp_path):
    archive_a = tmp_path / "archive_a.tar.gz"
    archive_b = tmp_path / "archive_b.tar.gz"
    _make_archive(archive_a, {"a_short": 0.1, "a_long": 0.6})
    _make_archive(archive_b, {"b_short": 0.2, "b_long": 0.7})

    samples = tmp_path / "samples"
    _touch(samples / "raw" / "raw_sample.wav")
    _touch(samples / "clean" / "raw_sample.wav")
    _touch(samples / "processed" / "processed_sample.wav")

    cfg = OmegaConf.create(
        {
            "input_dir": None,
            "cache_dir": str(tmp_path / "cache"),
            "manifest_path": None,
            "extensions": [".wav"],
            "transcript_extension": ".txt",
            "max_files": None,
            "min_duration_sec": None,
            "shuffle": False,
            "selections": [
                {
                    "input_dir": str(archive_a),
                    "max_files": 1,
                    "selection": "longest",
                    "recording_id_prefix": "a",
                },
                {
                    "input_dir": str(archive_b),
                    "max_files": 1,
                    "selection": "longest",
                    "recording_id_prefix": "b",
                },
                {
                    "input_dir": str(samples / "raw"),
                    "max_files": None,
                    "selection": "all",
                    "recording_id_prefix": "raw",
                },
            ],
        }
    )

    rows = build_manifest_from_config(cfg)

    assert [row["recording_id"] for row in rows] == ["a_a_long", "b_b_long", "raw_raw_sample"]
    assert {row["metadata"]["relative_path"] for row in rows} == {
        "calls/a_long.wav",
        "calls/b_long.wav",
        "raw_sample.wav",
    }


def test_config_input_dir_overrides_default_selections(tmp_path):
    archive_path = tmp_path / "archive.tar.gz"
    _make_archive(archive_path, {"archive_sample": 0.5})
    explicit = tmp_path / "explicit"
    _touch(explicit / "explicit_sample.wav")

    cfg = OmegaConf.create(
        {
            "input_dir": str(explicit),
            "cache_dir": str(tmp_path / "cache"),
            "manifest_path": None,
            "extensions": [".wav"],
            "transcript_extension": ".txt",
            "max_files": None,
            "min_duration_sec": None,
            "shuffle": False,
            "selections": [
                {
                    "input_dir": str(archive_path),
                    "max_files": 1,
                    "selection": "longest",
                    "recording_id_prefix": "archive",
                }
            ],
        }
    )

    rows = build_manifest_from_config(cfg)

    assert [row["recording_id"] for row in rows] == ["explicit_sample"]


def test_config_can_exclude_recording_ids_after_selection(tmp_path):
    archive_path = tmp_path / "calls.tar.gz"
    _make_archive(archive_path, {"short": 0.1, "long": 0.5, "mid": 0.3})

    cfg = OmegaConf.create(
        {
            "input_dir": None,
            "cache_dir": str(tmp_path / "cache"),
            "manifest_path": None,
            "extensions": [".wav"],
            "transcript_extension": ".txt",
            "max_files": None,
            "min_duration_sec": None,
            "shuffle": False,
            "exclude_recording_ids": ["archive_long"],
            "selections": [
                {
                    "input_dir": str(archive_path),
                    "max_files": 3,
                    "selection": "longest",
                    "recording_id_prefix": "archive",
                }
            ],
        }
    )

    rows = build_manifest_from_config(cfg)

    assert [row["recording_id"] for row in rows] == ["archive_mid", "archive_short"]
