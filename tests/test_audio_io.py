from __future__ import annotations

from pathlib import Path

from callclarity.io import audio_io


def test_audio_decode_temp_root_is_repo_local_cache() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = audio_io._local_temp_root().resolve()

    assert repo_root.resolve() in root.parents
    assert root.relative_to(repo_root) == Path("data/cache/tmp")
