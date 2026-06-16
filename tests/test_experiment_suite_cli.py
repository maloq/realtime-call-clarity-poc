from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from scipy.io import wavfile

from callclarity.cli import main


def test_experiment_command_writes_canonical_suite_layout(tmp_path: Path) -> None:
    sr = 16000
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    t = torch.arange(sr // 5).float() / sr
    wav = (0.05 * torch.sin(2 * torch.pi * 440 * t)).numpy()
    wavfile.write(data_dir / "sample.wav", sr, (wav * 32767).astype("int16"))

    out = tmp_path / "experiment_suite"
    cache = tmp_path / "input_cache"
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {out.as_posix()}",
                "clean_output_dir: false",
                "run_comparison: true",
                f"input_cache_root: {cache.as_posix()}",
                "refresh_inputs: true",
                "presets:",
                "  - baseline",
                "common_overrides:",
                "  - data.selections=[]",
                f"  - data.input_dir={data_dir.as_posix()}",
                "  - data.max_files=1",
                "  - sample_selector.num_examples=1000000",
                "  - metrics.no_reference.nisqa.enabled=false",
                "  - metrics.no_reference.dnsmos.enabled=false",
                "  - metrics.no_reference.squim.enabled=false",
                "  - runtime.parallel_eval=false",
                "  - runtime.num_workers=1",
                "per_preset_overrides: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    try:
        assert main(["experiment", "--config", str(config_path)]) == 0
    finally:
        os.chdir(previous_cwd)

    run_dir = out / "_internal" / "runs" / "baseline"
    assert (out / "comparison.csv").exists()
    assert (out / "method_audio.csv").exists()
    assert (out / "samples_index.csv").exists()
    assert (out / "README.md").exists()
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    assert (out / "_internal" / "suite_config_resolved.yaml").exists()
    assert (out / "_internal" / "comparison.json").exists()
    assert (run_dir / "metrics_per_file.csv").exists()
    assert (run_dir / "samples" / "selected_samples.csv").exists()

    sample_dirs = [path for path in (out / "samples").iterdir() if path.is_dir()]
    assert len(sample_dirs) == 1
    sample_dir = sample_dirs[0]
    assert (sample_dir / "raw.wav").exists()
    assert (sample_dir / "baseline.wav").exists()
    assert (sample_dir / "metrics.csv").exists()
    assert not (sample_dir / "metrics.json").exists()
    assert (out / "_internal" / "samples" / sample_dir.name / "metrics.json").exists()
    assert (out / "_internal" / "samples" / sample_dir.name / "info.json").exists()

    method_audio = pd.read_csv(out / "method_audio.csv")
    assert method_audio.iloc[0]["method"] == "baseline"
    assert Path(method_audio.iloc[0]["processed_wav"]).exists()


def test_experiment_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {(tmp_path / 'out').as_posix()}",
                "presets:",
                "  - baseline",
                "common_overrides:",
                "  - data.selections=[]",
                f"  - data.input_dir={tmp_path.as_posix()}",
                "per_preset_overrides: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    try:
        assert main(["experiment", "--config", str(config_path), "--dry-run"]) == 0
    finally:
        os.chdir(previous_cwd)
