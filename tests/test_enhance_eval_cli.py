import torch
import pandas as pd
from scipy.io import wavfile

from callclarity.cli import enhance_eval_main


def test_enhance_eval_command_writes_reports(tmp_path):
    sr = 16000
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    t = torch.arange(sr // 5).float() / sr
    wav = (0.05 * torch.sin(2 * torch.pi * 440 * t)).numpy()
    wavfile.write(data_dir / "sample.wav", sr, (wav * 32767).astype("int16"))

    out = tmp_path / "report"
    code = enhance_eval_main(
        [str(data_dir), "--preset", "receive_baseline", "--out", str(out), "--max-files", "1"]
        + ["--disable-neural-metrics"]
    )
    assert code == 0
    run_dir = out / "_internal" / "runs" / "receive_baseline"
    assert (out / "comparison.csv").exists()
    comparison = pd.read_csv(out / "comparison.csv")
    row = comparison.iloc[0]
    assert row["denoise_method"] == "spectral_gate"
    assert row["vad_method"] == "energy"
    assert row["leveler_method"] == "speech_aware_agc"
    assert row["rate_detector_method"] == "none"
    assert row["slowdown_method"] == "none"
    assert (out / "samples_index.csv").exists()
    assert (run_dir / "metrics_per_file.csv").exists()
    assert (run_dir / "per_chunk_metrics.csv").exists()
    assert (run_dir / "guardrails.csv").exists()
    assert (out / "README.md").exists()
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    report_md = (out / "report.md").read_text(encoding="utf-8")
    assert "_internal/plots/method_key_metrics.png" in report_md
    assert "<img" in (out / "report.html").read_text(encoding="utf-8")
    assert (out / "_internal" / "comparison.json").exists()
    assert (out / "_internal" / "samples_index.json").exists()
    assert (out / "_internal" / "plots" / "method_key_metrics.png").exists()
    assert not (out / "comparison.json").exists()
    assert not (out / "samples_index.json").exists()
    sample_dirs = [path for path in (out / "samples").iterdir() if path.is_dir()]
    assert len(sample_dirs) == 1
    assert (sample_dirs[0] / "receive_baseline.wav").exists()
    assert (sample_dirs[0] / "raw.wav").exists()
    assert (sample_dirs[0] / "metrics.csv").exists()
    assert not (sample_dirs[0] / "metrics.json").exists()
    assert not (sample_dirs[0] / "info.json").exists()
    assert (out / "_internal" / "samples" / sample_dirs[0].name / "metrics.json").exists()
    assert (out / "_internal" / "samples" / sample_dirs[0].name / "info.json").exists()
