import torch
from scipy.io import wavfile

from callclarity.config import compose_config
from callclarity.experiments.runner import run_eval


def test_integration_synthetic_eval(tmp_path):
    sr = 16000
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    t = torch.arange(sr // 2).float() / sr
    wav = (0.1 * torch.sin(2 * torch.pi * 220 * t)).numpy()
    wavfile.write(data_dir / "sample.wav", sr, (wav * 32767).astype("int16"))
    (data_dir / "sample.txt").write_text("hello world", encoding="utf-8")
    out = tmp_path / "out"
    cfg = compose_config(
        [
            f"data.input_dir={data_dir}",
            "data.max_files=1",
            "pipeline=denoise_agc",
            "metrics.no_reference.nisqa.enabled=false",
            "metrics.no_reference.dnsmos.enabled=false",
            "metrics.no_reference.squim.enabled=false",
            f"output_dir={out}",
            "sample_selector.num_examples=1",
        ]
    )
    summary = run_eval(cfg, out)
    assert summary["num_files"] == 1
    assert (out / "metrics_summary.json").exists()
    assert (out / "metrics_per_file.csv").exists()
    assert (out / "latency_summary.json").exists()
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    assert (out / "samples" / "sample" / "processed.wav").exists()
