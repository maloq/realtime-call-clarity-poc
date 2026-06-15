import tarfile
from pathlib import Path

import torch
from scipy.io import wavfile

from callclarity.train.crackle_classifier import (
    CrackleFeatureConfig,
    extract_crackle_features,
    predict_crackle_file,
    pseudolabel_crackle_dataset,
    train_crackle_classifier,
)
from callclarity.train.crackle_data import discover_audio_files


def test_crackle_feature_extraction_is_windowed_and_finite():
    sample_rate = 16000
    t = torch.arange(sample_rate // 2).float() / sample_rate
    waveform = 0.05 * torch.sin(2.0 * torch.pi * 440.0 * t).view(1, -1)
    features, times = extract_crackle_features(
        waveform,
        sample_rate,
        CrackleFeatureConfig(sample_rate=sample_rate, window_ms=100.0, hop_ms=50.0),
    )
    assert features.ndim == 2
    assert features.shape[0] == times.shape[0]
    assert features.shape[1] > 8
    assert torch.isfinite(torch.from_numpy(features)).all()


def test_crackle_audio_discovery_accepts_tar_archive(tmp_path: Path):
    sample_rate = 16000
    t = torch.arange(sample_rate // 10).float() / sample_rate
    wav = (0.05 * torch.sin(2.0 * torch.pi * 440.0 * t)).numpy()
    source = tmp_path / "call.wav"
    wavfile.write(source, sample_rate, (wav * 32767).astype("int16"))
    archive = tmp_path / "calls.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="calls/a/call.wav")

    files = discover_audio_files([archive], cache_dir=tmp_path / "cache", max_files=1)

    assert len(files) == 1
    assert files[0].suffix == ".wav"
    assert files[0].exists()


def test_crackle_classifier_train_predict_and_pseudolabel_smoke(tmp_path: Path):
    result = train_crackle_classifier(
        ["data/test_data_samples/clean"],
        ["data/test_data_samples/crackle_examples"],
        tmp_path / "model",
        epochs=2,
        hidden_size=8,
        max_files_per_class=8,
        device="cpu",
    )
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.exists()
    assert result["clean_file_count"] > 0
    assert result["crackle_file_count"] > 0
    assert result["train_window_count"] > 0

    prediction = predict_crackle_file(
        "data/test_data_samples/crackle_examples/crackle1.wav",
        checkpoint,
        device="cpu",
    )
    assert 0.0 <= prediction["crackle_prob_p95"] <= 1.0
    assert prediction["num_windows"] > 0

    summary = pseudolabel_crackle_dataset(
        ["data/test_data_samples/clean", "data/test_data_samples/crackle_examples"],
        checkpoint,
        tmp_path / "pseudo",
        device="cpu",
        max_files=6,
        copy_top_n=1,
    )
    assert Path(summary["output_csv"]).exists()
    assert (tmp_path / "pseudo" / "pseudolabels.json").exists()
    assert (tmp_path / "pseudo" / "review" / "crackle").exists()
    assert sum(summary["label_counts"].values()) == 6
