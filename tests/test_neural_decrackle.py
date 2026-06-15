from pathlib import Path

import torch

from callclarity.config import compose_config
from callclarity.methods.repair.neural_decrackle import NeuralDecrackleProcessor
from callclarity.models.tiny_decrackler import TinyDecrackler
from callclarity.train.crackle_data import SyntheticCrackleDataset, discover_audio_files
from callclarity.train.train_denoiser import train_denoiser
from callclarity.types import AudioChunk


def test_discover_crackle_audio_files_accepts_wav_and_opus():
    paths = discover_audio_files(
        [
            "data/test_data_samples/clean",
            "data/test_data_samples/crackle_examples",
        ]
    )
    suffixes = {path.suffix.lower() for path in paths}
    assert ".wav" in suffixes
    assert ".opus" in suffixes


def test_synthetic_crackle_dataset_from_mixed_audio_dirs():
    clean_files = discover_audio_files(["data/test_data_samples/clean"])
    dataset = SyntheticCrackleDataset(
        clean_files,
        sample_rate=16000,
        segment_samples=1024,
        synthetic_cfg={
            "clicks_per_second": 20.0,
            "min_amplitude": 0.2,
            "max_amplitude": 0.8,
            "noise_floor": 0.0,
        },
        length=2,
    )
    noisy, clean = dataset[0]
    assert noisy.shape == clean.shape == (1024,)
    assert torch.mean(torch.abs(noisy - clean)).item() > 0.0
    assert torch.isfinite(noisy).all()


def test_neural_decrackle_checkpoint_inference_preserves_shape(tmp_path: Path):
    model = TinyDecrackler(channels=8, kernel_size=3, dilations=(1, 2), max_correction=0.5)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = tmp_path / "tiny_decrackler.pt"
    torch.save({"model": model.state_dict(), "model_config": model.config_dict()}, checkpoint)
    processor = NeuralDecrackleProcessor(
        {
            "checkpoint_path": str(checkpoint),
            "device": "cpu",
            "blend": 1.0,
            "max_frame_correction": 0.35,
        }
    )
    samples = torch.randn(1, 320) * 0.05
    result = processor.process(AudioChunk(samples, 16000, 0.0))
    assert result.chunk.samples.shape == samples.shape
    assert torch.allclose(result.chunk.samples, samples, atol=1e-6)
    assert result.metrics["neural_decrackle_device"] == "cpu"
    assert torch.isfinite(result.chunk.samples).all()


def test_train_tiny_decrackler_smoke(tmp_path: Path):
    cfg = compose_config(
        [
            "train=tiny_decrackler",
            "runtime.device=cpu",
            "train.max_steps=1",
            "train.batch_size=2",
            "train.segment_ms=64",
            "train.model.channels=8",
            "train.model.kernel_size=3",
            "train.model.dilations=[1,2]",
            f"output_dir={tmp_path}",
        ]
    )
    result = train_denoiser(cfg, tmp_path)
    assert Path(result["checkpoint"]).exists()
    assert result["clean_file_count"] > 0
    assert result["crackle_file_count"] > 0
