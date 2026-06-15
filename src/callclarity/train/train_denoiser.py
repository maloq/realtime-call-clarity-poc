from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from callclarity.models.losses import magnitude_l1
from callclarity.models.tiny_decrackler import TinyDecrackler
from callclarity.models.tiny_mask_gru import TinyMaskGru
from callclarity.train.checkpoints import save_checkpoint
from callclarity.train.crackle_data import (
    SyntheticCrackleDataset,
    build_crackle_snippet_bank,
    discover_audio_files,
)
from callclarity.train.datamodules import synthetic_wave_batch
from callclarity.utils.device import resolve_torch_device
from callclarity.utils.files import ensure_dir, write_json


def _train_tiny_decrackler(cfg: Any, output_dir: str | Path) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    requested_device, device = resolve_torch_device(getattr(cfg.runtime, "device", "auto"))
    sample_rate = int(cfg.audio.sample_rate)
    clean_dirs = [str(path) for path in cfg.train.clean_dirs]
    crackle_dirs = [str(path) for path in cfg.train.get("crackle_dirs", [])]
    clean_files = discover_audio_files(clean_dirs)
    crackle_files = discover_audio_files(crackle_dirs)
    if not clean_files:
        message = (
            "tiny_decrackler training needs clean WAV/Opus files. "
            "Set train.clean_dirs to one or more folders/files."
        )
        write_json(out / "error.json", {"error": message, "clean_dirs": clean_dirs})
        raise RuntimeError(message)
    snippets = build_crackle_snippet_bank(
        crackle_files,
        sample_rate,
        max_files=int(cfg.train.get("max_crackle_files", 32)),
        max_snippets=int(cfg.train.get("max_crackle_snippets", 256)),
    )
    model_cfg = cfg.train.model
    model = TinyDecrackler(
        channels=int(model_cfg.channels),
        kernel_size=int(model_cfg.kernel_size),
        dilations=tuple(int(v) for v in model_cfg.dilations),
        max_correction=float(model_cfg.max_correction),
    ).to(torch.device(device))
    segment_samples = max(256, int(round(float(cfg.train.segment_ms) * sample_rate / 1000.0)))
    dataset = SyntheticCrackleDataset(
        clean_files,
        sample_rate,
        segment_samples,
        cfg.train.synthetic_crackle,
        snippets,
        length=max(int(cfg.train.max_steps) * int(cfg.train.batch_size), int(cfg.train.batch_size)),
        seed=int(cfg.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
    losses: list[float] = []
    correction_penalty = float(cfg.train.get("correction_l1_weight", 0.02))
    iterator = iter(loader)
    for _step in range(int(cfg.train.max_steps)):
        try:
            noisy, clean = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            noisy, clean = next(iterator)
        noisy = noisy.to(torch.device(device))
        clean = clean.to(torch.device(device))
        pred, correction = model(noisy)
        waveform_loss = torch.mean(torch.abs(pred - clean))
        residual_loss = torch.mean(torch.abs(correction)) * correction_penalty
        loss = waveform_loss + residual_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(float(loss.item()))
    save_checkpoint(
        out / "best_model.pt",
        model,
        {
            "loss": losses[-1] if losses else None,
            "model_config": model.config_dict(),
            "sample_rate": sample_rate,
            "clean_file_count": len(clean_files),
            "crackle_file_count": len(crackle_files),
            "crackle_snippet_count": len(snippets),
            "device_requested": requested_device,
            "device": device,
        },
    )
    (out / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    metrics = {
        "loss": losses[-1] if losses else None,
        "clean_file_count": len(clean_files),
        "crackle_file_count": len(crackle_files),
        "crackle_snippet_count": len(snippets),
        "checkpoint": str(out / "best_model.pt"),
        "device_requested": requested_device,
        "device": device,
    }
    write_json(out / "validation_metrics.json", metrics)
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out / "training_curves.png")
    plt.close()
    return metrics


def train_denoiser(cfg: Any, output_dir: str | Path) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    if str(cfg.train.name) == "tiny_decrackler":
        return _train_tiny_decrackler(cfg, out)
    if not bool(cfg.train.synthetic_noise.enabled):
        message = (
            "tiny_mask_gru training needs clean/noisy pairs or explicit synthetic-noise generation. "
            "Set train.synthetic_noise.enabled=true for a smoke run."
        )
        write_json(out / "error.json", {"error": message})
        raise RuntimeError(message)
    model_cfg = cfg.train.model
    model = TinyMaskGru(
        n_fft=int(model_cfg.n_fft),
        hop_length=int(model_cfg.hop_length),
        hidden_size=int(model_cfg.hidden_size),
        num_layers=int(model_cfg.num_layers),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
    losses = []
    for step in range(int(cfg.train.max_steps)):
        noisy, clean = synthetic_wave_batch(int(cfg.train.batch_size), int(cfg.audio.sample_rate))
        pred, _, _ = model(noisy)
        loss = magnitude_l1(pred, clean)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    save_checkpoint(out / "best_model.pt", model, {"loss": losses[-1] if losses else None})
    (out / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    write_json(out / "validation_metrics.json", {"loss": losses[-1] if losses else None})
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out / "training_curves.png")
    plt.close()
    return {"loss": losses[-1] if losses else None, "checkpoint": str(out / "best_model.pt")}
