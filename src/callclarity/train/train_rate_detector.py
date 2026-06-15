from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from callclarity.models.neural_rate_tcn import NeuralRateTcn
from callclarity.train.checkpoints import save_checkpoint
from callclarity.utils.files import ensure_dir, write_json


def train_rate_detector(cfg: Any, output_dir: str | Path) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    model_cfg = cfg.train.model
    model = NeuralRateTcn(
        n_mels=int(model_cfg.n_mels),
        channels=int(model_cfg.channels),
        layers=int(model_cfg.layers),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
    losses = []
    for _ in range(int(cfg.train.max_steps)):
        features = torch.randn(int(cfg.train.batch_size), int(model_cfg.n_mels), 64)
        target_fast = (torch.rand(int(cfg.train.batch_size), 1, 64) > 0.7).float()
        target_rate = target_fast * 6.5
        out_dict = model(features)
        loss = torch.nn.functional.binary_cross_entropy(out_dict["fast_speech_probability"], target_fast)
        loss = loss + torch.nn.functional.l1_loss(out_dict["syllables_per_sec"], target_rate)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    save_checkpoint(out / "best_model.pt", model, {"loss": losses[-1] if losses else None})
    scripted = torch.jit.trace(model, torch.randn(1, int(model_cfg.n_mels), 64), strict=False)
    scripted.save(str(out / "best_model.ts"))
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
