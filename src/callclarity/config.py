from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        return compose(config_name="config", overrides=overrides or [])
