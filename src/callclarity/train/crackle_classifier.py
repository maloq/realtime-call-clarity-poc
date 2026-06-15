from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal
from torch import nn

from callclarity.io.audio_io import load_audio
from callclarity.train.crackle_data import discover_audio_files
from callclarity.train.checkpoints import save_checkpoint
from callclarity.utils.device import resolve_torch_device
from callclarity.utils.files import ensure_dir, write_json


FEATURE_NAMES = [
    "rms",
    "peak",
    "crest",
    "clip_pct",
    "near_clip_pct",
    "zcr",
    "slope_p95",
    "slope_p99",
    "slope_crest",
    "median_res_p95",
    "median_res_p99",
    "median_res_p999",
    "median_res_ratio",
    "hf_energy_ratio",
    "spectral_centroid",
    "spectral_rolloff",
]


@dataclass
class CrackleFeatureConfig:
    sample_rate: int = 16000
    window_ms: float = 250.0
    hop_ms: float = 125.0
    median_kernel: int = 5
    high_frequency_hz: float = 3500.0
    rolloff_pct: float = 0.95


class CrackleFeatureClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        middle = max(4, self.hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(self.hidden_size, middle),
            nn.ReLU(),
            nn.Linear(middle, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


def _mono_np(waveform: torch.Tensor) -> np.ndarray:
    x = waveform.detach().float().cpu()
    if x.ndim == 2:
        x = x.mean(dim=0)
    return x.clamp(-1.0, 1.0).numpy().astype(np.float32, copy=False)


def _window_starts(num_samples: int, window: int, hop: int) -> list[int]:
    if num_samples <= 0:
        return [0]
    if num_samples <= window:
        return [0]
    starts = list(range(0, num_samples - window + 1, hop))
    if starts[-1] != num_samples - window:
        starts.append(num_samples - window)
    return starts


def _pad_window(x: np.ndarray, window: int) -> np.ndarray:
    if x.size >= window:
        return x[:window]
    return np.pad(x, (0, window - x.size), mode="constant")


def _safe_percentile(x: np.ndarray, percentile: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, percentile))


def _spectral_features(x: np.ndarray, sample_rate: int, cfg: CrackleFeatureConfig) -> tuple[float, float, float]:
    if x.size < 4:
        return 0.0, 0.0, 0.0
    windowed = x * np.hanning(x.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sample_rate))
    total = float(np.sum(spec) + 1e-12)
    hf = float(np.sum(spec[freqs >= float(cfg.high_frequency_hz)]) / total)
    centroid = float(np.sum(freqs * spec) / total)
    cumulative = np.cumsum(spec)
    target = float(cfg.rolloff_pct) * cumulative[-1]
    idx = int(np.searchsorted(cumulative, target))
    rolloff = float(freqs[min(idx, freqs.size - 1)])
    return hf, centroid, rolloff


def _feature_row(x: np.ndarray, sample_rate: int, cfg: CrackleFeatureConfig) -> list[float]:
    x = x.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    crest = peak / max(rms, 1e-6)
    clip_pct = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    near_clip_pct = float(np.mean(np.abs(x) >= 0.90)) if x.size else 0.0
    signs = np.signbit(x)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if x.size > 1 else 0.0
    slope = np.abs(np.diff(x, prepend=x[0] if x.size else 0.0))
    slope_p95 = _safe_percentile(slope, 95.0)
    slope_p99 = _safe_percentile(slope, 99.0)
    slope_crest = slope_p99 / max(float(np.mean(slope) + 1e-6), 1e-6)
    kernel = max(3, int(cfg.median_kernel))
    kernel = kernel if kernel % 2 else kernel + 1
    if x.size >= kernel:
        median = signal.medfilt(x, kernel_size=kernel)
    else:
        median = np.zeros_like(x)
    residual = np.abs(x - median)
    med_p95 = _safe_percentile(residual, 95.0)
    med_p99 = _safe_percentile(residual, 99.0)
    med_p999 = _safe_percentile(residual, 99.9)
    med_ratio = med_p999 / max(rms, 1e-6)
    hf, centroid, rolloff = _spectral_features(x, sample_rate, cfg)
    return [
        rms,
        peak,
        crest,
        clip_pct,
        near_clip_pct,
        zcr,
        slope_p95,
        slope_p99,
        slope_crest,
        med_p95,
        med_p99,
        med_p999,
        med_ratio,
        hf,
        centroid / max(sample_rate / 2.0, 1.0),
        rolloff / max(sample_rate / 2.0, 1.0),
    ]


def extract_crackle_features(
    waveform: torch.Tensor,
    sample_rate: int,
    cfg: CrackleFeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x = _mono_np(waveform)
    window = max(32, int(round(cfg.window_ms * sample_rate / 1000.0)))
    hop = max(16, int(round(cfg.hop_ms * sample_rate / 1000.0)))
    rows = []
    times = []
    for start in _window_starts(x.size, window, hop):
        frame = _pad_window(x[start : start + window], window)
        rows.append(_feature_row(frame, sample_rate, cfg))
        times.append(start / float(sample_rate))
    return np.asarray(rows, dtype=np.float32), np.asarray(times, dtype=np.float32)


def _load_features_for_files(
    files: list[Path],
    label: int,
    cfg: CrackleFeatureConfig,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = []
    labels = []
    meta: list[dict[str, Any]] = []
    selected = files[: int(max_files)] if max_files else files
    for path in selected:
        try:
            waveform, sample_rate = load_audio(path, cfg.sample_rate, "mono")
            features, _times = extract_crackle_features(waveform, sample_rate, cfg)
        except Exception as exc:
            meta.append({"path": str(path), "label": label, "error": str(exc), "windows": 0})
            continue
        rows.append(features)
        labels.append(np.full(features.shape[0], int(label), dtype=np.float32))
        meta.append({"path": str(path), "label": label, "error": None, "windows": int(features.shape[0])})
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.float32), meta
    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0), meta


def _normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / np.maximum(std, 1e-6)


def train_crackle_classifier(
    clean_paths: list[str | Path],
    crackle_paths: list[str | Path],
    output_dir: str | Path,
    *,
    sample_rate: int = 16000,
    window_ms: float = 250.0,
    hop_ms: float = 125.0,
    hidden_size: int = 48,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    device: str = "auto",
    max_files_per_class: int | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    clean_files = discover_audio_files(clean_paths, max_files=max_files_per_class)
    crackle_files = discover_audio_files(crackle_paths, max_files=max_files_per_class)
    if not clean_files or not crackle_files:
        raise RuntimeError(
            f"Need clean and crackle audio. Found clean={len(clean_files)}, crackle={len(crackle_files)}."
        )
    feature_cfg = CrackleFeatureConfig(sample_rate=sample_rate, window_ms=window_ms, hop_ms=hop_ms)
    clean_x, clean_y, clean_meta = _load_features_for_files(
        clean_files, 0, feature_cfg, max_files=max_files_per_class
    )
    crackle_x, crackle_y, crackle_meta = _load_features_for_files(
        crackle_files, 1, feature_cfg, max_files=max_files_per_class
    )
    features = np.concatenate([clean_x, crackle_x], axis=0)
    labels = np.concatenate([clean_y, crackle_y], axis=0)
    if features.shape[0] == 0:
        raise RuntimeError("No training windows could be decoded.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(features.shape[0])
    features = features[order]
    labels = labels[order]
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    x = torch.from_numpy(_normalize(features, mean, std)).float()
    y = torch.from_numpy(labels).float()
    requested_device, effective_device = resolve_torch_device(device)
    torch_device = torch.device(effective_device)
    model = CrackleFeatureClassifier(len(FEATURE_NAMES), hidden_size=hidden_size).to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=torch_device)
    losses = []
    for _epoch in range(int(epochs)):
        logits = model(x.to(torch_device))
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y.to(torch_device),
            pos_weight=pos_weight,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(torch_device))).detach().cpu().numpy()
    pred = probs >= 0.5
    accuracy = float(np.mean(pred == labels.astype(bool)))
    save_checkpoint(
        out / "best_model.pt",
        model,
        {
            "model_config": {"input_dim": len(FEATURE_NAMES), "hidden_size": hidden_size},
            "feature_config": asdict(feature_cfg),
            "feature_names": FEATURE_NAMES,
            "feature_mean": mean.astype(np.float32).tolist(),
            "feature_std": std.astype(np.float32).tolist(),
            "train_accuracy": accuracy,
            "loss": losses[-1] if losses else None,
            "clean_file_count": len(clean_files),
            "crackle_file_count": len(crackle_files),
            "train_window_count": int(features.shape[0]),
            "device_requested": requested_device,
            "device": effective_device,
        },
    )
    write_json(
        out / "training_summary.json",
        {
            "checkpoint": str(out / "best_model.pt"),
            "loss": losses[-1] if losses else None,
            "train_accuracy": accuracy,
            "clean_file_count": len(clean_files),
            "crackle_file_count": len(crackle_files),
            "train_window_count": int(features.shape[0]),
            "device_requested": requested_device,
            "device": effective_device,
            "clean_files": clean_meta,
            "crackle_files": crackle_meta,
        },
    )
    return {
        "checkpoint": str(out / "best_model.pt"),
        "loss": losses[-1] if losses else None,
        "train_accuracy": accuracy,
        "clean_file_count": len(clean_files),
        "crackle_file_count": len(crackle_files),
        "train_window_count": int(features.shape[0]),
        "device_requested": requested_device,
        "device": effective_device,
    }


def _load_classifier(checkpoint_path: str | Path, device: str = "auto") -> tuple[
    CrackleFeatureClassifier,
    CrackleFeatureConfig,
    np.ndarray,
    np.ndarray,
    str,
    str,
]:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = dict(state["model_config"])
    model = CrackleFeatureClassifier(
        int(model_cfg.get("input_dim", len(FEATURE_NAMES))),
        hidden_size=int(model_cfg.get("hidden_size", 48)),
    )
    model.load_state_dict(state["model"])
    requested, effective = resolve_torch_device(device)
    model.to(torch.device(effective)).eval()
    feature_cfg = CrackleFeatureConfig(**dict(state["feature_config"]))
    mean = np.asarray(state["feature_mean"], dtype=np.float32)
    std = np.asarray(state["feature_std"], dtype=np.float32)
    return model, feature_cfg, mean, std, requested, effective


def _predict_crackle_file_loaded(
    path: str | Path,
    model: CrackleFeatureClassifier,
    cfg: CrackleFeatureConfig,
    mean: np.ndarray,
    std: np.ndarray,
    requested: str,
    effective: str,
) -> dict[str, Any]:
    waveform, sample_rate = load_audio(path, cfg.sample_rate, "mono")
    features, times = extract_crackle_features(waveform, sample_rate, cfg)
    if features.shape[0] == 0:
        probs = np.zeros(1, dtype=np.float32)
        times = np.zeros(1, dtype=np.float32)
    else:
        x = torch.from_numpy(_normalize(features, mean, std)).float().to(torch.device(effective))
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).detach().cpu().numpy()
    top_idx = int(np.argmax(probs))
    active_ratio = float(np.mean(probs >= 0.65))
    return {
        "path": str(path),
        "duration_sec": float(waveform.shape[-1] / float(sample_rate)),
        "num_windows": int(len(probs)),
        "crackle_prob_mean": float(np.mean(probs)),
        "crackle_prob_p95": float(np.percentile(probs, 95)),
        "crackle_prob_max": float(np.max(probs)),
        "crackle_window_ratio": active_ratio,
        "top_window_time_sec": float(times[top_idx]) if len(times) else 0.0,
        "device_requested": requested,
        "device": effective,
    }


def predict_crackle_file(
    path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    model, cfg, mean, std, requested, effective = _load_classifier(checkpoint_path, device)
    return _predict_crackle_file_loaded(path, model, cfg, mean, std, requested, effective)


def _label(prob: float, threshold: float, uncertain_low: float, uncertain_high: float) -> str:
    if prob >= threshold:
        return "crackle"
    if uncertain_low <= prob < uncertain_high:
        return "uncertain"
    return "clean"


def _crackle_score(row: dict[str, Any], score_stat: str) -> float:
    mean = float(row.get("crackle_prob_mean") or 0.0)
    p95 = float(row.get("crackle_prob_p95") or 0.0)
    max_prob = float(row.get("crackle_prob_max") or 0.0)
    ratio = float(row.get("crackle_window_ratio") or 0.0)
    if score_stat == "mean":
        return mean
    if score_stat == "max":
        return max_prob
    if score_stat == "p95":
        return p95
    if score_stat == "event":
        return p95 * math.sqrt(max(ratio, 0.0))
    raise ValueError(f"Unknown crackle score stat: {score_stat}")


def _copy_review_files(rows: list[dict[str, Any]], output_dir: Path, copy_top_n: int) -> None:
    if copy_top_n <= 0:
        return
    review_root = ensure_dir(output_dir / "review")
    grouped = {
        "crackle": sorted(rows, key=lambda r: float(r["crackle_score"]), reverse=True),
        "clean": sorted(rows, key=lambda r: float(r["crackle_score"])),
        "uncertain": sorted(
            [r for r in rows if r["pseudo_label"] == "uncertain"],
            key=lambda r: abs(float(r["crackle_score"]) - 0.5),
        ),
    }
    for label, selected in grouped.items():
        dest_dir = ensure_dir(review_root / label)
        for idx, row in enumerate(selected[:copy_top_n], start=1):
            src = Path(str(row["path"]))
            suffix = src.suffix
            dest = dest_dir / f"{idx:04d}_{float(row['crackle_score']):.3f}_{src.stem}{suffix}"
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)


def pseudolabel_crackle_dataset(
    inputs: list[str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    threshold: float = 0.65,
    uncertain_low: float = 0.45,
    uncertain_high: float = 0.65,
    score_stat: str = "event",
    max_files: int | None = None,
    copy_top_n: int = 0,
) -> dict[str, Any]:
    score_stat = str(score_stat)
    if score_stat not in {"event", "mean", "p95", "max"}:
        raise ValueError("score_stat must be one of: event, mean, p95, max")
    out = ensure_dir(output_dir)
    files = discover_audio_files(inputs, max_files=max_files)
    if max_files is not None:
        files = files[: int(max_files)]
    rows = []
    model, cfg, mean, std, requested, effective = _load_classifier(checkpoint_path, device)
    for path in files:
        try:
            row = _predict_crackle_file_loaded(path, model, cfg, mean, std, requested, effective)
            row["error"] = None
        except Exception as exc:
            row = {
                "path": str(path),
                "duration_sec": None,
                "num_windows": 0,
                "crackle_prob_mean": None,
                "crackle_prob_p95": None,
                "crackle_prob_max": None,
                "crackle_window_ratio": None,
                "top_window_time_sec": None,
                "device_requested": requested,
                "device": effective,
                "error": str(exc),
            }
        score = _crackle_score(row, score_stat)
        row["crackle_score"] = score
        row["crackle_score_stat"] = score_stat
        row["pseudo_label"] = _label(score, threshold, uncertain_low, uncertain_high)
        rows.append(row)
    rows.sort(key=lambda r: float(r["crackle_score"] or -1.0), reverse=True)
    csv_path = out / "pseudolabels.csv"
    fieldnames = [
        "path",
        "pseudo_label",
        "crackle_score",
        "crackle_score_stat",
        "crackle_window_ratio",
        "crackle_prob_p95",
        "crackle_prob_max",
        "crackle_prob_mean",
        "top_window_time_sec",
        "duration_sec",
        "num_windows",
        "device",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    _copy_review_files(rows, out, copy_top_n)
    summary = {
        "checkpoint": str(checkpoint_path),
        "input_count": len(files),
        "output_csv": str(csv_path),
        "score_stat": score_stat,
        "threshold": threshold,
        "uncertain_low": uncertain_low,
        "uncertain_high": uncertain_high,
        "label_counts": {
            label: sum(1 for row in rows if row["pseudo_label"] == label)
            for label in ("crackle", "uncertain", "clean")
        },
        "copy_top_n": copy_top_n,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "pseudolabels.json", rows)
    return summary
