from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from callclarity.io.audio_io import load_audio


AUDIO_EXTENSIONS = {".wav", ".opus", ".flac", ".mp3", ".ogg", ".m4a"}


def _is_archive(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return path.is_file() and suffixes.endswith((".tar.gz", ".tgz", ".tar"))


def discover_audio_files(
    paths: list[str | Path],
    extensions: set[str] | None = None,
    cache_dir: str | Path = "data/cache/crackle_audio",
    max_files: int | None = None,
) -> list[Path]:
    exts = {ext.lower() for ext in (extensions or AUDIO_EXTENSIONS)}
    files: list[Path] = []
    for value in paths:
        if max_files is not None and len(files) >= int(max_files):
            break
        root = Path(value)
        if root.is_file() and root.suffix.lower() in exts:
            files.append(root)
        elif _is_archive(root):
            from callclarity.data.manifests import discover_audio_transcript_pairs

            remaining = None if max_files is None else max(0, int(max_files) - len(files))
            archive_limit = remaining if remaining is not None else 1_000_000_000
            rows = discover_audio_transcript_pairs(
                root,
                extensions=sorted(exts),
                cache_dir=cache_dir,
                extract_archives=True,
                max_files=archive_limit,
            )
            files.extend(Path(row["audio_path"]) for row in rows)
        elif root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in exts:
                    files.append(path)
                    if max_files is not None and len(files) >= int(max_files):
                        break
    return sorted(dict.fromkeys(files))


def _random_segment(waveform: torch.Tensor, segment_samples: int, rng: random.Random) -> torch.Tensor:
    x = waveform.detach().float()
    if x.ndim == 2:
        x = x.mean(dim=0)
    if x.numel() >= segment_samples:
        start = rng.randint(0, int(x.numel()) - segment_samples)
        return x[start : start + segment_samples].clone()
    repeats = (segment_samples + int(x.numel()) - 1) // max(int(x.numel()), 1)
    if x.numel() == 0:
        return torch.zeros(segment_samples)
    return x.repeat(repeats)[:segment_samples].clone()


def _load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, _ = load_audio(path, sample_rate, "mono")
    return waveform.squeeze(0).float().clamp(-1.0, 1.0)


def _median_filter_1d(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    padded = torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
    return padded.unfold(0, kernel_size, 1).median(dim=-1).values


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def build_crackle_snippet_bank(
    crackle_files: list[Path],
    sample_rate: int,
    max_files: int = 24,
    max_snippets: int = 256,
    snippet_radius: int = 16,
) -> list[torch.Tensor]:
    snippets: list[torch.Tensor] = []
    for path in crackle_files[: max(0, max_files)]:
        try:
            x = _load_mono(path, sample_rate)
        except Exception:
            continue
        if x.numel() < snippet_radius * 2 + 1:
            continue
        residual = x - _median_filter_1d(x, 5)
        abs_res = residual.abs()
        threshold = max(float(torch.quantile(abs_res, 0.995).item()), 0.20)
        candidates = torch.nonzero(abs_res > threshold).flatten()
        if candidates.numel() == 0:
            continue
        step = max(1, int(candidates.numel()) // 24)
        for idx in candidates[::step]:
            center = int(idx.item())
            start = center - snippet_radius
            end = center + snippet_radius + 1
            if start < 0 or end > residual.numel():
                continue
            snippet = residual[start:end].clone()
            peak = float(snippet.abs().max().item())
            if peak <= 1e-5:
                continue
            snippets.append((snippet / peak).float())
            if len(snippets) >= max_snippets:
                return snippets
    return snippets


def add_synthetic_crackle(
    clean: torch.Tensor,
    sample_rate: int,
    cfg: Any,
    snippets: list[torch.Tensor] | None = None,
    rng: random.Random | None = None,
) -> torch.Tensor:
    rng = rng or random.Random()
    noisy = clean.clone()
    duration_sec = noisy.numel() / float(sample_rate)
    density = float(_cfg_get(cfg, "clicks_per_second", 18.0))
    min_amp = float(_cfg_get(cfg, "min_amplitude", 0.18))
    max_amp = float(_cfg_get(cfg, "max_amplitude", 0.95))
    max_click_ms = float(_cfg_get(cfg, "max_click_ms", 2.0))
    ringing_prob = float(_cfg_get(cfg, "ringing_probability", 0.35))
    count = max(1, int(round(duration_sec * density * rng.uniform(0.5, 1.5))))
    max_len = max(1, int(round(max_click_ms * sample_rate / 1000.0)))
    for _ in range(count):
        pos = rng.randrange(0, max(noisy.numel(), 1))
        amp = rng.uniform(min_amp, max_amp) * (-1.0 if rng.random() < 0.5 else 1.0)
        if snippets and rng.random() < 0.65:
            snippet = snippets[rng.randrange(len(snippets))]
            gain = abs(amp) * rng.uniform(0.6, 1.2)
            start = pos - snippet.numel() // 2
            end = start + snippet.numel()
            s0 = max(0, start)
            s1 = min(noisy.numel(), end)
            if s1 > s0:
                src0 = s0 - start
                noisy[s0:s1] += snippet[src0 : src0 + (s1 - s0)].to(noisy) * gain
            continue
        length = rng.randint(1, max_len)
        end = min(noisy.numel(), pos + length)
        if end <= pos:
            continue
        if length == 1:
            noisy[pos] += amp
        else:
            noisy[pos:end] += torch.linspace(amp, -0.35 * amp, end - pos, dtype=noisy.dtype)
        if rng.random() < ringing_prob and end < noisy.numel():
            ring_len = min(noisy.numel() - end, max_len * 3)
            t = torch.arange(ring_len, dtype=noisy.dtype) / float(sample_rate)
            ring = 0.25 * amp * torch.sin(2.0 * torch.pi * rng.uniform(1800.0, 5200.0) * t)
            ring *= torch.linspace(1.0, 0.0, ring_len, dtype=noisy.dtype)
            noisy[end : end + ring_len] += ring
    noise_floor = float(_cfg_get(cfg, "noise_floor", 0.003))
    if noise_floor > 0:
        noisy += torch.randn_like(noisy) * noise_floor
    return noisy.clamp(-1.0, 1.0)


class SyntheticCrackleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clean_files: list[Path],
        sample_rate: int,
        segment_samples: int,
        synthetic_cfg: Any,
        crackle_snippets: list[torch.Tensor] | None = None,
        length: int = 1000,
        seed: int = 1337,
    ) -> None:
        if not clean_files:
            raise ValueError("No clean audio files found. Provide WAV/Opus files in train.clean_dirs.")
        self.clean_files = clean_files
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(segment_samples)
        self.synthetic_cfg = synthetic_cfg
        self.crackle_snippets = crackle_snippets or []
        self.length = int(length)
        self.seed = int(seed)
        self._cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.length

    def _load_cached(self, path: Path) -> torch.Tensor:
        if path not in self._cache:
            self._cache[path] = _load_mono(path, self.sample_rate)
        return self._cache[path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + int(index) * 7919)
        path = self.clean_files[rng.randrange(len(self.clean_files))]
        clean = _random_segment(self._load_cached(path), self.segment_samples, rng)
        noisy = add_synthetic_crackle(clean, self.sample_rate, self.synthetic_cfg, self.crackle_snippets, rng)
        return noisy, clean
