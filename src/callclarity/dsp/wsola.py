from __future__ import annotations

import numpy as np
import torch


def _linear_time_scale(waveform: torch.Tensor, tempo: float) -> torch.Tensor:
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    n = int(waveform.shape[-1])
    if n == 0:
        return waveform
    out_n = max(1, int(round(n / float(tempo))))
    if out_n == n:
        return waveform.clone()
    src_x = np.linspace(0.0, 1.0, n, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, out_n, endpoint=True)
    data = waveform.detach().cpu().numpy()
    out = np.stack([np.interp(dst_x, src_x, ch).astype(np.float32) for ch in data], axis=0)
    return torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype)


def wsola_time_scale(
    waveform: torch.Tensor,
    sample_rate: int,
    tempo: float,
    frame_ms: float = 40.0,
    analysis_hop_ms: float = 10.0,
    search_ms: float = 15.0,
    crossfade_ms: float = 8.0,
) -> torch.Tensor:
    """Small WSOLA-style TSM.

    For very short chunks this falls back to interpolation; for longer blocks it uses
    correlation-guided overlap-add. It is intentionally compact for POC benchmarking.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    n = int(waveform.shape[-1])
    frame = max(8, int(round(sample_rate * frame_ms / 1000.0)))
    hop_out = max(1, int(round(sample_rate * analysis_hop_ms / 1000.0)))
    search = max(0, int(round(sample_rate * search_ms / 1000.0)))
    crossfade = max(1, int(round(sample_rate * crossfade_ms / 1000.0)))
    if n < frame * 2 or abs(tempo - 1.0) < 1e-4:
        return _linear_time_scale(waveform, tempo)

    data = waveform.detach().cpu().numpy().astype(np.float32)
    out_len = max(1, int(round(n / tempo)))
    out = np.zeros((data.shape[0], out_len + frame), dtype=np.float32)
    weight = np.zeros(out_len + frame, dtype=np.float32)
    first = data[:, :frame]
    out[:, :frame] += first
    weight[:frame] += 1.0
    read_pos = 0.0
    write_pos = hop_out
    prev_read = 0
    while write_pos < out_len:
        predicted = int(round(read_pos + hop_out * tempo))
        lo = max(0, predicted - search)
        hi = min(n - frame, predicted + search)
        best = max(0, min(predicted, n - frame))
        if hi > lo and write_pos >= crossfade:
            target = out[:, int(write_pos) - crossfade : int(write_pos)]
            best_score = -np.inf
            for cand in range(lo, hi + 1, max(1, search // 12 or 1)):
                seg = data[:, cand : cand + crossfade]
                denom = np.linalg.norm(target) * np.linalg.norm(seg) + 1e-8
                score = float(np.sum(target * seg) / denom)
                if score > best_score:
                    best_score = score
                    best = cand
        frame_data = data[:, best : best + frame]
        end = int(write_pos) + frame_data.shape[-1]
        if end > out.shape[-1]:
            frame_data = frame_data[:, : out.shape[-1] - int(write_pos)]
            end = out.shape[-1]
        fade = np.ones(frame_data.shape[-1], dtype=np.float32)
        of = min(crossfade, frame_data.shape[-1])
        fade[:of] = np.linspace(0.0, 1.0, of, dtype=np.float32)
        out[:, int(write_pos) : end] += frame_data * fade[None, :]
        weight[int(write_pos) : end] += fade
        prev_read = best
        read_pos = float(prev_read)
        write_pos += hop_out
    result = out[:, :out_len] / np.maximum(weight[:out_len], 1e-6)[None, :]
    return torch.from_numpy(result).to(device=waveform.device, dtype=waveform.dtype)
