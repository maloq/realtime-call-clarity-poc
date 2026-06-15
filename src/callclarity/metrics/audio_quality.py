from __future__ import annotations

from typing import Any

import torch

from callclarity.dsp.envelope import rms_dbfs
from callclarity.io.audio_io import peak_dbfs, resample_if_needed
from callclarity.metrics.optional_quality import OPTIONAL_QUALITY_KEYS, optional_no_reference_quality


def _prefix_keys(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def _numeric_delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    try:
        return float(after) - float(before)
    except Exception:
        return None


def no_reference_quality(
    raw: torch.Tensor,
    processed: torch.Tensor,
    sample_rate: int | None = None,
    metric_cfg: Any | None = None,
    processed_sample_rate: int | None = None,
) -> dict[str, Any]:
    processed_sr = int(processed_sample_rate or sample_rate or 0)
    if sample_rate is not None and processed_sr and int(sample_rate) != processed_sr:
        processed_aligned = resample_if_needed(processed, processed_sr, int(sample_rate))
    else:
        processed_aligned = processed
    n = min(raw.shape[-1], processed_aligned.shape[-1])
    raw_aligned = raw[..., :n]
    processed_aligned = processed_aligned[..., :n]
    residual = processed_aligned - raw_aligned
    raw_rms = rms_dbfs(raw)
    proc_rms = rms_dbfs(processed)
    residual_rms = rms_dbfs(residual) if residual.numel() else None
    row: dict[str, Any] = {
        "raw_rms_dbfs": raw_rms,
        "processed_rms_dbfs": proc_rms,
        "rms_delta_db": proc_rms - raw_rms,
        "residual_rms_dbfs": residual_rms,
        "processed_peak_dbfs": peak_dbfs(processed),
        "sample_clipping_count": int((processed.abs() >= 1.0).sum().item()),
        "stoi": None,
        "si_sdr": None,
    }
    row.update(dict(OPTIONAL_QUALITY_KEYS))
    if sample_rate is None:
        return row

    raw_optional = optional_no_reference_quality(raw, sample_rate, metric_cfg)
    processed_optional = optional_no_reference_quality(
        processed,
        processed_sr or int(sample_rate),
        metric_cfg,
    )
    row.update(processed_optional)
    row.update(_prefix_keys(raw_optional, "raw"))
    for key in OPTIONAL_QUALITY_KEYS:
        if key == "plcmos":
            continue
        row[f"{key}_delta"] = _numeric_delta(processed_optional.get(key), raw_optional.get(key))
    row["stoi"] = row.get("squim_stoi_est")
    row["si_sdr"] = row.get("squim_si_sdr_est")
    return row
