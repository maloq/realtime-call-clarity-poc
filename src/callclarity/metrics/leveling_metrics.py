from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_leveling(stage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    speech_rms = [float(r["speech_rms_dbfs"]) for r in stage_rows if r.get("speech_rms_dbfs") is not None]
    target_error = [abs(float(r["target_error_db"])) for r in stage_rows if r.get("target_error_db") is not None]
    gain = [float(r["current_gain_db"]) for r in stage_rows if r.get("current_gain_db") is not None]
    silence_boost = [float(r.get("silence_noise_boost_db", 0.0) or 0.0) for r in stage_rows]
    clipping = [int(r.get("clipping_count", 0) or 0) for r in stage_rows]
    limiter = [float(r.get("limiter_gain_reduction_db", 0.0) or 0.0) for r in stage_rows]
    return {
        "speech_rms_dbfs_median": float(np.median(speech_rms)) if speech_rms else None,
        "speech_rms_dbfs_p10": float(np.percentile(speech_rms, 10)) if speech_rms else None,
        "speech_rms_dbfs_p90": float(np.percentile(speech_rms, 90)) if speech_rms else None,
        "target_error_db_abs_p90": float(np.percentile(target_error, 90)) if target_error else None,
        "speech_frames_within_3db_ratio": float(np.mean(np.asarray(target_error) <= 3.0)) if target_error else None,
        "gain_db_p50": float(np.percentile(gain, 50)) if gain else None,
        "gain_db_p95": float(np.percentile(gain, 95)) if gain else None,
        "gain_db_max": float(np.max(gain)) if gain else None,
        "silence_noise_boost_db_mean": float(np.mean(silence_boost)) if silence_boost else None,
        "clipping_count_total": int(sum(clipping)),
        "limiter_gain_reduction_db_p95": float(np.percentile(limiter, 95)) if limiter else None,
    }
