from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def _mean_or_none(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _common_or_mixed(values: list[Any], default: str = "none") -> str:
    vals = [str(v) for v in values if v not in (None, "")]
    if not vals:
        return default
    unique = sorted(set(vals))
    return unique[0] if len(unique) == 1 else "mixed"


def aggregate_run(
    run_id: str,
    pipeline_name: str,
    dataset_name: str,
    per_file_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    hours = sum(float(r.get("input_duration_sec", 0.0)) for r in per_file_rows) / 3600.0
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "pipeline_name": pipeline_name,
        "dataset_name": dataset_name,
        "num_files": len(per_file_rows),
        "audio_duration_hours": hours,
        "devices": {
            "runtime_device": _common_or_mixed([r.get("runtime_device") for r in per_file_rows], "cpu"),
            "method_device": _common_or_mixed([r.get("method_device") for r in per_file_rows], "cpu"),
            "metric_device": _common_or_mixed([r.get("metric_device") for r in per_file_rows], "none"),
            "metric_device_requested": _common_or_mixed(
                [r.get("metric_device_requested") for r in per_file_rows], "none"
            ),
            "latency_device": _common_or_mixed([r.get("latency_device") for r in per_file_rows], "cpu"),
            "gpu_latency_synchronized": any(bool(r.get("gpu_latency_synchronized")) for r in per_file_rows),
        },
        "latency": {
            "rtf_mean": _mean_or_none([r.get("rtf") for r in per_file_rows]) or 0.0,
            "chunk_ms_p50": _mean_or_none([r.get("chunk_ms_p50") for r in per_file_rows]) or 0.0,
            "chunk_ms_p95": _mean_or_none([r.get("chunk_ms_p95") for r in per_file_rows]) or 0.0,
            "chunk_ms_p99": _mean_or_none([r.get("chunk_ms_p99") for r in per_file_rows]) or 0.0,
            "total_added_latency_ms_p95": _mean_or_none([r.get("total_added_latency_ms_p95") for r in per_file_rows]) or 0.0,
            "total_added_latency_ms_p99": _mean_or_none([r.get("total_added_latency_ms_p99") for r in per_file_rows]) or 0.0,
            "max_dynamic_slowdown_buffer_ms": max([float(r.get("max_dynamic_slowdown_buffer_ms", 0.0)) for r in per_file_rows] or [0.0]),
            "budget_violation_count": int(sum(int(r.get("budget_violation_count", 0)) for r in per_file_rows)),
        },
        "quality": {
            "nisqa_mos_mean": _mean_or_none([r.get("nisqa_mos") for r in per_file_rows]),
            "nisqa_noisiness_mean": _mean_or_none([r.get("nisqa_noisiness") for r in per_file_rows]),
            "nisqa_coloration_mean": _mean_or_none([r.get("nisqa_coloration") for r in per_file_rows]),
            "nisqa_discontinuity_mean": _mean_or_none([r.get("nisqa_discontinuity") for r in per_file_rows]),
            "nisqa_loudness_mean": _mean_or_none([r.get("nisqa_loudness") for r in per_file_rows]),
            "dnsmos_p808_mean": _mean_or_none([r.get("dnsmos_p808") for r in per_file_rows]),
            "dnsmos_sig_mean": _mean_or_none([r.get("dnsmos_sig") for r in per_file_rows]),
            "dnsmos_bak_mean": _mean_or_none([r.get("dnsmos_bak") for r in per_file_rows]),
            "dnsmos_ovrl_mean": _mean_or_none([r.get("dnsmos_ovrl") for r in per_file_rows]),
            "squim_pesq_est_mean": _mean_or_none([r.get("squim_pesq_est") for r in per_file_rows]),
            "squim_stoi_est_mean": _mean_or_none([r.get("squim_stoi_est") for r in per_file_rows]),
            "squim_si_sdr_est_mean": _mean_or_none([r.get("squim_si_sdr_est") for r in per_file_rows]),
            "nisqa_mos_delta_mean": _mean_or_none([r.get("nisqa_mos_delta") for r in per_file_rows]),
            "dnsmos_ovrl_delta_mean": _mean_or_none([r.get("dnsmos_ovrl_delta") for r in per_file_rows]),
            "squim_stoi_est_delta_mean": _mean_or_none([r.get("squim_stoi_est_delta") for r in per_file_rows]),
            "stoi_mean": _mean_or_none([r.get("stoi") for r in per_file_rows]),
            "si_sdr_mean": _mean_or_none([r.get("si_sdr") for r in per_file_rows]),
            "wer_raw": None,
            "wer_processed": None,
        },
        "operational": {
            "input_clipping_pct_mean": _mean_or_none([r.get("input_clipping_pct") for r in per_file_rows]) or 0.0,
            "output_clipping_pct_mean": _mean_or_none([r.get("output_clipping_pct") for r in per_file_rows]) or 0.0,
            "dropout_count_total": int(sum(int(r.get("dropout_count", 0) or 0) for r in per_file_rows)),
            "discontinuity_count_total": int(sum(int(r.get("discontinuity_count", 0) or 0) for r in per_file_rows)),
            "zero_frame_count_total": int(sum(int(r.get("zero_frame_count", 0) or 0) for r in per_file_rows)),
            "repeated_frame_count_total": int(sum(int(r.get("repeated_frame_count", 0) or 0) for r in per_file_rows)),
            "decrackle_repaired_click_count_total": int(
                sum(int(r.get("decrackle_repaired_click_count", 0) or 0) for r in per_file_rows)
            ),
            "decrackle_repaired_samples_total": int(
                sum(int(r.get("decrackle_repaired_samples", 0) or 0) for r in per_file_rows)
            ),
            "narrowband_score_mean": _mean_or_none([r.get("narrowband_score") for r in per_file_rows]) or 0.0,
            "processing_rtf_mean": _mean_or_none([r.get("processing_rtf") for r in per_file_rows]) or 0.0,
            "added_latency_ms_max": max([float(r.get("added_latency_ms", 0.0) or 0.0) for r in per_file_rows] or [0.0]),
            "queue_underrun_count_total": int(sum(int(r.get("queue_underrun_count", 0) or 0) for r in per_file_rows)),
            "queue_overrun_count_total": int(sum(int(r.get("queue_overrun_count", 0) or 0) for r in per_file_rows)),
        },
        "guardrails": {
            "warning_count": int(sum(int(r.get("guardrail_warning_count", 0) or 0) for r in per_file_rows)),
            "error_count": int(sum(int(r.get("guardrail_error_count", 0) or 0) for r in per_file_rows)),
        },
        "leveling": {
            "speech_rms_dbfs_median": _mean_or_none([r.get("speech_rms_dbfs_median") for r in per_file_rows]),
            "target_error_db_abs_p90": _mean_or_none([r.get("target_error_db_abs_p90") for r in per_file_rows]),
            "speech_frames_within_3db_ratio": _mean_or_none([r.get("speech_frames_within_3db_ratio") for r in per_file_rows]),
            "silence_noise_boost_db_mean": _mean_or_none([r.get("silence_noise_boost_db_mean") for r in per_file_rows]),
            "clipping_count_total": int(sum(int(r.get("clipping_count_total", 0) or 0) for r in per_file_rows)),
        },
        "slowdown": {
            "slowdown_active_ratio": _mean_or_none([r.get("slowdown_active_ratio") for r in per_file_rows]) or 0.0,
            "average_active_tempo": _mean_or_none([r.get("average_active_tempo") for r in per_file_rows]),
            "min_tempo_used": _mean_or_none([r.get("min_tempo_used") for r in per_file_rows]),
            "hard_latency_limit_hit_count": int(sum(int(r.get("hard_latency_limit_hit_count", 0) or 0) for r in per_file_rows)),
            "output_input_duration_ratio": _mean_or_none([r.get("output_input_duration_ratio") for r in per_file_rows]) or 1.0,
        },
    }
