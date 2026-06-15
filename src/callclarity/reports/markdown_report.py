from __future__ import annotations

from typing import Any


def _fmt(value: Any, precision: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def render_run_markdown(summary: dict[str, Any]) -> str:
    latency = summary.get("latency", {})
    quality = summary.get("quality", {})
    operational = summary.get("operational", {})
    guardrails = summary.get("guardrails", {})
    leveling = summary.get("leveling", {})
    slowdown = summary.get("slowdown", {})
    guardrail_counts = f"{guardrails.get('warning_count', 0)} / {guardrails.get('error_count', 0)}"
    return f"""# Realtime Call Clarity Report

Run: `{summary.get("run_id")}`

Pipeline: `{summary.get("pipeline_name")}`  
Dataset: `{summary.get("dataset_name")}`  
Files: {summary.get("num_files", 0)}  
Audio hours: {_fmt(summary.get("audio_duration_hours", 0.0), 4)}

## Latency

- Mean RTF: {_fmt(latency.get("rtf_mean", 0.0), 4)}
- Chunk p95: {_fmt(latency.get("chunk_ms_p95", 0.0), 3)} ms
- Added latency p95: {_fmt(latency.get("total_added_latency_ms_p95", 0.0), 3)} ms
- Added latency p99: {_fmt(latency.get("total_added_latency_ms_p99", 0.0), 3)} ms
- Budget violations: {latency.get("budget_violation_count", 0)}

## Quality

- NISQA MOS mean: {_fmt(quality.get("nisqa_mos_mean"))}
- DNSMOS OVRL mean: {_fmt(quality.get("dnsmos_ovrl_mean"))}
- DNSMOS SIG mean: {_fmt(quality.get("dnsmos_sig_mean"))}
- DNSMOS BAK mean: {_fmt(quality.get("dnsmos_bak_mean"))}
- SQUIM STOI estimate mean: {_fmt(quality.get("squim_stoi_est_mean"))}
- STOI mean: {_fmt(quality.get("stoi_mean"))}
- SI-SDR mean: {_fmt(quality.get("si_sdr_mean"))}
- Guardrail warnings/errors: {guardrail_counts}

## Operational

- Input clipping mean: {_fmt(operational.get("input_clipping_pct_mean"))}
- Output clipping mean: {_fmt(operational.get("output_clipping_pct_mean"))}
- Dropouts: {operational.get("dropout_count_total", 0)}
- Discontinuities: {operational.get("discontinuity_count_total", 0)}
- Narrowband score mean: {_fmt(operational.get("narrowband_score_mean"))}
- Decrackle repaired clicks: {operational.get("decrackle_repaired_click_count_total", 0)}
- Decrackle repaired samples: {operational.get("decrackle_repaired_samples_total", 0)}

## Leveling

- Median speech RMS: {_fmt(leveling.get("speech_rms_dbfs_median"))}
- P90 abs target error: {_fmt(leveling.get("target_error_db_abs_p90"))}
- Speech frames within 3 dB: {_fmt(leveling.get("speech_frames_within_3db_ratio"))}
- Clipping count: {leveling.get("clipping_count_total", 0)}

## Slowdown

- Active ratio: {_fmt(slowdown.get("slowdown_active_ratio", 0.0), 4)}
- Average active tempo: {_fmt(slowdown.get("average_active_tempo"))}
- Min tempo used: {_fmt(slowdown.get("min_tempo_used"))}
- Output/input duration ratio: {_fmt(slowdown.get("output_input_duration_ratio", 1.0), 4)}
"""


def render_comparison_markdown(
    rows: list[dict[str, Any]],
    key_plot_rel: str | None = None,
) -> str:
    lines = [
        "# Call Clarity Comparison",
        "",
    ]
    if key_plot_rel:
        lines.extend(
            [
                "## Key Metrics",
                "",
                f"![Method comparison on key metrics]({key_plot_rel})",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            "| run | pipeline | denoise | vad | leveler | rate | slowdown method | "
            "method device | metric device | latency device | rtf | chunk p95 | added p95 | "
            "DNSMOS OVRL | level within 3 dB | slowdown active |",
            "|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('run_id')} | {row.get('pipeline_name')} | "
            f"{row.get('denoise_method', 'none')} | "
            f"{row.get('vad_method', 'none')} | "
            f"{row.get('leveler_method', 'none')} | "
            f"{row.get('rate_detector_method', 'none')} | "
            f"{row.get('slowdown_method', 'none')} | "
            f"{row.get('method_device') or 'cpu'} | "
            f"{row.get('metric_device') or 'none'} | "
            f"{row.get('latency_device') or 'cpu'} | "
            f"{_fmt(row.get('rtf_mean'), 4)} | "
            f"{_fmt(row.get('chunk_ms_p95'), 2)} | "
            f"{_fmt(row.get('total_added_latency_ms_p95'), 2)} | "
            f"{_fmt(row.get('dnsmos_ovrl_mean'), 3)} | "
            f"{_fmt(row.get('speech_frames_within_3db_ratio'), 3)} | "
            f"{_fmt(row.get('slowdown_active_ratio'), 3)} |"
        )
    return "\n".join(lines) + "\n"
