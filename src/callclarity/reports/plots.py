from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from callclarity.utils.files import ensure_dir


KEY_COMPARISON_METRICS: tuple[dict[str, Any], ...] = (
    {"key": "rtf_mean", "label": "RTF", "higher_is_better": False},
    {"key": "chunk_ms_p95", "label": "Chunk p95 ms", "higher_is_better": False},
    {"key": "total_added_latency_ms_p95", "label": "Added p95 ms", "higher_is_better": False},
    {"key": "dnsmos_ovrl_mean", "label": "DNSMOS OVRL", "higher_is_better": True},
    {"key": "dnsmos_sig_mean", "label": "DNSMOS SIG", "higher_is_better": True},
    {"key": "dnsmos_bak_mean", "label": "DNSMOS BAK", "higher_is_better": True},
    {"key": "stoi_mean", "label": "STOI", "higher_is_better": True},
    {"key": "si_sdr_mean", "label": "SI-SDR", "higher_is_better": True},
    {"key": "wer_processed", "label": "WER", "higher_is_better": False},
    {
        "key": "speech_frames_within_3db_ratio",
        "label": "Level within 3 dB",
        "higher_is_better": True,
    },
    {"key": "target_error_db_abs_p90", "label": "Target error p90", "higher_is_better": False},
    {"key": "output_clipping_pct_mean", "label": "Clipping pct", "higher_is_better": False},
    {"key": "budget_violation_count", "label": "Budget violations", "higher_is_better": False},
    {"key": "slowdown_active_ratio", "label": "Slowdown active", "higher_is_better": False},
)


def plot_latency_hist(per_chunk_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    p = Path(output_path)
    ensure_dir(p.parent)
    values = [float(r.get("processing_time_ms", 0.0)) for r in per_chunk_rows]
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30)
    plt.xlabel("Processing time per chunk (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def plot_tempo(events: list[dict[str, Any]], output_path: str | Path) -> None:
    p = Path(output_path)
    ensure_dir(p.parent)
    decisions = [e for e in events if e.get("type") == "tempo_decision"]
    x = [float(e.get("timestamp_sec", 0.0)) for e in decisions]
    tempo = [float(e.get("tempo", 1.0)) for e in decisions]
    buffer = [float(e.get("buffer_ms", 0.0)) for e in decisions]
    plt.figure(figsize=(7, 4))
    if x:
        plt.plot(x, tempo, label="tempo")
        plt.plot(x, [b / 100.0 for b in buffer], label="buffer / 100 ms")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def plot_metric_bars(rows: list[dict[str, Any]], metric: str, output_path: str | Path) -> None:
    p = Path(output_path)
    ensure_dir(p.parent)
    labels = [str(r.get("run_id", idx)) for idx, r in enumerate(rows)]
    values = [float(r.get(metric) or 0.0) for r in rows]
    plt.figure(figsize=(max(6, len(rows) * 1.5), 4))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _format_cell(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_key_metric_scorecard(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    p = Path(output_path)
    ensure_dir(p.parent)
    labels = [
        str(row.get("run_id") or row.get("pipeline_name") or idx)
        for idx, row in enumerate(rows)
    ]
    metrics = []
    raw_columns: list[list[float | None]] = []
    score_columns: list[list[float]] = []
    for metric in KEY_COMPARISON_METRICS:
        raw_values = [_finite_float(row.get(str(metric["key"]))) for row in rows]
        finite_values = [value for value in raw_values if value is not None]
        if not finite_values:
            continue
        low = min(finite_values)
        high = max(finite_values)
        if high == low:
            scores = [1.0 if value is not None else np.nan for value in raw_values]
        elif bool(metric["higher_is_better"]):
            scores = [
                (value - low) / (high - low) if value is not None else np.nan
                for value in raw_values
            ]
        else:
            scores = [
                (high - value) / (high - low) if value is not None else np.nan
                for value in raw_values
            ]
        metrics.append(metric)
        raw_columns.append(raw_values)
        score_columns.append(scores)

    if not metrics:
        plt.figure(figsize=(7, 2.8))
        plt.text(0.5, 0.5, "No comparable numeric metrics", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        return

    score_matrix = np.array(score_columns, dtype=float).T
    raw_matrix = np.array(raw_columns, dtype=object).T
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    fig_width = max(8.0, len(metrics) * 1.25)
    fig_height = max(4.0, len(labels) * 0.48 + 1.8)
    plt.figure(figsize=(fig_width, fig_height))
    image = plt.imshow(
        np.ma.masked_invalid(score_matrix),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(image, label="Normalized score")
    plt.xticks(
        range(len(metrics)),
        [
            f"{metric['label']}\n{'higher' if metric['higher_is_better'] else 'lower'} is better"
            for metric in metrics
        ],
        rotation=25,
        ha="right",
    )
    plt.yticks(range(len(labels)), labels)
    plt.title("Method comparison on key metrics")
    for row_idx in range(score_matrix.shape[0]):
        for col_idx in range(score_matrix.shape[1]):
            score = score_matrix[row_idx, col_idx]
            raw_value = raw_matrix[row_idx, col_idx]
            text_color = "black" if not np.isfinite(score) or 0.25 < score < 0.75 else "white"
            plt.text(
                col_idx,
                row_idx,
                _format_cell(raw_value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
