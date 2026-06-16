from __future__ import annotations

import json
import re
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from callclarity.reports.html_report import markdown_to_basic_html
from callclarity.reports.markdown_report import render_comparison_markdown
from callclarity.reports.plots import plot_key_metric_scorecard, plot_metric_bars
from callclarity.utils.files import ensure_dir, read_json, repo_relative_path, write_json


_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def _slugify(value: str, fallback: str = "sample", max_words: int = 8) -> str:
    text = value.strip().lower().translate(_CYRILLIC_TO_LATIN)
    words = re.findall(r"[a-z0-9]+", text)
    slug = "-".join(words[:max_words])
    return slug[:80].strip("-") or fallback


def _method_slug(value: str) -> str:
    return _slugify(value, fallback="method", max_words=12).replace("-", "_")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stage_method_summary(path: Path) -> dict[str, str]:
    summary = {
        "denoise_method": "none",
        "bandwidth_method": "none",
        "vad_method": "none",
        "leveler_method": "none",
        "rate_detector_method": "none",
        "slowdown_method": "none",
    }
    config_path = path / "config_resolved.yaml"
    if not config_path.exists():
        return summary
    try:
        cfg = OmegaConf.load(config_path)
        stages = OmegaConf.to_container(cfg.pipeline.stages, resolve=True)
    except Exception:
        return summary
    if not isinstance(stages, list):
        return summary
    stage_names: dict[str, list[str]] = {
        "denoise": [],
        "bandwidth": [],
        "vad": [],
        "leveler": [],
        "rate_detector": [],
        "slowdown": [],
    }
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_type = str(stage.get("type") or "")
        stage_name = str(stage.get("name") or "")
        if not stage_type or not stage_name:
            continue
        if stage_type in stage_names:
            stage_names[stage_type].append(stage_name)
        if stage_type == "enhance":
            stage_names["denoise"].append(stage_name)
    for stage_type, names in stage_names.items():
        key = f"{stage_type}_method"
        if names:
            summary[key] = "+".join(names)
    return summary


def _flatten_summary(path: Path) -> dict[str, Any]:
    summary = read_json(path / "metrics_summary.json")
    devices = summary.get("devices", {})
    stage_methods = _stage_method_summary(path)
    latency = summary.get("latency", {})
    quality = summary.get("quality", {})
    leveling = summary.get("leveling", {})
    operational = summary.get("operational", {})
    guardrails = summary.get("guardrails", {})
    slowdown = summary.get("slowdown", {})
    return {
        "run_id": summary.get("run_id", path.name),
        "pipeline_name": summary.get("pipeline_name"),
        "runtime_device": devices.get("runtime_device"),
        "method_device": devices.get("method_device"),
        "metric_device": devices.get("metric_device"),
        "latency_device": devices.get("latency_device"),
        "gpu_latency_synchronized": devices.get("gpu_latency_synchronized"),
        **stage_methods,
        "rtf_mean": latency.get("rtf_mean"),
        "chunk_ms_p95": latency.get("chunk_ms_p95"),
        "total_added_latency_ms_p95": latency.get("total_added_latency_ms_p95"),
        "total_added_latency_ms_p99": latency.get("total_added_latency_ms_p99"),
        "budget_violation_count": latency.get("budget_violation_count"),
        "dnsmos_sig_mean": quality.get("dnsmos_sig_mean"),
        "dnsmos_bak_mean": quality.get("dnsmos_bak_mean"),
        "dnsmos_ovrl_mean": quality.get("dnsmos_ovrl_mean"),
        "stoi_mean": quality.get("stoi_mean"),
        "si_sdr_mean": quality.get("si_sdr_mean"),
        "wer_processed": quality.get("wer_processed"),
        "speech_frames_within_3db_ratio": leveling.get("speech_frames_within_3db_ratio"),
        "target_error_db_abs_p90": leveling.get("target_error_db_abs_p90"),
        "output_clipping_pct_mean": operational.get("output_clipping_pct_mean"),
        "dropout_count_total": operational.get("dropout_count_total"),
        "discontinuity_count_total": operational.get("discontinuity_count_total"),
        "guardrail_warning_count": guardrails.get("warning_count"),
        "guardrail_error_count": guardrails.get("error_count"),
        "slowdown_active_ratio": slowdown.get("slowdown_active_ratio"),
        "max_dynamic_slowdown_buffer_ms": latency.get("max_dynamic_slowdown_buffer_ms"),
        "subjective_sample_dir": repo_relative_path(path / "samples"),
    }


def _load_run(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "metrics_summary.json")
    metrics_path = run_dir / "metrics_per_file.csv"
    samples_path = run_dir / "samples" / "selected_samples.csv"
    metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    samples = pd.read_csv(samples_path).to_dict("records") if samples_path.exists() else []
    manifest = _load_jsonl(run_dir / "manifest_eval.jsonl")
    return {
        "run_dir": run_dir,
        "method": str(summary.get("pipeline_name") or run_dir.name),
        "summary": summary,
        "metrics_by_id": {str(row.get("recording_id")): row for row in metrics},
        "samples_by_id": {str(row.get("recording_id")): row for row in samples},
        "manifest_by_id": {str(row.get("recording_id")): row for row in manifest},
    }


def _source_label(manifest_row: dict[str, Any] | None) -> str:
    if not manifest_row:
        return "sample"
    metadata = manifest_row.get("metadata") or {}
    source = str(metadata.get("source") or "")
    audio_path = str(manifest_row.get("audio_path") or "")
    if "test_data_samples" in source or "test_data_samples" in audio_path:
        return "test"
    if "asr_public_phone_calls" in source or "asr_public_phone_calls" in audio_path:
        return "dataset"
    return _slugify(source or Path(audio_path).parent.name, fallback="sample", max_words=3)


def _sample_slug(index: int, recording_id: str, manifest_row: dict[str, Any] | None) -> str:
    metadata = (manifest_row or {}).get("metadata") or {}
    explicit = str(metadata.get("display_name") or "").strip()
    transcript = str((manifest_row or {}).get("transcript") or "").strip()
    source = _source_label(manifest_row)
    text = explicit or transcript or recording_id
    readable = _slugify(text, fallback=recording_id, max_words=7)
    if source == "test" and readable.startswith("test-"):
        readable = readable.removeprefix("test-") or readable
    suffix = _slugify(recording_id, fallback=f"{index:02d}", max_words=3)
    if source == "test" or suffix == readable or suffix in readable:
        return f"{index:02d}_{source}_{readable}"
    return f"{index:02d}_{source}_{readable}_{suffix}"


def _compact_metric_row(
    method: str,
    metrics: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    latency_summary = summary.get("latency", {})
    slowdown_summary = summary.get("slowdown", {})
    return {
        "method": method,
        "input_duration_sec": metrics.get("input_duration_sec"),
        "output_duration_sec": metrics.get("output_duration_sec"),
        "rtf": metrics.get("rtf", latency_summary.get("rtf_mean")),
        "chunk_ms_p95": metrics.get("chunk_ms_p95", latency_summary.get("chunk_ms_p95")),
        "added_latency_ms_p95": metrics.get(
            "total_added_latency_ms_p95", latency_summary.get("total_added_latency_ms_p95")
        ),
        "dynamic_buffer_ms_max": metrics.get(
            "max_dynamic_slowdown_buffer_ms", latency_summary.get("max_dynamic_slowdown_buffer_ms")
        ),
        "budget_violations": metrics.get(
            "budget_violation_count",
            latency_summary.get("budget_violation_count"),
        ),
        "method_device": metrics.get("method_device"),
        "metric_device": metrics.get("metric_device"),
        "latency_device": metrics.get("latency_device"),
        "gpu_latency_synchronized": metrics.get("gpu_latency_synchronized"),
        "raw_rms_dbfs": metrics.get("raw_rms_dbfs"),
        "processed_rms_dbfs": metrics.get("processed_rms_dbfs"),
        "rms_delta_db": metrics.get("rms_delta_db"),
        "processed_peak_dbfs": metrics.get("processed_peak_dbfs"),
        "speech_rms_dbfs_median": metrics.get("speech_rms_dbfs_median"),
        "target_error_db_abs_p90": metrics.get("target_error_db_abs_p90"),
        "speech_frames_within_3db_ratio": metrics.get("speech_frames_within_3db_ratio"),
        "slowdown_active_ratio": metrics.get(
            "slowdown_active_ratio",
            slowdown_summary.get("slowdown_active_ratio"),
        ),
        "output_input_duration_ratio": metrics.get(
            "output_input_duration_ratio", slowdown_summary.get("output_input_duration_ratio")
        ),
        "decrackle_repaired_click_count": metrics.get("decrackle_repaired_click_count"),
        "decrackle_repaired_samples": metrics.get("decrackle_repaired_samples"),
        "clipping_count_total": metrics.get("clipping_count_total"),
    }


def compare_runs(run_dirs: list[str | Path], output_dir: str | Path) -> list[dict[str, Any]]:
    out = ensure_dir(output_dir)
    internal = ensure_dir(out / "_internal")
    plots_dir = ensure_dir(internal / "plots")
    run_contexts = [_load_run(Path(path)) for path in run_dirs]
    rows = [_flatten_summary(Path(path)) for path in run_dirs]
    for legacy_json in (out / "comparison.json", out / "samples_index.json"):
        if legacy_json.exists():
            legacy_json.unlink()
    sample_root = out / "samples"
    if sample_root.exists():
        shutil.rmtree(sample_root)
    ensure_dir(sample_root)
    internal_sample_root = internal / "samples"
    if internal_sample_root.exists():
        shutil.rmtree(internal_sample_root)
    ensure_dir(internal_sample_root)
    recording_ids: list[str] = []
    for context in run_contexts:
        for recording_id in context["samples_by_id"]:
            if recording_id not in recording_ids:
                recording_ids.append(recording_id)
    sample_index_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    for sample_index, recording_id in enumerate(recording_ids, start=1):
        manifest_row = next(
            (
                context["manifest_by_id"].get(recording_id)
                for context in run_contexts
                if recording_id in context["manifest_by_id"]
            ),
            None,
        )
        sample_name = _sample_slug(sample_index, recording_id, manifest_row)
        sample_dir = ensure_dir(sample_root / sample_name)
        internal_sample_dir = ensure_dir(internal_sample_root / sample_name)
        transcript = str((manifest_row or {}).get("transcript") or "").strip()
        actual_duration = next(
            (
                context["metrics_by_id"].get(recording_id, {}).get("input_duration_sec")
                for context in run_contexts
                if context["metrics_by_id"].get(recording_id, {}).get("input_duration_sec")
                is not None
            ),
            None,
        )
        if transcript:
            (sample_dir / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
        raw_dest = sample_dir / "raw.wav"
        metric_rows: list[dict[str, Any]] = []
        for context in run_contexts:
            method = context["method"]
            method_slug = _method_slug(method)
            sample = context["samples_by_id"].get(recording_id)
            metrics = context["metrics_by_id"].get(recording_id, {})
            if sample:
                raw_src = Path(str(sample.get("raw_path")))
                processed_src = Path(str(sample.get("processed_path")))
                if raw_src.exists() and not raw_dest.exists():
                    shutil.copy2(raw_src, raw_dest)
                processed_dest = sample_dir / f"{method_slug}.wav"
                if processed_src.exists():
                    shutil.copy2(processed_src, processed_dest)
                audio_rows.append(
                    {
                        "sample_name": sample_name,
                        "recording_id": recording_id,
                        "method": method,
                        "processed_wav": repo_relative_path(processed_dest)
                        if processed_dest.exists()
                        else None,
                    }
                )
            metric_rows.append(_compact_metric_row(method, metrics, context["summary"]))
        pd.DataFrame(metric_rows).to_csv(sample_dir / "metrics.csv", index=False)
        write_json(internal_sample_dir / "metrics.json", metric_rows)
        info = {
            "sample_name": sample_name,
            "recording_id": recording_id,
            "source": _source_label(manifest_row),
            "audio_path": (manifest_row or {}).get("audio_path"),
            "transcript": transcript,
            "metadata": (manifest_row or {}).get("metadata") or {},
            "raw_wav": repo_relative_path(raw_dest) if raw_dest.exists() else None,
            "methods": [context["method"] for context in run_contexts],
        }
        write_json(internal_sample_dir / "info.json", info)
        sample_index_rows.append(
            {
                "sample_name": sample_name,
                "recording_id": recording_id,
                "source": info["source"],
                "duration_sec": actual_duration or info["metadata"].get("duration_sec"),
                "transcript_preview": transcript[:120],
                "folder": repo_relative_path(sample_dir),
                "raw_wav": info["raw_wav"],
            }
        )
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(sample_index_rows).to_csv(out / "samples_index.csv", index=False)
    pd.DataFrame(audio_rows).to_csv(out / "method_audio.csv", index=False)
    write_json(internal / "comparison.json", rows)
    write_json(internal / "samples_index.json", sample_index_rows)
    key_plot_path = plots_dir / "method_key_metrics.png"
    plot_key_metric_scorecard(rows, key_plot_path)
    key_plot_rel = key_plot_path.relative_to(out).as_posix()
    markdown = render_comparison_markdown(rows, key_plot_rel)
    (out / "report.md").write_text(markdown, encoding="utf-8")
    (out / "report.html").write_text(
        markdown_to_basic_html(markdown, "Call Clarity Comparison"),
        encoding="utf-8",
    )
    readme = (
        "# Focused Audio Comparison\n\n"
        "Open `samples/` to inspect one folder per audio file. Each sample folder contains "
        "`raw.wav`, one processed WAV per method, transcripts when available, "
        "and `metrics.csv`.\n\n"
        "Open `report.md` or `report.html` for the summary and key-metrics plot. "
        "JSON/JSONL artifacts are kept under `_internal/`, including compact comparison indexes "
        "and per-sample `info.json` / `metrics.json` files.\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    plot_metric_bars(rows, "chunk_ms_p95", plots_dir / "metric_bars_latency.png")
    plot_metric_bars(rows, "dnsmos_sig_mean", plots_dir / "metric_bars_quality.png")
    plot_metric_bars(rows, "speech_frames_within_3db_ratio", plots_dir / "metric_bars_leveling.png")
    plot_metric_bars(rows, "slowdown_active_ratio", plots_dir / "metric_bars_slowdown.png")
    return rows
