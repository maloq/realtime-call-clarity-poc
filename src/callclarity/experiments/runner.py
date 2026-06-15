from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from callclarity.data.manifests import build_manifest_from_config
from callclarity.io.audio_io import load_audio
from callclarity.metrics.aggregate import aggregate_run
from callclarity.metrics.audio_quality import no_reference_quality
from callclarity.metrics.guardrails import evaluate_guardrails
from callclarity.metrics.latency_metrics import flatten_latency_summary
from callclarity.metrics.leveling_metrics import aggregate_leveling
from callclarity.metrics.rate_metrics import aggregate_slowdown
from callclarity.reports.report_writer import EvalReportWriter, write_selected_samples_csv
from callclarity.reports.sample_export import export_sample
from callclarity.streaming.pipeline import Pipeline
from callclarity.streaming.realtime_simulator import RealtimeSimulator, SimulationResult
from callclarity.utils.files import ensure_dir


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    try:
        return obj.get(key, default)
    except Exception:
        return getattr(obj, key, default)


def _stage_device(stage_metric_rows: list[dict[str, Any]]) -> str:
    devices = []
    for row in stage_metric_rows:
        for key in (
            "deepfilternet_device",
            "neural_decrackle_device",
            "ap_bwe_device",
            "flashsr_device",
        ):
            value = row.get(key)
            if value and value not in devices:
                devices.append(str(value))
    if not devices:
        return "cpu"
    return "mixed" if len(devices) > 1 else devices[0]


def _latency_device(method_device: str, gpu_synchronized: bool) -> str:
    if str(method_device).startswith("cuda"):
        return f"{method_device}:{'synchronized' if gpu_synchronized else 'unsynchronized'}"
    if method_device == "mixed":
        return f"mixed:{'synchronized' if gpu_synchronized else 'unsynchronized'}"
    return "cpu"


def _runtime_bool(cfg: Any, key: str, default: bool = False) -> bool:
    return bool(_cfg_get(getattr(cfg, "runtime", {}), key, default))


def _runtime_int(cfg: Any, key: str, default: int = 1) -> int:
    value = _cfg_get(getattr(cfg, "runtime", {}), key, default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _run_waveform(
    cfg: Any,
    waveform,
    sample_rate: int,
    metadata: dict[str, Any],
) -> SimulationResult:
    pipeline = Pipeline.from_config(cfg.pipeline)
    pipeline.synchronize_gpu_timing = bool(
        _cfg_get(getattr(cfg, "latency", {}), "synchronize_gpu_for_timing", False)
    )
    chunk_ms = float(cfg.pipeline.get("chunk_ms", cfg.audio.chunk_ms))
    max_added_latency_ms = float(
        cfg.pipeline.get("max_added_latency_ms", cfg.latency.max_added_latency_ms)
    )
    simulator = RealtimeSimulator(
        pipeline,
        chunk_ms=chunk_ms,
        max_added_latency_ms=max_added_latency_ms,
    )
    return simulator.run(
        waveform,
        sample_rate,
        stream_id=metadata.get("recording_id", "default"),
        metadata=metadata,
    )


def _eval_manifest_row(
    cfg: Any,
    output_dir: Path,
    row: dict[str, Any],
    idx: int,
    total_rows: int,
    sample_limit: int,
    print_file_timing: bool,
) -> dict[str, Any]:
    file_start = perf_counter()
    waveform, sample_rate = load_audio(
        row["audio_path"],
        int(cfg.audio.sample_rate),
        str(cfg.audio.channels),
    )
    metadata = {
        "recording_id": row["recording_id"],
        "transcript": row.get("transcript", ""),
    }
    sim = _run_waveform(cfg, waveform, sample_rate, metadata)
    metric_cfg = getattr(getattr(cfg, "metrics", None), "no_reference", None)
    quality = no_reference_quality(
        waveform,
        sim.output,
        sample_rate,
        metric_cfg,
        processed_sample_rate=sim.sample_rate,
    )
    latency = flatten_latency_summary(sim.latency_summary)
    leveling = aggregate_leveling(sim.stage_metric_rows)
    slowdown = aggregate_slowdown(
        sim.events,
        waveform.shape[-1],
        sim.output.shape[-1],
        input_sample_rate=sample_rate,
        output_sample_rate=sim.sample_rate,
    )
    operational = sim.operational_summary
    gpu_synchronized = bool(
        _cfg_get(getattr(cfg, "latency", {}), "synchronize_gpu_for_timing", False)
    )
    method_device = _stage_device(sim.stage_metric_rows)
    file_row = {
        "recording_id": row["recording_id"],
        "audio_path": row["audio_path"],
        "input_duration_sec": waveform.shape[-1] / float(sample_rate),
        "output_duration_sec": sim.output.shape[-1] / float(sim.sample_rate),
        "input_sample_rate": int(sample_rate),
        "output_sample_rate": int(sim.sample_rate),
        "preset": str(cfg.pipeline.name),
        "latency_budget_ms": float(
            cfg.pipeline.get("max_added_latency_ms", cfg.latency.max_added_latency_ms)
        ),
        "runtime_device": str(_cfg_get(getattr(cfg, "runtime", {}), "device", "cpu")),
        "method_device": method_device,
        "metric_device": quality.get("metric_device", "none"),
        "metric_device_requested": quality.get("metric_device_requested", "none"),
        "gpu_latency_synchronized": gpu_synchronized,
        "latency_device": _latency_device(method_device, gpu_synchronized),
        **latency,
        **quality,
        **operational,
        **leveling,
        **slowdown,
    }
    guardrail_cfg = getattr(getattr(cfg, "metrics", None), "guardrails", None)
    guardrails = evaluate_guardrails(file_row, guardrail_cfg)
    file_row["guardrail_warning_count"] = sum(
        1 for warning in guardrails if warning.get("severity") != "error"
    )
    file_row["guardrail_error_count"] = sum(
        1 for warning in guardrails if warning.get("severity") == "error"
    )
    file_row["guardrails"] = "; ".join(warning["rule"] for warning in guardrails)

    events = []
    for event in sim.events:
        event.setdefault("recording_id", row["recording_id"])
        events.append(event)
    stage_latency = [
        {"recording_id": row["recording_id"], **stage}
        for stage in sim.stage_latency_rows
    ]
    per_chunk = [
        {"recording_id": row["recording_id"], **chunk_row}
        for chunk_row in sim.per_chunk_rows
    ]
    guardrail_rows = [
        {"recording_id": row["recording_id"], **warning}
        for warning in guardrails
    ]
    sample_info = None
    if idx < sample_limit:
        sample_info = export_sample(
            output_dir / "samples",
            row["recording_id"],
            waveform,
            sim.output,
            sample_rate,
            file_row,
            processed_sample_rate=sim.sample_rate,
        )
    timing_line = None
    if print_file_timing:
        elapsed_sec = perf_counter() - file_start
        input_duration_sec = waveform.shape[-1] / float(sample_rate)
        rtf = elapsed_sec / max(input_duration_sec, 1e-12)
        timing_line = (
            "[eval] "
            f"{idx + 1}/{total_rows} "
            f"{cfg.pipeline.name} "
            f"{row['recording_id']} "
            f"audio={input_duration_sec:.2f}s "
            f"wall={elapsed_sec:.2f}s "
            f"rtf={rtf:.3f}"
        )
    return {
        "idx": idx,
        "file_row": file_row,
        "events": events,
        "stage_latency": stage_latency,
        "per_chunk": per_chunk,
        "guardrails": guardrail_rows,
        "sample_info": sample_info,
        "timing_line": timing_line,
    }


def run_file(
    cfg: Any,
    input_path: str | Path,
    transcript_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    transcript = (
        Path(transcript_path).read_text(encoding="utf-8").strip() if transcript_path else ""
    )
    waveform, sample_rate = load_audio(
        input_path,
        int(cfg.audio.sample_rate),
        str(cfg.audio.channels),
    )
    result = _run_waveform(
        cfg,
        waveform,
        sample_rate,
        {"recording_id": Path(input_path).stem, "transcript": transcript},
    )
    from callclarity.io.audio_io import write_wav
    from callclarity.utils.files import write_json, write_jsonl

    write_wav(output_dir / "raw.wav", waveform, sample_rate)
    write_wav(output_dir / "processed.wav", result.output, result.sample_rate)
    write_json(output_dir / "latency_summary.json", result.latency_summary)
    write_jsonl(output_dir / "events.jsonl", result.events)
    metric_cfg = getattr(getattr(cfg, "metrics", None), "no_reference", None)
    quality = no_reference_quality(
        waveform,
        result.output,
        sample_rate,
        metric_cfg,
        processed_sample_rate=result.sample_rate,
    )
    metrics = {
        **flatten_latency_summary(result.latency_summary),
        **quality,
        **result.operational_summary,
    }
    write_json(output_dir / "metrics_summary.json", metrics)
    return metrics


def run_eval(
    cfg: Any,
    output_dir: str | Path,
    manifest_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    rows = (
        list(manifest_rows)
        if manifest_rows is not None
        else build_manifest_from_config(cfg.data, Path(output_dir) / "manifest_eval.jsonl")
    )
    per_file: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_stage_latency: list[dict[str, Any]] = []
    all_per_chunk: list[dict[str, Any]] = []
    all_guardrails: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    sample_limit = int(cfg.sample_selector.num_examples)
    print_file_timing = bool(_cfg_get(getattr(cfg, "runtime", {}), "print_file_timing", False))
    parallel_eval = _runtime_bool(cfg, "parallel_eval", False)
    num_workers = max(1, _runtime_int(cfg, "num_workers", 1))
    workers = min(num_workers, len(rows)) if parallel_eval else 1
    results: list[dict[str, Any]] = []
    if workers > 1:
        print(
            f"[eval] parallel workers={workers} preset={cfg.pipeline.name} files={len(rows)}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _eval_manifest_row,
                    cfg,
                    Path(output_dir),
                    row,
                    idx,
                    len(rows),
                    sample_limit,
                    print_file_timing,
                )
                for idx, row in enumerate(rows)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result["timing_line"]:
                    print(result["timing_line"], flush=True)
                results.append(result)
        results.sort(key=lambda item: int(item["idx"]))
    else:
        for idx, row in enumerate(rows):
            result = _eval_manifest_row(
                cfg,
                Path(output_dir),
                row,
                idx,
                len(rows),
                sample_limit,
                print_file_timing,
            )
            if result["timing_line"]:
                print(result["timing_line"], flush=True)
            results.append(result)

    for result in results:
        per_file.append(result["file_row"])
        all_events.extend(result["events"])
        all_stage_latency.extend(result["stage_latency"])
        all_per_chunk.extend(result["per_chunk"])
        all_guardrails.extend(result["guardrails"])
        if result["sample_info"] is not None:
            sample_rows.append(result["sample_info"])
    summary = aggregate_run(
        run_id=Path(output_dir).name,
        pipeline_name=str(cfg.pipeline.name),
        dataset_name=str(cfg.data.name),
        per_file_rows=per_file,
    )
    writer = EvalReportWriter(output_dir)
    writer.write_run(
        cfg,
        rows,
        per_file,
        summary,
        all_stage_latency,
        all_events,
        all_per_chunk,
        all_guardrails,
    )
    write_selected_samples_csv(Path(output_dir) / "samples", sample_rows)
    return summary


def benchmark_latency(cfg: Any, output_dir: str | Path) -> dict[str, Any]:
    return run_eval(cfg, output_dir)


def make_samples(cfg: Any, output_dir: str | Path) -> dict[str, Any]:
    return run_eval(cfg, output_dir)
