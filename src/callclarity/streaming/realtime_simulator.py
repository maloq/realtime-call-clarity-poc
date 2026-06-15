from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from callclarity.metrics.operational import OperationalMetricsTracker
from callclarity.streaming.chunker import iter_audio_chunks, reconstruct_chunks
from callclarity.streaming.latency import LatencyTracker
from callclarity.streaming.pipeline import Pipeline


@dataclass
class SimulationResult:
    output: torch.Tensor
    sample_rate: int
    latency_summary: dict[str, Any]
    stage_latency_rows: list[dict[str, Any]]
    events: list[dict[str, Any]]
    per_chunk_rows: list[dict[str, Any]]
    stage_metric_rows: list[dict[str, Any]] = field(default_factory=list)
    operational_summary: dict[str, Any] = field(default_factory=dict)


class RealtimeSimulator:
    def __init__(
        self,
        pipeline: Pipeline,
        chunk_ms: float = 10.0,
        max_added_latency_ms: float = 200.0,
    ) -> None:
        self.pipeline = pipeline
        self.chunk_ms = float(chunk_ms)
        self.max_added_latency_ms = float(max_added_latency_ms)

    def run(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        stream_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        self.pipeline.reset()
        self.pipeline.warmup(sample_rate)
        tracker = LatencyTracker(
            max_added_latency_ms=self.max_added_latency_ms,
            algorithmic_latency_ms=self.pipeline.algorithmic_latency_ms,
        )
        output_chunks = []
        events: list[dict[str, Any]] = []
        per_chunk_rows: list[dict[str, Any]] = []
        stage_metric_rows: list[dict[str, Any]] = []
        op_tracker = OperationalMetricsTracker()
        for chunk_index, chunk in enumerate(
            iter_audio_chunks(
                waveform,
                sample_rate,
                self.chunk_ms,
                stream_id=stream_id,
                metadata=metadata,
            )
        ):
            result = self.pipeline.process(chunk)
            output_chunks.append(result.chunk)
            tracker.add_chunk(result.processing_time_ms, result.dynamic_buffer_ms)
            for metric in result.stage_metrics:
                tracker.add_stage(metric["stage"], metric["processing_time_ms"])
                stage_metric_rows.append(metric)
            events.extend(result.events)
            total_added_latency_ms = result.algorithmic_latency_ms + result.dynamic_buffer_ms
            operational = op_tracker.add_chunk(
                chunk,
                result.chunk,
                processing_time_ms=result.processing_time_ms,
                added_latency_ms=total_added_latency_ms,
                dynamic_buffer_ms=result.dynamic_buffer_ms,
                stage_metrics=result.stage_metrics,
            )
            per_chunk_rows.append(
                {
                    "chunk_index": chunk_index,
                    "start_time_sec": chunk.start_time_sec,
                    "input_samples": chunk.num_samples,
                    "output_samples": result.chunk.num_samples,
                    "processing_time_ms": result.processing_time_ms,
                    "dynamic_buffer_ms": result.dynamic_buffer_ms,
                    "total_added_latency_ms": total_added_latency_ms,
                    **operational,
                }
            )
        output = reconstruct_chunks(output_chunks)
        output_sample_rate = (
            int(output_chunks[-1].sample_rate) if output_chunks else int(sample_rate)
        )
        input_duration = waveform.shape[-1] / float(sample_rate)
        total_processing_sec = sum(row["processing_time_ms"] for row in per_chunk_rows) / 1000.0
        latency_summary = tracker.summary()
        latency_summary.update(
            {
                "total_audio_duration_sec": input_duration,
                "total_processing_wall_time_sec": total_processing_sec,
                "real_time_factor": total_processing_sec / max(input_duration, 1e-12),
                "max_added_latency_ms": self.max_added_latency_ms,
            }
        )
        return SimulationResult(
            output=output,
            sample_rate=output_sample_rate,
            latency_summary=latency_summary,
            stage_latency_rows=tracker.stage_summary(),
            events=events,
            per_chunk_rows=per_chunk_rows,
            stage_metric_rows=stage_metric_rows,
            operational_summary=op_tracker.summary(),
        )
