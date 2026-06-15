from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from callclarity.registry import create_method
from callclarity.types import AudioChunk, ProcessResult, StreamingProcessor


@dataclass
class PipelineResult:
    chunk: AudioChunk
    stage_metrics: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    algorithmic_latency_ms: float = 0.0
    dynamic_buffer_ms: float = 0.0


class Pipeline:
    def __init__(self, processors: list[StreamingProcessor], name: str = "pipeline") -> None:
        self.processors = processors
        self.name = name
        self.synchronize_gpu_timing = False

    @classmethod
    def from_config(cls, pipeline_cfg: DictConfig | dict[str, Any]) -> "Pipeline":
        cfg = (
            OmegaConf.to_container(pipeline_cfg, resolve=True)
            if OmegaConf.is_config(pipeline_cfg)
            else pipeline_cfg
        )
        processors: list[StreamingProcessor] = []
        for stage in cfg["stages"]:
            config = stage.get("config") or {}
            processors.append(create_method(stage["type"], stage["name"], dict(config)))
        return cls(processors, name=str(cfg.get("name", "pipeline")))

    def reset(self) -> None:
        for processor in self.processors:
            processor.reset()

    def warmup(self, sample_rate: int) -> None:
        for processor in self.processors:
            processor.warmup(sample_rate)

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(sum(p.algorithmic_latency_ms for p in self.processors))

    def _sync_for_timing(self) -> None:
        if self.synchronize_gpu_timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    def process(self, chunk: AudioChunk) -> PipelineResult:
        current = chunk
        stage_metrics: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        dynamic_buffer_ms = float(current.metadata.get("dynamic_buffer_ms", 0.0))
        self._sync_for_timing()
        total_start = time.perf_counter()
        for idx, processor in enumerate(self.processors):
            stage_name = f"{idx:02d}_{processor.name}"
            self._sync_for_timing()
            start = time.perf_counter()
            result: ProcessResult = processor.process(current)
            self._sync_for_timing()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.processing_time_ms = elapsed_ms
            current = result.chunk
            dynamic_buffer_ms = float(current.metadata.get("dynamic_buffer_ms", dynamic_buffer_ms))
            metrics = dict(result.metrics)
            metrics.update(
                {
                    "stage": stage_name,
                    "processor": processor.name,
                    "processing_time_ms": elapsed_ms,
                    "algorithmic_latency_ms": result.algorithmic_latency_ms,
                    "chunk_start_time_sec": chunk.start_time_sec,
                }
            )
            stage_metrics.append(metrics)
            for event in result.events:
                enriched = dict(event)
                enriched.setdefault("stage", stage_name)
                enriched.setdefault("processor", processor.name)
                enriched.setdefault("timestamp_sec", chunk.start_time_sec)
                events.append(enriched)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return PipelineResult(
            chunk=current,
            stage_metrics=stage_metrics,
            events=events,
            processing_time_ms=total_ms,
            algorithmic_latency_ms=self.algorithmic_latency_ms,
            dynamic_buffer_ms=dynamic_buffer_ms,
        )
