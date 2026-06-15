from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyTracker:
    max_added_latency_ms: float = 200.0
    algorithmic_latency_ms: float = 0.0
    processing_ms: list[float] = field(default_factory=list)
    dynamic_buffer_ms: list[float] = field(default_factory=list)
    total_added_latency_ms: list[float] = field(default_factory=list)
    stage_processing_ms: dict[str, list[float]] = field(default_factory=dict)
    violation_count: int = 0

    def add_chunk(self, processing_ms: float, dynamic_buffer_ms: float = 0.0) -> None:
        self.processing_ms.append(float(processing_ms))
        self.dynamic_buffer_ms.append(float(dynamic_buffer_ms))
        total = float(self.algorithmic_latency_ms) + float(dynamic_buffer_ms)
        self.total_added_latency_ms.append(total)
        if total > self.max_added_latency_ms:
            self.violation_count += 1

    def add_stage(self, stage: str, processing_ms: float) -> None:
        self.stage_processing_ms.setdefault(stage, []).append(float(processing_ms))

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "processing_ms": self._percentiles(self.processing_ms),
            "dynamic_slowdown_buffer_ms": self._percentiles(self.dynamic_buffer_ms),
            "total_added_latency_ms": self._percentiles(self.total_added_latency_ms),
            "declared_algorithmic_latency_ms": float(self.algorithmic_latency_ms),
            "latency_budget_violation_count": int(self.violation_count),
        }

    def stage_summary(self) -> list[dict[str, Any]]:
        rows = []
        for stage, values in self.stage_processing_ms.items():
            stats = self._percentiles(values)
            rows.append({"stage": stage, **{f"processing_ms_{k}": v for k, v in stats.items()}})
        return rows
