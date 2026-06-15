from __future__ import annotations

from typing import Any


def flatten_latency_summary(summary: dict[str, Any]) -> dict[str, float]:
    processing = summary.get("processing_ms", {})
    total = summary.get("total_added_latency_ms", {})
    dynamic = summary.get("dynamic_slowdown_buffer_ms", {})
    return {
        "rtf": float(summary.get("real_time_factor", 0.0)),
        "chunk_ms_p50": float(processing.get("p50", 0.0)),
        "chunk_ms_p95": float(processing.get("p95", 0.0)),
        "chunk_ms_p99": float(processing.get("p99", 0.0)),
        "total_added_latency_ms_p95": float(total.get("p95", 0.0)),
        "total_added_latency_ms_p99": float(total.get("p99", 0.0)),
        "max_dynamic_slowdown_buffer_ms": float(dynamic.get("max", 0.0)),
        "budget_violation_count": int(summary.get("latency_budget_violation_count", 0)),
    }
