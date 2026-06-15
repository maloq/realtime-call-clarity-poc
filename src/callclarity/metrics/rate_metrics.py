from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_slowdown(
    events: list[dict[str, Any]],
    input_samples: int,
    output_samples: int,
    input_sample_rate: int = 1,
    output_sample_rate: int | None = None,
) -> dict[str, Any]:
    decisions = [e for e in events if e.get("type") == "tempo_decision"]
    tempos = [float(e.get("tempo", 1.0)) for e in decisions]
    buffers = [float(e.get("buffer_ms", 0.0)) for e in decisions]
    active = [t for t in tempos if t < 0.999]
    catchup = [t for t in tempos if t > 1.001]
    input_duration = input_samples / max(float(input_sample_rate), 1.0)
    output_duration = output_samples / max(float(output_sample_rate or input_sample_rate), 1.0)
    return {
        "slowdown_active_ratio": len(active) / max(len(tempos), 1),
        "average_active_tempo": float(np.mean(active)) if active else None,
        "min_tempo_used": float(np.min(tempos)) if tempos else None,
        "catchup_active_ratio": len(catchup) / max(len(tempos), 1),
        "max_dynamic_slowdown_buffer_ms": float(np.max(buffers)) if buffers else 0.0,
        "hard_latency_limit_hit_count": int(sum(1 for e in decisions if e.get("hard_limit_active"))),
        "output_input_duration_ratio": output_duration / max(input_duration, 1e-12),
    }
