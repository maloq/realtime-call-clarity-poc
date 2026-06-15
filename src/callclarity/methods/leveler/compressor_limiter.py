from __future__ import annotations

from callclarity.dsp.compressor import static_compress
from callclarity.dsp.limiter import limit_peak
from callclarity.methods.base import BaseStreamingProcessor
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("leveler", "compressor_limiter")
class CompressorLimiter(BaseStreamingProcessor):
    name = "compressor_limiter"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        y = chunk.samples
        metrics = {}
        comp_cfg = self.config.get("compressor", {})
        if comp_cfg.get("enabled", True):
            y, comp_metrics = static_compress(
                y,
                threshold_dbfs=float(comp_cfg.get("threshold_dbfs", -18.0)),
                ratio=float(comp_cfg.get("ratio", 3.0)),
            )
            metrics.update(comp_metrics)
        lim_cfg = self.config.get("limiter", {})
        if lim_cfg.get("enabled", True):
            y, lim_metrics = limit_peak(y, float(lim_cfg.get("ceiling_dbfs", -1.5)))
            metrics.update(lim_metrics)
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, dict(chunk.metadata)),
            metrics=metrics,
            algorithmic_latency_ms=0.0,
        )


@register_method("limiter", "limiter")
class LimiterProcessor(BaseStreamingProcessor):
    name = "limiter"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        y, metrics = limit_peak(chunk.samples, float(self.config.get("ceiling_dbfs", -1.5)))
        return ProcessResult(
            chunk=AudioChunk(y, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, dict(chunk.metadata)),
            metrics=metrics,
            algorithmic_latency_ms=0.0,
        )
