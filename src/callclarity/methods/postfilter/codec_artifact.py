from __future__ import annotations

from callclarity.metrics.operational import _spectral_metrics
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("postfilter", "codec_artifact")
class CodecArtifactPostFilter(BaseStreamingProcessor):
    name = "codec_artifact"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        spectral = _spectral_metrics(chunk.samples, chunk.sample_rate)
        enabled = bool(self.config.get("enabled", False))
        metadata = merge_metadata(
            chunk,
            codec_postfilter_enabled=enabled,
            codec_postfilter_scaffold=True,
            spectral_rolloff_hz=spectral["spectral_rolloff_hz"],
            high_frequency_energy_ratio=spectral["high_frequency_energy_ratio"],
        )
        # This POC receives decoded PCM. The live-safe first step is logging and
        # an interface hook; learned codec-domain or STFT-mask post-filters can
        # plug in here without touching the rest of the chain.
        return ProcessResult(
            chunk=AudioChunk(
                chunk.samples,
                chunk.sample_rate,
                chunk.start_time_sec,
                chunk.stream_id,
                metadata,
            ),
            metrics={
                "codec_postfilter_enabled": enabled,
                "codec_postfilter_applied": False,
                "spectral_rolloff_hz": spectral["spectral_rolloff_hz"],
                "high_frequency_energy_ratio": spectral["high_frequency_energy_ratio"],
                "narrowband_likelihood": spectral["narrowband_likelihood"],
            },
            algorithmic_latency_ms=0.0,
        )
