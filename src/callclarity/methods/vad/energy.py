from __future__ import annotations

from callclarity.dsp.vad_energy import AdaptiveEnergyVad
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("vad", "energy")
class EnergyVadProcessor(BaseStreamingProcessor):
    name = "energy_vad"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.vad = AdaptiveEnergyVad(
            threshold_db_above_noise=float(self.config.get("threshold_db_above_noise", 8.0)),
            min_speech_dbfs=float(self.config.get("min_speech_dbfs", -46.0)),
            noise_alpha=float(self.config.get("noise_alpha", 0.995)),
            prob_slope_db=float(self.config.get("prob_slope_db", 6.0)),
        )

    def reset(self) -> None:
        self.vad.reset()

    def process(self, chunk: AudioChunk) -> ProcessResult:
        prob, is_speech, metrics = self.vad.update(chunk.samples)
        metadata = merge_metadata(chunk, speech_prob=prob, is_speech=is_speech)
        return ProcessResult(
            chunk=AudioChunk(chunk.samples, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={"speech_prob": prob, "is_speech": is_speech, **metrics},
            algorithmic_latency_ms=0.0,
        )
