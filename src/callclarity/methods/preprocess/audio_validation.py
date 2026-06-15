from __future__ import annotations

import torch

from callclarity.io.audio_io import resample_if_needed
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


@register_method("preprocess", "audio_validation")
class AudioValidationProcessor(BaseStreamingProcessor):
    name = "audio_validation"

    def process(self, chunk: AudioChunk) -> ProcessResult:
        if not bool(self.config.get("enabled", True)):
            return ProcessResult(chunk=chunk, algorithmic_latency_ms=0.0)
        x = chunk.samples.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        input_channels = int(x.shape[0])
        nan_inf_count = int((~torch.isfinite(x)).sum().item())
        if nan_inf_count:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mono = bool(self.config.get("mono", True))
        if mono and x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        target_sample_rate = self.config.get("target_sample_rate", None)
        output_sample_rate = chunk.sample_rate
        resampled = False
        if target_sample_rate is not None and int(target_sample_rate) != int(chunk.sample_rate):
            x = resample_if_needed(x, int(chunk.sample_rate), int(target_sample_rate))
            output_sample_rate = int(target_sample_rate)
            resampled = True
        peak_clipped = int((x.abs() > 1.0).sum().item())
        x = x.clamp(-1.0, 1.0).contiguous()
        metadata = merge_metadata(
            chunk,
            input_channels=input_channels,
            channels=int(x.shape[0]),
            validation_resampled=resampled,
        )
        return ProcessResult(
            chunk=AudioChunk(x, output_sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics={
                "input_channels": input_channels,
                "output_channels": int(x.shape[0]),
                "nan_inf_count": nan_inf_count,
                "pre_validation_clip_count": peak_clipped,
                "resampled": resampled,
                "sample_rate": output_sample_rate,
            },
            algorithmic_latency_ms=0.0,
        )
