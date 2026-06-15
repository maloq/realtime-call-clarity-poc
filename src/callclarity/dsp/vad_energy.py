from __future__ import annotations

import math

import torch

from callclarity.dsp.envelope import rms_dbfs


class AdaptiveEnergyVad:
    def __init__(
        self,
        threshold_db_above_noise: float = 8.0,
        min_speech_dbfs: float = -46.0,
        noise_alpha: float = 0.995,
        prob_slope_db: float = 6.0,
    ) -> None:
        self.threshold_db_above_noise = float(threshold_db_above_noise)
        self.min_speech_dbfs = float(min_speech_dbfs)
        self.noise_alpha = float(noise_alpha)
        self.prob_slope_db = float(prob_slope_db)
        self.noise_floor_dbfs = -70.0

    def reset(self) -> None:
        self.noise_floor_dbfs = -70.0

    def update(self, waveform: torch.Tensor) -> tuple[float, bool, dict[str, float]]:
        db = rms_dbfs(waveform)
        threshold = max(self.min_speech_dbfs, self.noise_floor_dbfs + self.threshold_db_above_noise)
        x = (db - threshold) / max(self.prob_slope_db, 1e-6)
        speech_prob = 1.0 / (1.0 + math.exp(-x))
        is_speech = speech_prob >= 0.5
        if not is_speech:
            self.noise_floor_dbfs = (
                self.noise_alpha * self.noise_floor_dbfs + (1.0 - self.noise_alpha) * db
            )
        return speech_prob, is_speech, {
            "rms_dbfs": db,
            "vad_threshold_dbfs": threshold,
            "noise_floor_dbfs": self.noise_floor_dbfs,
        }
