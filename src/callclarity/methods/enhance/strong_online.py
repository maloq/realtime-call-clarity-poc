from __future__ import annotations

from typing import Any

import torch
from scipy import signal

from callclarity.dsp.compressor import apply_gain_db
from callclarity.dsp.envelope import rms_dbfs
from callclarity.dsp.limiter import limit_peak
from callclarity.methods.base import BaseStreamingProcessor, merge_metadata
from callclarity.methods.denoise.spectral_gate import SpectralGateDenoiser
from callclarity.methods.leveler.speech_aware_agc import SpeechAwareAgc
from callclarity.methods.vad.energy import EnergyVadProcessor
from callclarity.registry import register_method
from callclarity.types import AudioChunk, ProcessResult


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@register_method("enhance", "strong_online")
class StrongOnlineEnhancer(BaseStreamingProcessor):
    """Composite speech enhancer for online use when quality matters more than CPU latency."""

    name = "strong_online"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.vad = EnergyVadProcessor(self.config.get("vad", {}))
        self.denoiser = SpectralGateDenoiser(self.config.get("denoise", {}))
        self.agc = SpeechAwareAgc(self.config.get("agc", {}))
        self.filter_cfg = self.config.get("filters", {})
        self.gate_cfg = self.config.get("noise_gate", {})
        self.limiter_cfg = self.config.get("limiter", {})
        self.makeup_cfg = self.config.get("final_makeup", {})
        self.current_makeup_db = 0.0
        self._sample_rate: int | None = None
        self._sos = None
        self._sos_state: torch.Tensor | None = None
        self._presence_ba = None
        self._presence_state: torch.Tensor | None = None
        self.current_makeup_db = 0.0

    @property
    def algorithmic_latency_ms(self) -> float:
        return float(self.config.get("algorithmic_latency_ms", 120.0))

    @property
    def lookahead_ms(self) -> float:
        return self.algorithmic_latency_ms

    def reset(self) -> None:
        self.vad.reset()
        self.denoiser.reset()
        self.agc.reset()
        self._sample_rate = None
        self._sos = None
        self._sos_state = None
        self._presence_ba = None
        self._presence_state = None
        self.current_makeup_db = 0.0

    def warmup(self, sample_rate: int) -> None:
        self._configure_filters(sample_rate, channels=1)
        self.vad.warmup(sample_rate)
        self.denoiser.warmup(sample_rate)
        self.agc.warmup(sample_rate)

    def _configure_filters(self, sample_rate: int, channels: int) -> None:
        if self._sample_rate == sample_rate and self._sos_state is not None:
            return
        self._sample_rate = sample_rate
        highpass_hz = float(self.filter_cfg.get("highpass_hz", 120.0))
        lowpass_hz = min(float(self.filter_cfg.get("lowpass_hz", 7200.0)), sample_rate / 2.0 - 100.0)
        order = int(self.filter_cfg.get("order", 2))
        self._sos = signal.butter(
            order,
            [highpass_hz, lowpass_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        zi = signal.sosfilt_zi(self._sos)
        self._sos_state = torch.tensor(zi[:, None, :].repeat(channels, axis=1), dtype=torch.float32)

        presence_hz = min(float(self.filter_cfg.get("presence_hz", 2800.0)), sample_rate / 2.0 - 100.0)
        presence_q = float(self.filter_cfg.get("presence_q", 1.0))
        self._presence_ba = signal.iirpeak(presence_hz, presence_q, fs=sample_rate)
        b, a = self._presence_ba
        zi2 = signal.lfilter_zi(b, a)
        self._presence_state = torch.tensor(zi2[None, :].repeat(channels, axis=0), dtype=torch.float32)

    def _apply_filters(self, samples: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, dict[str, float]]:
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        self._configure_filters(sample_rate, samples.shape[0])
        assert self._sos is not None
        assert self._sos_state is not None
        assert self._presence_ba is not None
        assert self._presence_state is not None

        x_np = samples.detach().cpu().numpy()
        sos_state = self._sos_state.detach().cpu().numpy()
        filtered, new_state = signal.sosfilt(self._sos, x_np, axis=-1, zi=sos_state)
        self._sos_state = torch.tensor(new_state, dtype=torch.float32)

        b, a = self._presence_ba
        presence_state = self._presence_state.detach().cpu().numpy()
        presence, presence_new_state = signal.lfilter(b, a, filtered, axis=-1, zi=presence_state)
        self._presence_state = torch.tensor(presence_new_state, dtype=torch.float32)
        boost = _db_to_linear(float(self.filter_cfg.get("presence_boost_db", 3.5))) - 1.0
        enhanced = torch.tensor(filtered + boost * presence, dtype=samples.dtype, device=samples.device)
        return enhanced.contiguous(), {"presence_boost_db": float(self.filter_cfg.get("presence_boost_db", 3.5))}

    def _noise_gate_gain_db(self, speech_prob: float) -> float:
        threshold = float(self.gate_cfg.get("speech_threshold", 0.48))
        floor_prob = float(self.gate_cfg.get("floor_prob", 0.18))
        attenuation = float(self.gate_cfg.get("full_attenuation_db", -14.0))
        if speech_prob >= threshold:
            return 0.0
        if speech_prob <= floor_prob:
            return attenuation
        mix = (speech_prob - floor_prob) / max(threshold - floor_prob, 1e-6)
        return attenuation * (1.0 - mix)

    def _slew_makeup(self, desired: float, duration_sec: float, rate_db_per_sec: float) -> None:
        delta = desired - self.current_makeup_db
        step = max(0.0, duration_sec) * max(rate_db_per_sec, 0.0)
        if abs(delta) <= step:
            self.current_makeup_db = desired
        else:
            self.current_makeup_db += step if delta > 0 else -step

    def _update_makeup_gain_db(
        self,
        samples: torch.Tensor,
        duration_sec: float,
        speech_prob: float,
    ) -> tuple[float, float]:
        if not bool(self.makeup_cfg.get("enabled", True)):
            return 0.0, rms_dbfs(samples)
        level_db = rms_dbfs(samples)
        speech_threshold = float(self.makeup_cfg.get("speech_threshold", 0.3))
        if speech_prob < speech_threshold or level_db < float(self.makeup_cfg.get("min_signal_dbfs", -58.0)):
            self._slew_makeup(
                0.0,
                duration_sec,
                float(self.makeup_cfg.get("silence_release_db_per_sec", 24.0)),
            )
            return self.current_makeup_db, level_db
        target = float(self.makeup_cfg.get("target_rms_dbfs", -18.5))
        desired = target - level_db
        desired = max(
            -float(self.makeup_cfg.get("max_cut_db", 8.0)),
            min(float(self.makeup_cfg.get("max_boost_db", 18.0)), desired),
        )
        rate = (
            float(self.makeup_cfg.get("attack_db_per_sec", 30.0))
            if desired > self.current_makeup_db
            else float(self.makeup_cfg.get("release_db_per_sec", 8.0))
        )
        self._slew_makeup(desired, duration_sec, rate)
        return self.current_makeup_db, level_db

    def process(self, chunk: AudioChunk) -> ProcessResult:
        filtered, filter_metrics = self._apply_filters(chunk.samples.float(), chunk.sample_rate)
        filtered_chunk = AudioChunk(
            filtered,
            chunk.sample_rate,
            chunk.start_time_sec,
            chunk.stream_id,
            dict(chunk.metadata),
        )
        vad_result = self.vad.process(filtered_chunk)
        denoise_result = self.denoiser.process(vad_result.chunk)
        speech_prob = float(denoise_result.chunk.metadata.get("speech_prob", 0.0))
        gate_gain_db = self._noise_gate_gain_db(speech_prob)
        gated = apply_gain_db(denoise_result.chunk.samples, gate_gain_db)
        gated_chunk = AudioChunk(
            gated,
            chunk.sample_rate,
            chunk.start_time_sec,
            chunk.stream_id,
            merge_metadata(denoise_result.chunk, strong_gate_gain_db=gate_gain_db),
        )
        agc_result = self.agc.process(gated_chunk)
        limited, limiter_metrics = limit_peak(
            agc_result.chunk.samples,
            float(self.limiter_cfg.get("ceiling_dbfs", -1.0)),
        )
        makeup_gain_db, pre_makeup_rms_dbfs = self._update_makeup_gain_db(
            limited,
            chunk.duration_sec,
            speech_prob,
        )
        made_up = apply_gain_db(limited, makeup_gain_db)
        limited, final_limiter_metrics = limit_peak(
            made_up,
            float(self.limiter_cfg.get("ceiling_dbfs", -1.0)),
        )
        metadata = merge_metadata(
            agc_result.chunk,
            speech_prob=speech_prob,
            strong_gate_gain_db=gate_gain_db,
        )
        events = vad_result.events + denoise_result.events + agc_result.events
        metrics = {
            "speech_prob": speech_prob,
            "strong_gate_gain_db": gate_gain_db,
            **filter_metrics,
            **{f"vad_{k}": v for k, v in vad_result.metrics.items()},
            **{f"denoise_{k}": v for k, v in denoise_result.metrics.items()},
            **agc_result.metrics,
            **limiter_metrics,
            "final_makeup_gain_db": makeup_gain_db,
            "pre_makeup_rms_dbfs": pre_makeup_rms_dbfs,
            "final_limiter_gain_reduction_db": final_limiter_metrics.get("limiter_gain_reduction_db", 0.0),
        }
        return ProcessResult(
            chunk=AudioChunk(limited, chunk.sample_rate, chunk.start_time_sec, chunk.stream_id, metadata),
            metrics=metrics,
            events=events,
            algorithmic_latency_ms=self.algorithmic_latency_ms,
        )
