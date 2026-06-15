from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from callclarity.io.audio_io import resample_if_needed
from callclarity.utils.device import resolve_torch_device


OPTIONAL_QUALITY_KEYS = {
    "nisqa_mos": None,
    "nisqa_noisiness": None,
    "nisqa_coloration": None,
    "nisqa_discontinuity": None,
    "nisqa_loudness": None,
    "dnsmos_p808": None,
    "dnsmos_sig": None,
    "dnsmos_bak": None,
    "dnsmos_ovrl": None,
    "squim_pesq_est": None,
    "squim_stoi_est": None,
    "squim_si_sdr_est": None,
    "plcmos": None,
}

_DNSMOS_CACHE: dict[tuple[int, bool, str | None, int | None], Any] = {}
_NISQA_CACHE: dict[tuple[int, str], Any] = {}
_SQUIM_OBJECTIVE_MODELS: dict[str, Any] = {}


def _as_plain_mapping(cfg: Any) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return cfg
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True) or {}
    except Exception:
        pass
    return {}


def _section(cfg: Any, key: str) -> Mapping[str, Any]:
    mapping = _as_plain_mapping(cfg)
    section = mapping.get(key, {})
    return _as_plain_mapping(section)


def _enabled(cfg: Any, key: str) -> bool:
    return bool(_section(cfg, key).get("enabled", False))


def _device_for(cfg: Any, key: str) -> tuple[str, str]:
    mapping = _as_plain_mapping(cfg)
    section = _section(cfg, key)
    return resolve_torch_device(section.get("device", mapping.get("device", "cpu")))


def _status(prefix: str, status: str, error: str | None = None) -> dict[str, str | None]:
    return {f"{prefix}_status": status, f"{prefix}_error": error}


def _mono_1d(waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    x = waveform.detach().float()
    if x.ndim == 1:
        out = x.clamp(-1.0, 1.0).contiguous()
    elif x.numel() == 0:
        out = torch.zeros(0)
    else:
        out = x.mean(dim=0).clamp(-1.0, 1.0).contiguous()
    return out.to(device)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(torch.as_tensor(value).detach().cpu().flatten()[0].item())
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def _compute_nisqa(waveform: torch.Tensor, sample_rate: int, cfg: Any) -> dict[str, Any]:
    requested_device, device = _device_for(cfg, "nisqa")
    out: dict[str, Any] = {
        "nisqa_mos": None,
        "nisqa_noisiness": None,
        "nisqa_coloration": None,
        "nisqa_discontinuity": None,
        "nisqa_loudness": None,
        "nisqa_device_requested": requested_device,
        "nisqa_device": device,
        **_status("nisqa", "disabled"),
    }
    if not _enabled(cfg, "nisqa"):
        return out
    try:
        cache_key = (int(sample_rate), device)
        metric = _NISQA_CACHE.get(cache_key)
        if metric is None:
            try:
                from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
            except Exception:
                from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

            metric = NonIntrusiveSpeechQualityAssessment(int(sample_rate))
            metric = metric.to(device)
            metric.eval()
            _NISQA_CACHE[cache_key] = metric
        with torch.no_grad():
            values = torch.as_tensor(metric(_mono_1d(waveform, device))).detach().cpu().flatten()
        if values.numel() < 5:
            raise RuntimeError(f"NISQA returned {values.numel()} values, expected 5")
        out.update(
            {
                "nisqa_mos": float(values[0]),
                "nisqa_noisiness": float(values[1]),
                "nisqa_discontinuity": float(values[2]),
                "nisqa_coloration": float(values[3]),
                "nisqa_loudness": float(values[4]),
                **_status("nisqa", "ok"),
            }
        )
    except Exception as exc:
        out.update(_status("nisqa", "skipped", str(exc)))
    return out


def _compute_dnsmos(waveform: torch.Tensor, sample_rate: int, cfg: Any) -> dict[str, Any]:
    requested_device, device = _device_for(cfg, "dnsmos")
    out: dict[str, Any] = {
        "dnsmos_p808": None,
        "dnsmos_sig": None,
        "dnsmos_bak": None,
        "dnsmos_ovrl": None,
        "dnsmos_device_requested": requested_device,
        "dnsmos_device": device,
        **_status("dnsmos", "disabled"),
    }
    dnsmos_cfg = _section(cfg, "dnsmos")
    if not bool(dnsmos_cfg.get("enabled", False)):
        return out
    personalized = bool(dnsmos_cfg.get("personalized", False))
    num_threads = dnsmos_cfg.get("num_threads", None)
    cache_key = (
        int(sample_rate),
        personalized,
        device,
        int(num_threads) if num_threads else None,
    )
    try:
        metric = _get_dnsmos_metric(int(sample_rate), personalized, device, num_threads)
        with torch.no_grad():
            values = torch.as_tensor(metric(_mono_1d(waveform, "cpu"))).detach().cpu().flatten()
        if values.numel() < 4:
            raise RuntimeError(f"DNSMOS returned {values.numel()} values, expected 4")
        out.update(
            {
                "dnsmos_p808": float(values[0]),
                "dnsmos_sig": float(values[1]),
                "dnsmos_bak": float(values[2]),
                "dnsmos_ovrl": float(values[3]),
                **_status("dnsmos", "ok"),
            }
        )
    except Exception as exc:
        if device.startswith("cuda"):
            try:
                metric = _get_dnsmos_metric(int(sample_rate), personalized, "cpu", num_threads)
                with torch.no_grad():
                    values = (
                        torch.as_tensor(metric(_mono_1d(waveform, "cpu")))
                        .detach()
                        .cpu()
                        .flatten()
                    )
                if values.numel() < 4:
                    raise RuntimeError(f"DNSMOS returned {values.numel()} values, expected 4")
                out.update(
                    {
                        "dnsmos_p808": float(values[0]),
                        "dnsmos_sig": float(values[1]),
                        "dnsmos_bak": float(values[2]),
                        "dnsmos_ovrl": float(values[3]),
                        "dnsmos_device": "cpu",
                        "dnsmos_device_fallback": f"{device}: {exc}",
                        **_status("dnsmos", "ok"),
                    }
                )
            except Exception as fallback_exc:
                out.update(_status("dnsmos", "skipped", str(fallback_exc)))
        else:
            out.update(_status("dnsmos", "skipped", str(exc)))
    return out


def _get_dnsmos_metric(
    sample_rate: int,
    personalized: bool,
    device: str,
    num_threads: Any,
) -> Any:
    cache_key = (
        int(sample_rate),
        personalized,
        device,
        int(num_threads) if num_threads else None,
    )
    metric = _DNSMOS_CACHE.get(cache_key)
    if metric is not None:
        return metric
    try:
        from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
    except Exception:
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

    kwargs: dict[str, Any] = {
        "fs": int(sample_rate),
        "personalized": personalized,
        "device": device,
        "num_threads": num_threads,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        metric = DeepNoiseSuppressionMeanOpinionScore(**kwargs, cache_sessions=True)
    except TypeError:
        metric = DeepNoiseSuppressionMeanOpinionScore(**kwargs, cache_session=True)
    metric.eval()
    _DNSMOS_CACHE[cache_key] = metric
    return metric


def _compute_squim(waveform: torch.Tensor, sample_rate: int, cfg: Any) -> dict[str, Any]:
    requested_device, device = _device_for(cfg, "squim")
    out: dict[str, Any] = {
        "squim_pesq_est": None,
        "squim_stoi_est": None,
        "squim_si_sdr_est": None,
        "squim_device_requested": requested_device,
        "squim_device": device,
        **_status("squim", "disabled"),
    }
    if not _enabled(cfg, "squim"):
        return out
    try:
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        model = _SQUIM_OBJECTIVE_MODELS.get(device)
        if model is None:
            model = SQUIM_OBJECTIVE.get_model().to(device)
            model.eval()
            _SQUIM_OBJECTIVE_MODELS[device] = model
        x = _mono_1d(waveform, device).unsqueeze(0)
        target_sr = 16000
        if int(sample_rate) != target_sr:
            x = resample_if_needed(x, int(sample_rate), target_sr)
        with torch.no_grad():
            stoi, pesq, si_sdr = model(x)
        out.update(
            {
                "squim_stoi_est": _float_or_none(stoi),
                "squim_pesq_est": _float_or_none(pesq),
                "squim_si_sdr_est": _float_or_none(si_sdr),
                **_status("squim", "ok"),
            }
        )
    except Exception as exc:
        out.update(_status("squim", "skipped", str(exc)))
    return out


def _compute_plcmos(waveform: torch.Tensor, sample_rate: int, cfg: Any) -> dict[str, Any]:
    del waveform, sample_rate
    out: dict[str, Any] = {"plcmos": None, **_status("plcmos", "disabled")}
    if _enabled(cfg, "plcmos"):
        out.update(_status("plcmos", "skipped", "PLCMOS wrapper is not configured in this POC."))
    return out


def optional_no_reference_quality(
    waveform: torch.Tensor,
    sample_rate: int,
    metric_cfg: Any | None = None,
) -> dict[str, Any]:
    cfg = _as_plain_mapping(metric_cfg)
    out: dict[str, Any] = dict(OPTIONAL_QUALITY_KEYS)
    out.update(_compute_nisqa(waveform, sample_rate, cfg))
    out.update(_compute_dnsmos(waveform, sample_rate, cfg))
    out.update(_compute_squim(waveform, sample_rate, cfg))
    out.update(_compute_plcmos(waveform, sample_rate, cfg))
    active_devices = [
        str(out.get(f"{name}_device"))
        for name in ("nisqa", "dnsmos", "squim")
        if out.get(f"{name}_status") == "ok" and out.get(f"{name}_device")
    ]
    requested_devices = [
        str(out.get(f"{name}_device_requested"))
        for name in ("nisqa", "dnsmos", "squim")
        if out.get(f"{name}_device_requested")
    ]
    out["metric_device"] = (
        "mixed"
        if len(set(active_devices)) > 1
        else (active_devices[0] if active_devices else "none")
    )
    out["metric_device_requested"] = (
        "mixed"
        if len(set(requested_devices)) > 1
        else (requested_devices[0] if requested_devices else "none")
    )
    return out
