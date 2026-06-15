from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _cfg_value(cfg: Any, key: str, default: float) -> float:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    try:
        return float(value)
    except Exception:
        return default


def _num(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _warn(warnings: list[dict[str, Any]], rule: str, message: str, severity: str = "warning") -> None:
    warnings.append({"rule": rule, "severity": severity, "message": message})


def evaluate_guardrails(row: Mapping[str, Any], cfg: Any | None = None) -> list[dict[str, Any]]:
    sig_regression = _cfg_value(cfg, "max_dnsmos_sig_regression", 0.08)
    quality_regression = _cfg_value(cfg, "max_quality_regression", 0.08)
    clipping_epsilon = _cfg_value(cfg, "max_clipping_pct_increase", 0.001)
    rtf_limit = _cfg_value(cfg, "max_processing_rtf", 0.85)
    spectral_delta_limit = _cfg_value(cfg, "max_cleanish_hf_ratio_delta", 0.20)
    warnings: list[dict[str, Any]] = []

    bak_delta = _num(row, "dnsmos_bak_delta")
    sig_delta = _num(row, "dnsmos_sig_delta")
    if bak_delta is not None and sig_delta is not None and bak_delta > 0.05 and sig_delta < -sig_regression:
        _warn(
            warnings,
            "noise_improved_speech_regressed",
            f"DNSMOS BAK improved by {bak_delta:.3f}, but SIG regressed by {sig_delta:.3f}.",
        )

    mos_delta = _num(row, "nisqa_mos_delta")
    dnsmos_ovrl_delta = _num(row, "dnsmos_ovrl_delta")
    if mos_delta is not None and mos_delta < -quality_regression:
        _warn(warnings, "nisqa_mos_regressed", f"NISQA MOS regressed by {mos_delta:.3f}.")
    if dnsmos_ovrl_delta is not None and dnsmos_ovrl_delta < -quality_regression:
        _warn(warnings, "dnsmos_ovrl_regressed", f"DNSMOS OVRL regressed by {dnsmos_ovrl_delta:.3f}.")

    coloration_delta = _num(row, "nisqa_coloration_delta")
    if coloration_delta is not None and coloration_delta < -quality_regression:
        _warn(
            warnings,
            "coloration_regressed",
            f"NISQA coloration regressed by {coloration_delta:.3f}; "
            "bandwidth extension or EQ may be too strong.",
        )

    discontinuity_delta = _num(row, "nisqa_discontinuity_delta")
    if discontinuity_delta is not None and discontinuity_delta < -quality_regression:
        _warn(
            warnings,
            "nisqa_discontinuity_regressed",
            f"NISQA discontinuity regressed by {discontinuity_delta:.3f}.",
        )
    input_disc = _num(row, "input_discontinuity_count")
    output_disc = _num(row, "output_discontinuity_count")
    if input_disc is not None and output_disc is not None and output_disc > input_disc:
        _warn(
            warnings,
            "operational_discontinuity_increased",
            f"Output discontinuity count increased from {input_disc:.0f} to {output_disc:.0f}.",
        )

    input_clip = _num(row, "input_clipping_pct")
    output_clip = _num(row, "output_clipping_pct")
    if input_clip is not None and output_clip is not None and output_clip > input_clip + clipping_epsilon:
        _warn(
            warnings,
            "clipping_increased",
            f"Output clipping increased from {input_clip:.4f}% to {output_clip:.4f}%.",
        )

    rtf = _num(row, "processing_rtf") or _num(row, "rtf")
    if rtf is not None and rtf > rtf_limit:
        _warn(warnings, "rtf_too_slow", f"Processing RTF {rtf:.3f} exceeds limit {rtf_limit:.3f}.", "error")

    added_latency = _num(row, "added_latency_ms") or _num(row, "total_added_latency_ms_p95")
    budget = _num(row, "latency_budget_ms")
    if added_latency is not None and budget is not None and added_latency > budget:
        _warn(
            warnings,
            "latency_budget_exceeded",
            f"Added latency {added_latency:.1f} ms exceeds {budget:.1f} ms.",
            "error",
        )

    stoi_delta = _num(row, "squim_stoi_est_delta")
    if stoi_delta is not None and stoi_delta < -quality_regression:
        _warn(
            warnings,
            "estimated_intelligibility_regressed",
            f"SQUIM estimated STOI regressed by {stoi_delta:.3f}.",
        )

    raw_narrowband = _num(row, "narrowband_score")
    hf_delta = _num(row, "high_frequency_energy_ratio_delta")
    raw_clip = _num(row, "input_clipping_pct") or 0.0
    raw_dropouts = _num(row, "dropout_count") or 0.0
    if (
        raw_narrowband is not None
        and hf_delta is not None
        and raw_narrowband < 0.4
        and raw_clip < 0.01
        and raw_dropouts == 0.0
        and abs(hf_delta) > spectral_delta_limit
    ):
        _warn(
            warnings,
            "cleanish_spectral_change",
            f"High-frequency energy ratio changed by {hf_delta:.3f} on clean-ish/wideband input.",
        )

    return warnings
