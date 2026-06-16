from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from callclarity.types import MethodUnavailable, StreamingProcessor

ProcessorT = TypeVar("ProcessorT", bound=type)

_REGISTRY: dict[tuple[str, str], type[StreamingProcessor]] = {}

_BUILTIN_MODULES = [
    "callclarity.methods.preprocess.audio_validation",
    "callclarity.methods.repair.dropout_click",
    "callclarity.methods.repair.decrackle",
    "callclarity.methods.repair.neural_decrackle",
    "callclarity.methods.filter.dc_highpass",
    "callclarity.methods.filter.adaptive_clarity",
    "callclarity.methods.filter.dpdfnet_naturalize",
    "callclarity.methods.denoise.passthrough",
    "callclarity.methods.denoise.spectral_gate",
    "callclarity.methods.denoise.noisereduce_wrapper",
    "callclarity.methods.denoise.deepfilternet_wrapper",
    "callclarity.methods.denoise.dpdfnet",
    "callclarity.methods.denoise.rnnoise_external",
    "callclarity.methods.denoise.webrtc_apm",
    "callclarity.methods.denoise.dtln_onnx",
    "callclarity.methods.denoise.tiny_mask_gru_infer",
    "callclarity.methods.postfilter.codec_artifact",
    "callclarity.methods.bandwidth.guarded_exciter",
    "callclarity.methods.bandwidth.ap_bwe",
    "callclarity.methods.bandwidth.flashsr",
    "callclarity.methods.enhance.dpdfnet_detail_rescue",
    "callclarity.methods.enhance.dpdfnet_remaster",
    "callclarity.methods.vad.energy",
    "callclarity.methods.vad.silero",
    "callclarity.methods.vad.webrtcvad_wrapper",
    "callclarity.methods.leveler.speech_aware_agc",
    "callclarity.methods.leveler.compressor_limiter",
    "callclarity.methods.rate_detector.transcript_duration",
    "callclarity.methods.rate_detector.syllable_nuclei",
    "callclarity.methods.rate_detector.onset_flux",
    "callclarity.methods.rate_detector.asr_alignment",
    "callclarity.methods.rate_detector.neural_rate_tcn",
    "callclarity.methods.slowdown.none",
    "callclarity.methods.slowdown.pause_only",
    "callclarity.methods.slowdown.streaming_wsola",
    "callclarity.methods.slowdown.phase_vocoder",
    "callclarity.methods.slowdown.external_tsm",
]


def register_method(category: str, name: str) -> Callable[[ProcessorT], ProcessorT]:
    def decorator(cls: ProcessorT) -> ProcessorT:
        key = (category, name)
        if key in _REGISTRY:
            raise ValueError(f"Duplicate method registration for {category}/{name}")
        _REGISTRY[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def import_builtin_methods() -> None:
    for module_name in _BUILTIN_MODULES:
        importlib.import_module(module_name)


def get_method_class(category: str, name: str) -> type[StreamingProcessor]:
    import_builtin_methods()
    try:
        return _REGISTRY[(category, name)]
    except KeyError as exc:
        known = ", ".join(f"{cat}/{method}" for cat, method in sorted(_REGISTRY))
        raise KeyError(f"Unknown method {category}/{name}. Known methods: {known}") from exc


def create_method(category: str, name: str, config: dict[str, Any] | None = None) -> StreamingProcessor:
    cls = get_method_class(category, name)
    try:
        return cls(config or {})  # type: ignore[call-arg]
    except MethodUnavailable:
        raise
    except TypeError:
        return cls()  # type: ignore[call-arg]


def registered_methods() -> dict[str, list[str]]:
    import_builtin_methods()
    out: dict[str, list[str]] = {}
    for category, name in sorted(_REGISTRY):
        out.setdefault(category, []).append(name)
    return out
