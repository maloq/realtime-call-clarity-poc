# Receive-Side Speech Enhancement Audit and Plan

This project processes already-decoded audio. It cannot assume a clean reference, sender control, RTP packet access, codec internals, or offline latency.

## Current Pipeline Audit

- Ingress/decoding: `callclarity.io.audio_io.load_audio` decodes WAV directly and otherwise tries Torchaudio, PyAV, soundfile, then ffmpeg. `callclarity.io.opus_decode.decode_opus` is a thin mono Opus/PCM loader.
- Live representation: decoded float PCM tensors in `AudioChunk`, shaped `[channels, samples]`, with `sample_rate`, `start_time_sec`, `stream_id`, and metadata.
- Default sample format: `configs/config.yaml` sets 16 kHz, mono, 10 ms chunks, WAV output.
- Frame size: `streaming.iter_audio_chunks` defaults to 10 ms; pipelines can override `chunk_ms`.
- Latency budget: root config uses 200 ms max added latency, 150 ms warning; `RealtimeSimulator` records processing time, algorithmic latency, dynamic buffer latency, and real-time factor.
- Buffering: slowdown paths use bounded dynamic buffering; base denoise/AGC paths are chunk-by-chunk.
- Codec/RTP domain: there is no RTP packet, sequence-number, codec-frame, or sender timestamp path in the current live pipeline. Dropout handling must infer issues from decoded PCM and chunk timestamps.
- Existing DSP/enhancement: spectral-gate denoise, energy VAD, speech-aware AGC, static compressor, peak limiter, compact WSOLA slowdown, strong composite enhancer, and optional wrappers/stubs for noisereduce, DTLN ONNX, DeepFilterNet, RNNoise, Silero VAD, and WebRTC VAD.
- Existing metrics before this pass: RMS/peak/residual proxy metrics, latency summaries, leveling summaries, slowdown summaries, report writers, and smoke tests. DNSMOS/STOI/SI-SDR were placeholders.

## Added in This Pass

- `pipeline=receive_baseline`: a modular receive-side processor chain with bypassable stages:
  validation/mono/resampling -> dropout/click repair -> DC blocker/high-pass -> VAD -> spectral gate -> codec post-filter hook -> bandwidth extension hook -> speech-aware AGC -> limiter.
- Offline no-reference metric wrappers:
  NISQA v2, DNSMOS/P.835, SQUIM objective, and PLCMOS placeholder. NISQA/DNSMOS/SQUIM are enabled in the default metrics config and report `*_status` / `*_error` instead of crashing when optional dependencies are missing.
- Cheap streaming operational metrics:
  input/output RMS, speech-active RMS, clipping pct, limiter gain reduction, zero/repeated frames, dropout/timestamp gaps, discontinuity/click counts, noise-floor estimate, approximate SNR, spectral centroid, rolloff, high-frequency energy ratio, narrowband score, RTF, added latency, and queue counters.
- Evaluation outputs:
  `per_chunk_metrics.csv`, `guardrails.csv`, and `guardrails.jsonl` in addition to existing CSV/JSON/Markdown/HTML outputs.
- `enhance-eval` CLI:
  `enhance-eval input_dir --preset receive_baseline --out reports/baseline`.
- Guardrails for regressions:
  warns on noise-score improvements paired with speech-quality drops, worse coloration/discontinuity, clipping increases, RTF/latency budget failures, estimated intelligibility drops, and large spectral changes on clean-ish input.

## Metrics

Heavy neural metrics are not run in the audio callback. They are enabled for recorded-window/batch evaluation by default:

```bash
enhance-eval /path/to/audio \
  --preset receive_baseline \
  --out reports/receive_baseline
```

Equivalent Hydra flags:

```bash
callclarity eval \
  data.input_dir=/path/to/audio \
  pipeline=receive_baseline \
  metrics.no_reference.nisqa.enabled=true \
  metrics.no_reference.dnsmos.enabled=true \
  metrics.no_reference.squim.enabled=true \
  output_dir=reports/receive_baseline
```

Columns include processed scores, raw scores with `raw_` prefixes, and deltas:

- NISQA: `nisqa_mos`, `nisqa_noisiness`, `nisqa_coloration`, `nisqa_discontinuity`, `nisqa_loudness`.
- DNSMOS: `dnsmos_p808`, `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovrl`.
- SQUIM objective: `squim_pesq_est`, `squim_stoi_est`, `squim_si_sdr_est`.
- PLCMOS: scaffolded as `plcmos`; only useful for packet-loss/gap-concealment experiments once a backend is configured.

Watch the deltas, not only absolute scores. A preset is suspicious if DNSMOS BAK improves while DNSMOS SIG falls, NISQA discontinuity worsens, SQUIM STOI estimate falls, output clipping increases, or RTF approaches 1.0.

## Real-Time DSP Stages

### Audio Validation

Config: stage `preprocess/audio_validation`.

- Sanitizes NaN/Inf samples.
- Ensures `[channels, samples]`.
- Mixes to mono when configured.
- Resamples only if the configured target sample rate differs.
- Clamps to full-scale safety.

Expected latency: 0 ms algorithmic, but resampling should normally be done before hard real-time callbacks.

### Dropout and Click Repair

Config: stage `repair/dropout_click`.

- Detects timestamp gaps, zero frames, repeated frames, click-like jumps, and tiny zero runs.
- Repairs isolated clicks with local interpolation.
- Repairs very small zero gaps with interpolation.
- Conceals all-zero frames by fading the previous frame.

Expected latency: 0 ms beyond the current chunk. It does not insert missing packets, so duration is preserved.

### De-Crackle / De-Click

Config: stage `repair/decrackle`.

This is a conservative streaming impulse suppressor for bad-record-player crackle, tiny digital
pops, and short packet glitches. It keeps a short history tail so clicks at chunk boundaries can
be detected, compares each sample with a local median predictor, and only repairs short
high-confidence outlier runs. Repair uses short interpolation with a crossfaded blend; it does not
hard-mute the audio. The stage adds no algorithmic latency beyond the current frame.

Recommended placement:

```text
input validation / high-pass -> decrackle -> denoise/post-filter -> EQ/AGC/limiter
```

Example settings:

```yaml
# Safe default for live calls.
enabled: true
strength: mild

# More audible cleanup for noisy recordings or offline sample generation.
enabled: true
strength: medium

# Last resort. Listen carefully: this can soften consonants/transients.
enabled: true
strength: aggressive
```

Main parameters:

- `strength`: `mild`, `medium`, `aggressive`, or a numeric value from 0 to 1.
- `max_click_duration_ms`: longest outlier run to repair.
- `detection_threshold`: local median-residual threshold multiplier.
- `abs_threshold`: absolute residual floor; higher values reduce false positives.
- `repair_blend`: interpolation blend amount.
- `max_repair_fraction`: cap on repaired samples per frame.

Watch `decrackle_repaired_click_count`, `decrackle_repaired_samples`,
`input_discontinuity_count`, `output_discontinuity_count`, clipping, and no-reference metrics.
False positives are worse than leaving a little crackle, so the receive-side default is mild.

Known limits: this removes short impulses only. It will not recover long dropouts, missing speech,
or codec warbling/metallic artifacts. Aggressive settings can damage consonants, fricatives, and
plosives, so use them only after listening tests.

### Neural De-Crackle

Config: stage `repair/neural_decrackle`; pipeline preset `pipeline=receive_neural_decrackle`.

This is a lightweight causal waveform TCN that predicts a bounded residual correction. It is meant
to run after the conservative DSP de-crackle stage, so the rule-based stage catches obvious spikes
and the neural stage learns smoother restoration for residual crackle. It is checkpoint-backed:
do not enable it in production without training/listening tests on representative data.

Training accepts both WAV and Opus files recursively:

```bash
python -m callclarity.cli train-denoiser \
  train=tiny_decrackler \
  train.clean_dirs='[data/test_data_samples/clean]' \
  train.crackle_dirs='[data/test_data_samples/crackle_examples]' \
  runtime.device=cpu \
  output_dir=data/checkpoints/neural_decrackle_train
```

The clean folder provides supervised targets. The crackle folder is unpaired; it is used to extract
real crackle-shaped residual snippets that are injected into clean speech during training. More clean
speech and more diverse crackle examples should improve the model more than simply making the model
larger.

Evaluation example:

```bash
python -m callclarity.cli enhance-eval data/test_data_samples/crackle_examples \
  --preset receive_baseline \
  --preset receive_neural_decrackle \
  --repair-checkpoint data/checkpoints/neural_decrackle_train/best_model.pt \
  --out outputs/neural_decrackle_compare \
  --disable-neural-metrics
```

Watch residual crackle by listening to `samples/*/receive_neural_decrackle.wav` and by comparing
short-window median-residual/impulse metrics. The current model is intentionally tiny and causal;
it will not hallucinate missing speech, and a weak checkpoint may only make residual crackle quieter.

### Crackle Classifier and Pseudolabeling

For building a larger crackle cleanup dataset, use the lightweight feature classifier to separate
likely clean files, likely crackle files, and uncertain files for manual review. It accepts WAV,
Opus, FLAC, MP3, OGG, and M4A files recursively.

```bash
python -m callclarity.cli train-crackle-classifier \
  --clean data/test_data_samples/clean \
  --crackle data/test_data_samples/crackle_examples \
  --out data/checkpoints/crackle_classifier \
  --device auto \
  --epochs 80
```

Then scan any mixed folder or `.tar.gz` dataset:

```bash
python -m callclarity.cli pseudolabel-crackle data/some_mixed_audio_folder \
  --checkpoint data/checkpoints/crackle_classifier/best_model.pt \
  --out outputs/crackle_pseudolabels \
  --device auto \
  --score-stat event \
  --copy-top-n 100
```

Outputs:

- `pseudolabels.csv`: sorted by crackle probability for spreadsheet review.
- `pseudolabels.json`: same rows with exact paths and probabilities.
- `review/crackle`, `review/uncertain`, `review/clean`: optional copied examples for fast listening.

The default `event` score is `p95_probability * sqrt(active_window_ratio)`. It is less twitchy than
plain `p95` because one suspicious window in an otherwise clean call goes to `uncertain` instead of
immediately becoming `crackle`. Use `--score-stat p95` when hunting rare isolated clicks and
accepting more false positives.

This classifier is only a triage tool. Keep the `uncertain` band fairly wide while growing the
dataset, then retrain after manual corrections. The labels are based on short-window impulse and
spectral features, so it is good at crackle-like clicks but not a substitute for listening.

### DC Blocker and High-Pass

Config: stage `filter/dc_highpass`.

- Causal one-pole DC blocker.
- Stateful Butterworth high-pass, default 90 Hz.
- Optional presence boost is available but off by default.

Enable for almost all receive-side phone speech. Watch output clipping and NISQA coloration if presence boost is used.

### VAD, Denoise, AGC, Limiter

Existing stages remain:

- `vad/energy`: adaptive energy speech probability.
- `denoise/spectral_gate`: lightweight real-time noise suppression baseline.
- `denoise/dpdfnet`: optional DPDFNet streaming ONNX denoiser baseline via the official `dpdfnet` package.
- `denoise/webrtc_apm`: optional WebRTC Audio Processing Module baseline via `webrtc-audio-processing`, with 10 ms receive-side noise suppression and conservative digital AGC.
- `leveler/speech_aware_agc`: conservative speech-active gain control and optional compressor.
- `limiter/limiter`: final peak safety.

The external RNNoise demo wrapper is now marked not live-safe because the demo binary expects raw 48 kHz mono PCM files. A native stateful RNNoise binding can plug into the existing `denoise/rnnoise_external` interface later.

### WebRTC APM

`denoise/webrtc_apm` wraps the optional `webrtc-audio-processing` Python binding.
The baseline processes mono 10 ms PCM frames, preserves streaming chunk duration, and
enables receive-side noise suppression plus conservative adaptive digital AGC by
default. Echo cancellation stays off unless the product also has synchronized local
playback/capture reference audio.

### Codec Artifact Post-Filter

Config: stage `postfilter/codec_artifact`, disabled by default.

Current behavior is a decoded-PCM logging/interface scaffold. It records spectral rolloff, high-frequency energy ratio, and narrowband likelihood. Future work can add a causal STFT mask or learned residual model trained from clean speech encoded/decoded through target codecs.

### Bandwidth Extension

Config: stage `bandwidth/guarded_exciter`, disabled by default.

The current implementation includes a narrowband detector and an optional very light high-frequency exciter. It only applies when `narrowband_score` exceeds the threshold and should be treated as an experiment. Watch NISQA coloration, SQUIM STOI estimate, high-frequency energy ratio delta, and listener fatigue.

## Evaluation Examples

```bash
enhance-eval input_dir --preset baseline --out reports/baseline
enhance-eval input_dir --preset dpdfnet --out reports/dpdfnet
enhance-eval input_dir --preset dpdfnet,dpdfnet_decrackle --out reports/dpdfnet
enhance-eval input_dir --preset webrtc_apm --out reports/webrtc_apm
enhance-eval input_dir --preset receive_baseline --out reports/receive_baseline
enhance-eval input_dir --preset receive_baseline,strong_online --out reports/compare
```

Each run writes:

- enhanced WAV samples under `samples/`;
- `metrics_per_file.csv` / `.jsonl`;
- `metrics_summary.json`;
- `per_chunk_metrics.csv`;
- `stage_latency.csv`;
- `guardrails.csv` / `.jsonl`;
- `report.md` / `.html`.

## Future Experiments

- Native RNNoise binding with stateful 48 kHz frame processing and wet/dry strength.
- WebRTC APM adapter using only receive-relevant modules.
- DeepFilterNet live path with bounded queueing and automatic fallback when RTF is too slow.
- Codec-specific post-filter training harness: clean speech -> target codec encode/decode -> causal model -> no-reference and listening evaluation.
- PLCMOS backend and stronger PLC only if real timestamp/RTP or decoded underflow artifacts become available.

## Sources

- WebRTC Audio Processing Module: https://webrtc.googlesource.com/src/+/main/modules/audio_processing/g3doc/audio_processing_module.md
- RNNoise: https://github.com/xiph/rnnoise and https://arxiv.org/pdf/1709.08243.pdf
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet, https://arxiv.org/abs/2110.05588, https://arxiv.org/abs/2205.05474, https://arxiv.org/abs/2305.08227
- DPDFNet: https://github.com/ceva-ip/DPDFNet and https://huggingface.co/Ceva-IP/DPDFNet
- NISQA: https://github.com/gabrielmittag/NISQA and https://lightning.ai/docs/torchmetrics/stable/audio/non_intrusive_speech_quality_assessment.html
- DNSMOS: https://lightning.ai/docs/torchmetrics/stable/audio/deep_noise_suppression_mean_opinion_score.html and https://arxiv.org/abs/2110.01763
- SQUIM: https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html and https://arxiv.org/abs/2304.01448
- Bandwidth extension: https://research.google/pubs/real-time-speech-frequency-bandwidth-extension/
- PLC Challenge / PLCMOS: https://github.com/microsoft/plc-challenge and https://arxiv.org/abs/2305.15127
