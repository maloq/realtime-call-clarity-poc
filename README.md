# Realtime Call Clarity POC

Research-friendly Python proof of concept for CPU-first, streaming call audio processing. It implements and benchmarks:

- speech denoising,
- speech-aware loudness leveling / AGC,
- fast-speech detection,
- bounded-latency speech slowdown,
- PyTorch model skeletons for trainable denoising and rate detection,
- latency, quality, operational, guardrail, leveling, slowdown, and subjective-sample reports.

## Latency Model

The slowdown path is deliberately not "stretch the whole call forever." That cannot satisfy a bounded real-time call budget. The implemented controller only slows detected fast speech while buffer budget is available, catches up during pauses or mild non-fast speech, and clamps dynamic buffer latency to a hard configured limit. The default run budget is 200 ms maximum added latency, with the slowdown controller defaulting to a 180 ms hard buffer.

## Install

Use the requested conda environment:

```bash
conda activate pointnet
cd /home/infres/vmorozov/code/realtime-call-clarity-poc
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[metrics,denoise,vad,asr,tracking,train]"
```

Heavy or fragile backends are optional. If a selected method needs a missing backend, it raises `MethodUnavailable` with installation/configuration guidance.

## Dataset Layout

Expected layout:

```text
dataset/
  sample_001.opus
  sample_001.txt
  sample_002.opus
  sample_002.txt
```

The dataloader scans recursively. `data.input_dir` may also point at a `.tar.gz` archive; preprocessing/eval extracts it under `data.cache_dir`.

The local default points at:

```text
/home/infres/vmorozov/asr_public_phone_calls_1.tar.gz
```

Opus decoding uses available local backends in this order: Torchaudio, soundfile, ffmpeg. Install PyAV/ffmpeg/soundfile if your Torchaudio build cannot decode Opus.

## Quickstart

```bash
callclarity preprocess \
  data=local_opus_txt \
  data.input_dir=/path/to/data \
  data.cache_dir=/path/to/cache \
  output_dir=outputs/preprocess

callclarity run-file \
  input=/path/audio.opus \
  transcript=/path/audio.txt \
  pipeline=denoise_agc \
  output_dir=outputs/debug_run

callclarity eval \
  data=local_opus_txt \
  data.input_dir=/path/to/data \
  pipeline=denoise_agc \
  output_dir=outputs/eval_runs/test

callclarity experiment

enhance-eval /path/to/data \
  --preset receive_baseline \
  --out reports/receive_baseline
```

`pipeline=dpdfnet` runs the optional DPDFNet streaming ONNX baseline through the
official `dpdfnet` package. Install `pip install dpdfnet` or
`pip install -e ".[denoise]"`, then pre-download a model with
`dpdfnet download dpdfnet2` or set `denoise.onnx_path=/path/to/model.onnx`.
Use `pipeline=dpdfnet_decrackle` for the same DPDFNet path with decrackling first.

`pipeline=webrtc_apm` runs mild decrackling plus the optional WebRTC Audio
Processing Module baseline through `webrtc-audio-processing`. It enables receive-side
noise suppression and conservative adaptive digital AGC by default; echo cancellation
stays disabled because these evals do not provide a synchronized playback reference.

`callclarity experiment` is the default way to make method-comparison experiments.
It uses `configs/experiment_suite.yaml` and writes the canonical suite layout:
top-level `comparison.csv`, `method_audio.csv`, `report.md`, `report.html`, and
`samples/<sample>/raw.wav` plus one processed WAV per method. Detailed per-method
runs live under `_internal/runs/<method>`, with compact JSON artifacts under `_internal/`.

`enhance-eval` uses the same comparison layout for ad hoc input directories when you
want to choose presets from the command line.

Add `--sync-gpu-latency` only when measuring method latency; otherwise GPU timing stays
unsynchronized for faster metric-focused experiments. Result tables include `method_device`,
`metric_device`, and `latency_device`.

For a tiny local smoke test:

```bash
pytest -q
```

## Compare Methods

The preferred comparison workflow is:

```bash
callclarity experiment
```

Use `callclarity experiment --dry-run` to inspect the configured runs, or
`callclarity experiment --config configs/experiment_suite.yaml` to pass an explicit
suite file.

For lower-level manual comparisons, run multiple evals, then:

```bash
callclarity compare \
  runs='[outputs/eval_runs/baseline,outputs/eval_runs/slowdown]' \
  output_dir=outputs/comparisons/comparison_001
```

Comparison output uses the same suite-style structure: `comparison.csv`,
`method_audio.csv`, `report.md`, `report.html`, `_internal/`, and `samples/`.

## Eval Outputs

Every eval writes:

```text
config_resolved.yaml
manifest_eval.jsonl
metrics_summary.json
metrics_per_file.csv
metrics_per_file.jsonl
latency_summary.json
stage_latency.csv
per_chunk_metrics.csv
guardrails.csv
guardrails.jsonl
events.jsonl
samples/
plots/
report.md
report.html
```

Samples contain raw and processed WAVs plus `comparison_info.json`.

## Experiment Outputs

`callclarity experiment` writes the suite comparison layout:

```text
comparison.csv
method_audio.csv
samples_index.csv
README.md
report.md
report.html
samples/
  <sample>/
    raw.wav
    <method>.wav
    transcript.txt
    metrics.csv
_internal/
  suite_config_resolved.yaml
  comparison.json
  samples_index.json
  runs/
    <method>/
      config_resolved.yaml
      metrics_summary.json
      metrics_per_file.csv
      samples/
  samples/
    <sample>/
      info.json
      metrics.json
  plots/
```

## Implemented Pipelines

- `pipeline=baseline`: decrackle plus passthrough.
- `pipeline=dpdfnet`: optional DPDFNet streaming ONNX denoiser baseline.
- `pipeline=dpdfnet_decrackle`: decrackle plus DPDFNet.
- `pipeline=webrtc_apm`: decrackle plus optional WebRTC APM noise suppression and digital AGC.
- `pipeline=receive_baseline`: receive-side validation, dropout/click repair, DC/high-pass cleanup, VAD, spectral gate, codec/BWE hooks, AGC, limiter.
- `pipeline=denoise_agc`: decrackle, spectral gate, energy VAD, speech-aware AGC, limiter.

See [docs/receive_side_enhancement.md](docs/receive_side_enhancement.md) for the receive-side audio audit, metric wrappers, guardrails, method tradeoffs, and source links.

## Trainable Models

Smoke train the denoiser with explicit synthetic-noise mode:

```bash
callclarity train-denoiser \
  train=tiny_mask_gru \
  train.synthetic_noise.enabled=true \
  output_dir=outputs/train/denoiser_smoke
```

Train the rate detector smoke model:

```bash
callclarity train-rate-detector \
  train=neural_rate_tcn \
  output_dir=outputs/train/rate_detector_smoke
```

The denoiser command refuses to pretend real noisy recordings are clean targets. Provide clean/noisy pairs in future datamodules or enable synthetic noise for smoke testing.

## Known Limitations

- The built-in WSOLA implementation is compact and POC-oriented; external Rubber Band, SoundTouch, or Signalsmith wrappers are stubs until those binaries are installed.
- DeepFilterNet, DPDFNet, WebRTC APM, RNNoise, DTLN ONNX, Silero VAD, WebRTC VAD, and noisereduce are optional wrappers or adapter hooks.
- Neural no-reference quality metrics are enabled in the default metrics config; missing optional backends report skip reasons instead of crashing.
- Opus decoding depends on local codec support.
