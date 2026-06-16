from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from callclarity.config import compose_config
from callclarity.data.manifests import build_manifest_from_config, write_manifest
from callclarity.experiments.compare import compare_runs
from callclarity.experiments.runner import benchmark_latency, make_samples, run_eval, run_file
from callclarity.experiments.suite import main as experiment_suite_main
from callclarity.train.crackle_classifier import (
    pseudolabel_crackle_dataset,
    train_crackle_classifier,
)
from callclarity.train.train_denoiser import train_denoiser
from callclarity.train.train_rate_detector import train_rate_detector
from callclarity.utils.files import ensure_dir, write_json
from callclarity.utils.logging import configure_logging
from callclarity.utils.seed import seed_everything


COMMANDS = {
    "preprocess",
    "run-file",
    "eval",
    "experiment",
    "enhance-eval",
    "compare",
    "benchmark-latency",
    "make-samples",
    "train-denoiser",
    "train-rate-detector",
    "train-crackle-classifier",
    "pseudolabel-crackle",
}


def _split_presets(values: list[str] | None) -> list[str]:
    if not values:
        return ["receive_baseline"]
    presets: list[str] = []
    for value in values:
        presets.extend([part.strip() for part in value.split(",") if part.strip()])
    return presets or ["receive_baseline"]


def _split_paths(values: list[str] | None, default: list[str] | None = None) -> list[str]:
    paths: list[str] = []
    for value in values or []:
        paths.extend([part.strip() for part in value.split(",") if part.strip()])
    return paths or list(default or [])


def train_crackle_classifier_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train-crackle-classifier",
        description="Train a lightweight clean-vs-crackle classifier for pseudolabeling.",
    )
    parser.add_argument(
        "--clean",
        action="append",
        help="Clean audio directory/file. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--crackle",
        action="append",
        help="Crackle audio directory/file. Can be repeated or comma-separated.",
    )
    parser.add_argument("--out", default="data/checkpoints/crackle_classifier", help="Output directory.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=250.0)
    parser.add_argument("--hop-ms", type=float, default=125.0)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    ns = parser.parse_args(list(argv or []))
    result = train_crackle_classifier(
        _split_paths(ns.clean, ["data/test_data_samples/clean"]),
        _split_paths(ns.crackle, ["data/test_data_samples/crackle_examples"]),
        ns.out,
        sample_rate=ns.sample_rate,
        window_ms=ns.window_ms,
        hop_ms=ns.hop_ms,
        hidden_size=ns.hidden_size,
        epochs=ns.epochs,
        learning_rate=ns.lr,
        device=ns.device,
        max_files_per_class=ns.max_files_per_class,
        seed=ns.seed,
    )
    print(OmegaConf.to_yaml(result))
    return 0


def pseudolabel_crackle_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pseudolabel-crackle",
        description="Score audio files as clean/crackle/uncertain for manual dataset cleanup.",
    )
    parser.add_argument("input", nargs="+", help="Audio directory, file, or .tar/.tar.gz archive to scan.")
    parser.add_argument("--checkpoint", required=True, help="Classifier checkpoint from train-crackle-classifier.")
    parser.add_argument("--out", default="outputs/crackle_pseudolabels", help="Output directory.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--uncertain-low", type=float, default=0.45)
    parser.add_argument("--uncertain-high", type=float, default=0.65)
    parser.add_argument(
        "--score-stat",
        choices=["event", "mean", "p95", "max"],
        default="event",
        help="File-level score for labels. event=p95*sqrt(active-window-ratio), p95 is more sensitive.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--copy-top-n",
        type=int,
        default=0,
        help="Copy top clean/crackle/uncertain files into review folders.",
    )
    ns = parser.parse_args(list(argv or []))
    result = pseudolabel_crackle_dataset(
        _split_paths(ns.input),
        ns.checkpoint,
        ns.out,
        device=ns.device,
        threshold=ns.threshold,
        uncertain_low=ns.uncertain_low,
        uncertain_high=ns.uncertain_high,
        score_stat=ns.score_stat,
        max_files=ns.max_files,
        copy_top_n=ns.copy_top_n,
    )
    print(OmegaConf.to_yaml(result))
    return 0


def enhance_eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="enhance-eval")
    parser.add_argument(
        "input_dir",
        help="Directory, manifest input, or archive containing degraded audio.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        help="Pipeline preset name. Can be repeated or comma-separated.",
    )
    parser.add_argument("--out", default="reports/enhance_eval", help="Output report directory.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=float, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch/metric device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--repair-checkpoint",
        default=None,
        help="Checkpoint path for repair stages such as neural_decrackle.",
    )
    parser.add_argument(
        "--sync-gpu-latency",
        action="store_true",
        help="Synchronize CUDA before/after stage timing for accurate GPU method latency.",
    )
    parser.add_argument(
        "--dnsmos-only",
        action="store_true",
        help="Faster tuning mode: keep DNSMOS enabled and disable NISQA/SQUIM.",
    )
    parser.add_argument(
        "--enable-neural-metrics",
        action="store_true",
        help="Enable NISQA, DNSMOS, and SQUIM wrappers.",
    )
    parser.add_argument(
        "--disable-neural-metrics",
        action="store_true",
        help="Disable NISQA, DNSMOS, and SQUIM wrappers for quick smoke runs.",
    )
    ns = parser.parse_args(list(argv or []))

    presets = _split_presets(ns.preset)
    summaries = {}
    run_dirs: list[Path] = []
    out_root = Path(ns.out)
    runs_root = out_root / "_internal" / "runs"
    for preset in presets:
        run_out = runs_root / preset
        input_path = Path(ns.input_dir)
        overrides = [
            f"pipeline={preset}",
            f"audio.sample_rate={ns.sample_rate}",
            f"output_dir={run_out}",
            "sample_selector.num_examples=1000000",
        ]
        if input_path.suffix.lower() in {".jsonl", ".json"}:
            overrides.extend(
                [
                    "data.selections=[]",
                    f"data.manifest_path={ns.input_dir}",
                    "data.input_dir=.",
                ]
            )
        else:
            overrides.extend(["data.selections=[]", f"data.input_dir={ns.input_dir}"])
        if (Path("configs") / "enhance" / f"{preset}.yaml").exists():
            overrides.append(f"enhance={preset}")
        if (Path("configs") / "denoise" / f"{preset}.yaml").exists():
            overrides.append(f"denoise={preset}")
        if ns.max_files is not None:
            overrides.append(f"data.max_files={ns.max_files}")
        if ns.chunk_ms is not None:
            overrides.append(f"audio.chunk_ms={ns.chunk_ms}")
        if ns.device is not None:
            overrides.extend(
                [
                    f"runtime.device={ns.device}",
                    f"metrics.no_reference.device={ns.device}",
                    f"metrics.no_reference.nisqa.device={ns.device}",
                    f"metrics.no_reference.dnsmos.device={ns.device}",
                    f"metrics.no_reference.squim.device={ns.device}",
                    f"++enhance.deepfilternet.device={ns.device}",
                    f"++enhance.metric_selector.metric_cfg.dnsmos.device={ns.device}",
                    f"repair.device={ns.device}",
                ]
            )
        if ns.repair_checkpoint is not None:
            overrides.append(f"repair.checkpoint_path={ns.repair_checkpoint}")
        if ns.sync_gpu_latency:
            overrides.append("latency.synchronize_gpu_for_timing=true")
        if ns.enable_neural_metrics and ns.disable_neural_metrics:
            raise SystemExit("Use only one of --enable-neural-metrics or --disable-neural-metrics.")
        if ns.enable_neural_metrics:
            overrides.extend(
                [
                    "metrics.no_reference.nisqa.enabled=true",
                    "metrics.no_reference.dnsmos.enabled=true",
                    "metrics.no_reference.squim.enabled=true",
                ]
            )
        if ns.disable_neural_metrics:
            overrides.extend(
                [
                    "metrics.no_reference.nisqa.enabled=false",
                    "metrics.no_reference.dnsmos.enabled=false",
                    "metrics.no_reference.squim.enabled=false",
                ]
            )
        if ns.dnsmos_only:
            overrides.extend(
                [
                    "metrics.no_reference.nisqa.enabled=false",
                    "metrics.no_reference.dnsmos.enabled=true",
                    "metrics.no_reference.squim.enabled=false",
                ]
            )
        cfg = compose_config(overrides)
        summary = run_eval(cfg, ensure_dir(run_out))
        summaries[preset] = summary
        run_dirs.append(run_out)
        print(f"Wrote {preset} evaluation to {run_out}")
    compare_runs(run_dirs, ensure_dir(out_root))
    write_json(out_root / "_internal" / "summaries.json", summaries)
    print(f"Wrote focused comparison to {out_root}")
    return 0


def _parse(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(prog="callclarity")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)
    return ns.command, list(ns.overrides)


def _output_dir(cfg) -> Path:
    return ensure_dir(cfg.output_dir or cfg.output.root_dir)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "enhance-eval":
        return enhance_eval_main(argv[1:])
    if argv and argv[0] == "experiment":
        return experiment_suite_main(argv[1:])
    if argv and argv[0] == "train-crackle-classifier":
        return train_crackle_classifier_main(argv[1:])
    if argv and argv[0] == "pseudolabel-crackle":
        return pseudolabel_crackle_main(argv[1:])
    command, overrides = _parse(list(sys.argv[1:] if argv is None else argv))
    cfg = compose_config(overrides)
    configure_logging()
    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    if command == "preprocess":
        out = _output_dir(cfg)
        rows = build_manifest_from_config(cfg.data)
        write_manifest(rows, out / "manifest.jsonl")
        write_json(
            out / "preprocess_summary.json",
            {"num_files": len(rows), "manifest": str(out / "manifest.jsonl")},
        )
        print(f"Wrote manifest with {len(rows)} files to {out / 'manifest.jsonl'}")
    elif command == "run-file":
        if cfg.input is None:
            raise SystemExit("run-file requires input=/path/audio")
        metrics = run_file(cfg, cfg.input, cfg.transcript, _output_dir(cfg))
        print(OmegaConf.to_yaml(metrics))
    elif command == "eval":
        summary = run_eval(cfg, _output_dir(cfg))
        print(OmegaConf.to_yaml(summary))
    elif command == "experiment":
        return experiment_suite_main(overrides)
    elif command == "enhance-eval":
        return enhance_eval_main(overrides)
    elif command == "compare":
        runs = list(cfg.runs)
        if not runs:
            raise SystemExit("compare requires runs='[run_a,run_b]'")
        compare_runs(runs, _output_dir(cfg))
        print(f"Wrote comparison to {_output_dir(cfg)}")
    elif command == "benchmark-latency":
        benchmark_latency(cfg, _output_dir(cfg))
        print(f"Wrote latency benchmark to {_output_dir(cfg)}")
    elif command == "make-samples":
        make_samples(cfg, _output_dir(cfg))
        print(f"Wrote samples to {_output_dir(cfg)}")
    elif command == "train-denoiser":
        result = train_denoiser(cfg, _output_dir(cfg))
        print(OmegaConf.to_yaml(result))
    elif command == "train-rate-detector":
        result = train_rate_detector(cfg, _output_dir(cfg))
        print(OmegaConf.to_yaml(result))
    elif command == "train-crackle-classifier":
        return train_crackle_classifier_main(overrides)
    elif command == "pseudolabel-crackle":
        return pseudolabel_crackle_main(overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
