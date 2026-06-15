from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from callclarity.config import compose_config  # noqa: E402
from callclarity.data.manifests import (  # noqa: E402
    build_manifest_from_config,
    read_manifest,
    write_manifest,
)
from callclarity.experiments.compare import compare_runs  # noqa: E402
from callclarity.experiments.runner import run_eval  # noqa: E402
from callclarity.utils.files import ensure_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_suite.yaml"
DEFAULT_INPUT_CACHE_ROOT = "data/cache/inputs"


def _list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    raise TypeError(f"{key} must be a string or list of strings.")


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(value.items())
    if isinstance(value, dict):
        return value
    raise TypeError(f"{key} must be a mapping.")


def _safe_clean_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    repo = REPO_ROOT.resolve()
    if resolved == repo or repo not in resolved.parents:
        raise RuntimeError(f"Refusing to clean output_dir outside repo: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _suite_config_path(value: str | None) -> Path:
    if value is None:
        return DEFAULT_CONFIG
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _suite_path(suite: Any, key: str, default: str) -> Path:
    path = Path(str(suite.get(key, default)))
    return path if path.is_absolute() else REPO_ROOT / path


def _all_audio_paths_exist(rows: list[dict[str, Any]]) -> bool:
    return all(Path(str(row.get("audio_path", ""))).exists() for row in rows)


def _plain_config(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _strip_cache_dirs(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_cache_dirs(item)
            for key, item in value.items()
            if str(key) != "cache_dir"
        }
    if isinstance(value, list):
        return [_strip_cache_dirs(item) for item in value]
    return value


def _resolve_repo_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _source_paths(data_cfg: Any) -> list[Path]:
    data = _plain_config(data_cfg)
    if not isinstance(data, dict):
        return []
    values: list[str] = []
    manifest_path = data.get("manifest_path")
    if manifest_path:
        values.append(str(manifest_path))
    selections = data.get("selections") or []
    if selections:
        for selection in selections:
            if isinstance(selection, dict) and selection.get("input_dir"):
                values.append(str(selection["input_dir"]))
    elif data.get("input_dir"):
        values.append(str(data["input_dir"]))
    paths: list[Path] = []
    for value in values:
        path = _resolve_repo_path(value)
        if path is not None:
            paths.append(path)
    return paths


def _path_fingerprint(path: Path) -> dict[str, Any]:
    display = _display_path(path)
    if not path.exists():
        return {"path": display, "exists": False}
    if path.is_file():
        stat = path.stat()
        return {
            "path": display,
            "exists": True,
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    if path.is_dir():
        files = []
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            stat = child.stat()
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return {"path": display, "exists": True, "kind": "dir", "files": files}
    return {"path": display, "exists": True, "kind": "other"}


def _data_cache_fingerprint(data_cfg: Any) -> tuple[str, dict[str, Any]]:
    identity_config = _strip_cache_dirs(_plain_config(data_cfg))
    payload = {
        "data": identity_config,
        "sources": [_path_fingerprint(path) for path in _source_paths(data_cfg)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16], payload


def _data_cfg_for_cache(data_cfg: Any, cache_dir: Path) -> Any:
    cfg = OmegaConf.create(_plain_config(data_cfg))
    cfg.cache_dir = _display_path(cache_dir)
    selections = cfg.get("selections", None)
    if selections:
        for selection in selections:
            selection.cache_dir = _display_path(cache_dir)
    return cfg


def _load_or_prepare_inputs(
    suite: Any,
    common_overrides: list[str],
    refresh_inputs: bool,
) -> list[dict[str, Any]]:
    manifest_cfg = compose_config(common_overrides)
    fingerprint, fingerprint_payload = _data_cache_fingerprint(manifest_cfg.data)
    cache_root = _suite_path(suite, "input_cache_root", DEFAULT_INPUT_CACHE_ROOT)
    cache_dir = cache_root / fingerprint
    extracted_cache_dir = cache_dir / "files"
    manifest_path = cache_dir / "manifest.jsonl"
    data_config_path = cache_dir / "data_config_resolved.yaml"
    fingerprint_path = cache_dir / "input_fingerprint.json"
    data_config_text = OmegaConf.to_yaml(_data_cfg_for_cache(manifest_cfg.data, extracted_cache_dir), resolve=True)

    if not refresh_inputs and manifest_path.exists() and data_config_path.exists():
        cached_data_config = data_config_path.read_text(encoding="utf-8")
        rows = read_manifest(manifest_path)
        if cached_data_config == data_config_text and _all_audio_paths_exist(rows):
            print(
                f"[suite] inputs cached files={len(rows)} manifest={_display_path(manifest_path)}",
                flush=True,
            )
            return rows

    start = perf_counter()
    ensure_dir(cache_dir)
    cached_data_cfg = _data_cfg_for_cache(manifest_cfg.data, extracted_cache_dir)
    rows = build_manifest_from_config(cached_data_cfg)
    write_manifest(rows, manifest_path)
    data_config_path.write_text(data_config_text, encoding="utf-8")
    fingerprint_path.write_text(
        json.dumps(fingerprint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[suite] inputs prepared "
        f"files={len(rows)} "
        f"wall={perf_counter() - start:.2f}s "
        f"manifest={_display_path(manifest_path)}",
        flush=True,
    )
    return rows


def run_suite(
    config_path: str | None = None,
    dry_run: bool = False,
    refresh_inputs: bool = False,
) -> int:
    os.chdir(REPO_ROOT)
    suite_path = _suite_config_path(config_path)
    suite = OmegaConf.load(suite_path)
    output_dir = Path(str(suite.get("output_dir", "outputs/experiment_suite")))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    presets = _list(suite.get("presets"), "presets")
    if not presets:
        raise RuntimeError("Suite config requires at least one preset.")
    common_overrides = _list(suite.get("common_overrides"), "common_overrides")
    per_preset = _mapping(suite.get("per_preset_overrides"), "per_preset_overrides")

    run_dirs: list[Path] = []
    print(f"[suite] config={_display_path(suite_path)}")
    print(f"[suite] output_dir={_display_path(output_dir)}")
    print(f"[suite] presets={','.join(presets)}")
    manifest_cfg = compose_config(common_overrides)
    fingerprint, _ = _data_cache_fingerprint(manifest_cfg.data)
    input_cache_root = _suite_path(suite, "input_cache_root", DEFAULT_INPUT_CACHE_ROOT)
    input_cache_dir = input_cache_root / fingerprint
    print(f"[suite] input_cache_root={_display_path(input_cache_root)}")
    print(f"[suite] input_cache_dir={_display_path(input_cache_dir)}")

    if dry_run:
        for preset in presets:
            run_dir = output_dir / "_internal" / "runs" / preset
            preset_overrides = _list(per_preset.get(preset), f"per_preset_overrides.{preset}")
            overrides = [
                *common_overrides,
                *preset_overrides,
                f"pipeline={preset}",
                f"output_dir={_display_path(run_dir)}",
            ]
            print(f"[suite:dry-run] {preset}: {' '.join(overrides)}")
        return 0

    if bool(suite.get("clean_output_dir", False)):
        _safe_clean_output_dir(output_dir)
    internal_dir = ensure_dir(output_dir / "_internal")
    (internal_dir / "suite_config_resolved.yaml").write_text(
        OmegaConf.to_yaml(suite, resolve=True),
        encoding="utf-8",
    )
    manifest_rows = _load_or_prepare_inputs(
        suite,
        common_overrides,
        refresh_inputs=refresh_inputs or bool(suite.get("refresh_inputs", False)),
    )

    for preset in presets:
        run_dir = output_dir / "_internal" / "runs" / preset
        preset_overrides = _list(per_preset.get(preset), f"per_preset_overrides.{preset}")
        overrides = [
            *common_overrides,
            *preset_overrides,
            f"pipeline={preset}",
            f"output_dir={_display_path(run_dir)}",
        ]
        print(f"[suite] running {preset} -> {_display_path(run_dir)}", flush=True)
        cfg = compose_config(overrides)
        run_start = perf_counter()
        summary = run_eval(cfg, run_dir, manifest_rows=manifest_rows)
        print(
            "[suite] completed "
            f"{preset} "
            f"files={summary.get('num_files', len(manifest_rows))} "
            f"wall={perf_counter() - run_start:.2f}s",
            flush=True,
        )
        run_dirs.append(run_dir)

    if bool(suite.get("run_comparison", True)):
        print(f"[suite] comparing {len(run_dirs)} runs", flush=True)
        compare_runs(run_dirs, output_dir)
    print(f"[suite] done -> {_display_path(output_dir)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the configured call clarity experiment suite.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Suite config YAML. Defaults to configs/experiment_suite.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved suite runs without executing.",
    )
    parser.add_argument(
        "--refresh-inputs",
        action="store_true",
        help="Rescan archives and rewrite the prepared input cache before running.",
    )
    args = parser.parse_args(argv)
    return run_suite(args.config, dry_run=args.dry_run, refresh_inputs=args.refresh_inputs)


if __name__ == "__main__":
    raise SystemExit(main())
