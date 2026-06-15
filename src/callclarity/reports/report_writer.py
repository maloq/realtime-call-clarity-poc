from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from callclarity.reports.html_report import markdown_to_basic_html
from callclarity.reports.markdown_report import render_run_markdown
from callclarity.reports.plots import plot_latency_hist, plot_tempo
from callclarity.utils.files import ensure_dir, write_json, write_jsonl


class EvalReportWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        ensure_dir(self.output_dir / "samples")
        ensure_dir(self.output_dir / "plots")

    def write_config(self, cfg: Any) -> None:
        text = OmegaConf.to_yaml(cfg, resolve=True) if not isinstance(cfg, str) else cfg
        (self.output_dir / "config_resolved.yaml").write_text(text, encoding="utf-8")

    def write_run(
        self,
        cfg: Any,
        manifest_rows: list[dict[str, Any]],
        per_file_rows: list[dict[str, Any]],
        summary: dict[str, Any],
        latency_rows: list[dict[str, Any]],
        events: list[dict[str, Any]],
        per_chunk_rows: list[dict[str, Any]],
        guardrail_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.write_config(cfg)
        write_jsonl(self.output_dir / "manifest_eval.jsonl", manifest_rows)
        write_json(self.output_dir / "metrics_summary.json", summary)
        pd.DataFrame(per_file_rows).to_csv(self.output_dir / "metrics_per_file.csv", index=False)
        write_jsonl(self.output_dir / "metrics_per_file.jsonl", per_file_rows)
        write_json(self.output_dir / "latency_summary.json", summary.get("latency", {}))
        pd.DataFrame(latency_rows).to_csv(self.output_dir / "stage_latency.csv", index=False)
        pd.DataFrame(per_chunk_rows).to_csv(self.output_dir / "per_chunk_metrics.csv", index=False)
        write_jsonl(self.output_dir / "events.jsonl", events)
        guardrail_rows = guardrail_rows or []
        pd.DataFrame(guardrail_rows).to_csv(self.output_dir / "guardrails.csv", index=False)
        write_jsonl(self.output_dir / "guardrails.jsonl", guardrail_rows)
        markdown = render_run_markdown(summary)
        (self.output_dir / "report.md").write_text(markdown, encoding="utf-8")
        (self.output_dir / "report.html").write_text(markdown_to_basic_html(markdown), encoding="utf-8")
        plot_latency_hist(per_chunk_rows, self.output_dir / "plots" / "latency_hist.png")
        plot_tempo(events, self.output_dir / "plots" / "tempo_decisions.png")
        plot_tempo(events, self.output_dir / "plots" / "slowdown_buffer_over_time.png")


def write_selected_samples_csv(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(Path(output_dir) / "selected_samples.csv", index=False)
