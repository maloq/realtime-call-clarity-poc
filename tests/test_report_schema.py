from callclarity.metrics.aggregate import aggregate_run


def test_report_schema_contains_required_sections_with_null_metrics():
    summary = aggregate_run("run", "pipeline", "dataset", [{"input_duration_sec": 1.0, "stoi": None}])
    assert set(summary) >= {"run_id", "latency", "quality", "leveling", "slowdown"}
    assert summary["quality"]["stoi_mean"] is None
    assert summary["latency"]["budget_violation_count"] == 0
