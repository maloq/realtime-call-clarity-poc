from callclarity.metrics.aggregate import aggregate_run


def test_metrics_aggregation_handles_nulls():
    summary = aggregate_run(
        "r",
        "p",
        "d",
        [
            {"input_duration_sec": 1.0, "rtf": 0.1, "speech_frames_within_3db_ratio": None},
            {"input_duration_sec": 2.0, "rtf": None, "speech_frames_within_3db_ratio": 0.5},
        ],
    )
    assert summary["latency"]["rtf_mean"] == 0.1
    assert summary["leveling"]["speech_frames_within_3db_ratio"] == 0.5
