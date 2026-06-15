from callclarity.methods.slowdown.latency_controller import SlowdownLatencyController


def test_slowdown_controller_never_exceeds_hard_buffer():
    controller = SlowdownLatencyController(
        {
            "target": {"max_added_latency_ms": 160, "preferred_buffer_ms": 70, "hard_buffer_ms": 35},
            "tempo": {"min_tempo": 0.5, "burst_min_tempo": 0.5, "smoothing_per_100ms": 1.0},
            "decision": {"fast_rate_threshold_syllables_per_sec": 5.5, "min_confidence": 0.1, "min_speech_prob": 0.1},
        }
    )
    rate = {"is_fast": True, "confidence": 1.0, "syllables_per_sec": 11.0}
    for idx in range(1000):
        decision = controller.decide(idx * 0.01, input_ms=10.0, speech_prob=1.0, rate=rate)
        assert decision.buffer_ms <= 35.0 + 1e-6
