from callclarity.metrics.guardrails import evaluate_guardrails


def test_guardrails_warn_when_noise_improves_but_speech_regresses():
    warnings = evaluate_guardrails(
        {
            "dnsmos_bak_delta": 0.3,
            "dnsmos_sig_delta": -0.2,
            "input_clipping_pct": 0.0,
            "output_clipping_pct": 0.0,
            "processing_rtf": 0.1,
        }
    )
    assert any(warning["rule"] == "noise_improved_speech_regressed" for warning in warnings)


def test_guardrails_error_when_rtf_exceeds_budget():
    warnings = evaluate_guardrails({"processing_rtf": 1.2}, {"max_processing_rtf": 0.85})
    assert any(warning["rule"] == "rtf_too_slow" and warning["severity"] == "error" for warning in warnings)
