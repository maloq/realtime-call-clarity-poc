import builtins

import torch

from callclarity.metrics.optional_quality import optional_no_reference_quality


def test_optional_quality_metrics_disabled_return_expected_keys():
    row = optional_no_reference_quality(torch.zeros(1, 16000), 16000, {})
    assert row["nisqa_mos"] is None
    assert row["dnsmos_ovrl"] is None
    assert row["squim_stoi_est"] is None
    assert row["nisqa_status"] == "disabled"


def test_missing_squim_dependency_is_reported_as_skip(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("torchaudio"):
            raise ImportError("torchaudio intentionally hidden")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    row = optional_no_reference_quality(torch.zeros(1, 16000), 16000, {"squim": {"enabled": True}})
    assert row["squim_status"] == "skipped"
    assert "torchaudio intentionally hidden" in row["squim_error"]
