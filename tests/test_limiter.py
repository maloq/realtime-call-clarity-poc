import torch

from callclarity.dsp.limiter import limit_peak
from callclarity.dsp.envelope import db_to_linear


def test_limiter_prevents_clipping_above_ceiling():
    y, metrics = limit_peak(torch.tensor([[0.0, 2.0, -2.0]]), ceiling_dbfs=-1.5)
    assert float(y.abs().max()) <= db_to_linear(-1.5) + 1e-6
    assert metrics["limiter_gain_reduction_db"] < 0.0
