import torch

from callclarity.streaming.ring_buffer import RingBuffer


def test_ring_buffer_keeps_recent_samples():
    rb = RingBuffer(capacity=5, channels=1)
    rb.append(torch.tensor([[1.0, 2.0, 3.0]]))
    rb.append(torch.tensor([[4.0, 5.0, 6.0]]))
    assert rb.size == 5
    assert torch.allclose(rb.read(), torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]))
