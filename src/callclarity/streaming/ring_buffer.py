from __future__ import annotations

import torch


class RingBuffer:
    def __init__(self, capacity: int, channels: int = 1) -> None:
        self.capacity = int(capacity)
        self.channels = int(channels)
        self._data = torch.zeros(self.channels, self.capacity)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._data.zero_()
        self._size = 0

    def append(self, samples: torch.Tensor) -> None:
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[0] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {samples.shape[0]}")
        n = int(samples.shape[-1])
        if n >= self.capacity:
            self._data = samples[..., -self.capacity :].detach().clone()
            self._size = self.capacity
            return
        keep = min(self._size, self.capacity - n)
        if keep > 0:
            self._data[..., :keep] = self._data[..., self._size - keep : self._size].clone()
        self._data[..., keep : keep + n] = samples
        self._size = keep + n

    def read(self, n: int | None = None) -> torch.Tensor:
        n = self._size if n is None else min(int(n), self._size)
        return self._data[..., self._size - n : self._size].clone()
