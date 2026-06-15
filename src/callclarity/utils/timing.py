from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def timer_ms() -> Iterator[list[float]]:
    box = [0.0]
    start = time.perf_counter()
    try:
        yield box
    finally:
        box[0] = (time.perf_counter() - start) * 1000.0
