from __future__ import annotations

from collections.abc import Iterator, Sequence

from callclarity.types import DatasetItem


class AudioDataset(Sequence[DatasetItem]):
    def __iter__(self) -> Iterator[DatasetItem]:
        for idx in range(len(self)):
            yield self[idx]
