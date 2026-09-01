from typing import List

import numpy as np


class AudioBuffer:
    """Accumulates float32 audio chunks for one client.

    Chunks are held in a list and joined once on read. The previous approach
    called np.append per chunk, which reallocates and copies the whole buffer
    every time - O(n^2) over an utterance, on the event loop.
    """

    __slots__ = ("_chunks",)

    def __init__(self) -> None:
        self._chunks: List[np.ndarray] = []

    def append(self, chunk: np.ndarray) -> None:
        if chunk.size:
            self._chunks.append(chunk)

    def take(self) -> np.ndarray:
        """Return everything buffered so far and reset."""
        if not self._chunks:
            return np.array([], dtype=np.float32)
        joined = np.concatenate(self._chunks)
        self._chunks.clear()
        return joined

    def clear(self) -> None:
        self._chunks.clear()
