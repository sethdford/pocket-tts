"""Abstract base class for speech-to-text backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import numpy as np


class STTBackend(ABC):
    """Interface for streaming speech-to-text backends.

    Implementations must support streaming: they accept audio frames
    incrementally and yield partial text as it becomes available.
    """

    @abstractmethod
    async def load(self):
        """Load model weights. Called once before any transcription."""

    @abstractmethod
    async def transcribe_stream(
        self, audio_frames: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        """Transcribe a stream of audio frames into text chunks.

        Args:
            audio_frames: Async iterator of float32 numpy arrays at 24kHz mono.
                Each frame is typically 1920 samples (80ms at 24kHz).

        Yields:
            Text chunks as they are recognized (word or sub-word level).
        """
        yield ""  # pragma: no cover

    @abstractmethod
    def reset(self):
        """Reset internal state for a new utterance."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected input sample rate in Hz."""

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Expected frame size in samples."""
