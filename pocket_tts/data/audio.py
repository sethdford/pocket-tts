"""
Audio IO methods are defined in this module (info, read, write).
Uses av library for faster read when possible, otherwise soundfile/wave.
"""

import logging
import os
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from beartype.typing import Iterator

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def audio_read(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Read audio file. WAV uses built-in wave module; other formats require soundfile.

    Returns:
        tuple of (audio_array, sample_rate) where audio_array has shape [1, samples].
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        with wave.open(str(filepath), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            raw_data = wav_file.readframes(-1)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return samples.reshape(1, -1), sample_rate

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        wav = data.reshape(1, -1)
    else:
        wav = data.mean(axis=1).reshape(1, -1)
    return wav, sample_rate


class StreamingWAVWriter:
    """WAV writer using Python's standard library wave module."""

    def __init__(self, output_stream, sample_rate: int):
        self.output_stream = output_stream
        self.sample_rate = sample_rate
        self.wave_writer = None
        self.first_chunk_buffer = []

    def write_header(self, sample_rate: int):
        """Initialize WAV writer with header."""
        self.wave_writer = wave.open(self.output_stream, "wb")
        self.wave_writer.setnchannels(1)
        self.wave_writer.setsampwidth(2)
        self.wave_writer.setframerate(sample_rate)
        self.wave_writer.setnframes(1_000_000_000)

    def write_pcm_data(self, audio_chunk: np.ndarray):
        """Write PCM data using wave module. audio_chunk is a 1D numpy array.

        Uses NEON SIMD when available for single-pass float32→int16 conversion.
        Falls back to numpy otherwise.
        """
        from pocket_tts.native import pcm_convert

        chunk_bytes = pcm_convert(audio_chunk)

        if self.first_chunk_buffer is not None:
            self.first_chunk_buffer.append(chunk_bytes)
            total_length = sum(len(c) for c in self.first_chunk_buffer)
            target_length = int(self.sample_rate * FIRST_CHUNK_LENGTH_SECONDS) * 2
            if total_length < target_length:
                return
            self._flush()
            return

        self.wave_writer.writeframesraw(chunk_bytes)

    def _flush(self):
        if self.first_chunk_buffer is not None:
            self.wave_writer.writeframesraw(b"".join(self.first_chunk_buffer))
            self.first_chunk_buffer = None

    def finalize(self):
        """Close the wave writer."""
        self._flush()

        silence_duration_sec = 0.2
        num_silence_samples = int(self.sample_rate * silence_duration_sec)
        self.wave_writer.writeframesraw(bytes(num_silence_samples * 2))

        if self.wave_writer:
            self.wave_writer._patchheader = lambda: None
            self.wave_writer.close()


def is_file_like(obj):
    """Check if object has basic file-like methods."""
    return all(hasattr(obj, attr) for attr in ["write", "close"])


def stream_audio_chunks(
    path: str | Path | None | Any, audio_chunks: Iterator[np.ndarray], sample_rate: int
):
    """Stream audio chunks to a WAV file or stdout, optionally playing them."""
    if path == "-":
        f = sys.stdout.buffer
    elif path is None:
        f = nullcontext()
    elif is_file_like(path):
        f = path
    else:
        f = open(path, "wb")

    with f:
        if path is not None:
            writer = StreamingWAVWriter(f, sample_rate)
            writer.write_header(sample_rate)

        for chunk in audio_chunks:
            if path is not None:
                writer.write_pcm_data(chunk)

        if path is not None:
            writer.finalize()
