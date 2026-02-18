"""Various utilities for audio conversion (pcm format, sample rate and channels),
and volume normalization."""

import math

import numpy as np
from scipy.signal import resample_poly


def convert_audio(
    wav: np.ndarray, from_rate: int | float, to_rate: int | float, to_channels: int
) -> np.ndarray:
    """Convert audio to new sample rate and number of audio channels.

    Args:
        wav: Audio array of shape [channels, samples].
        from_rate: Source sample rate.
        to_rate: Target sample rate.
        to_channels: Target number of channels.

    Returns:
        Resampled audio array.
    """
    if from_rate != to_rate:
        gcd = math.gcd(int(from_rate), int(to_rate))
        up = int(to_rate // gcd)
        down = int(from_rate // gcd)
        wav = resample_poly(wav, up, down, axis=-1).astype(np.float32)

    assert wav.shape[-2] == to_channels
    return wav
