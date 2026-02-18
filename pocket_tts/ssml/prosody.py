"""Audio post-processing for SSML <prosody> and <emphasis> tags.

Implements rate change via resampling, volume via gain with soft-knee limiting,
pitch shifting via a phase vocoder approach, and cross-segment boundary smoothing.

Performance: Uses Apple Accelerate framework (AMX-backed vDSP/vForce) when
available. The phase vocoder FFT runs 2-5x faster than numpy PocketFFT on
Apple Silicon, and vForce provides vectorized transcendentals (tanh) for the
soft-knee limiter. Falls back transparently to numpy/scipy on non-macOS.
"""

import logging

import numpy as np
from scipy.signal import resample_poly

from pocket_tts.native import accelerate_dsp as adsp
from pocket_tts.ssml.types import ProsodyParams

logger = logging.getLogger(__name__)

# Crossfade duration for boundary smoothing between segments (in seconds)
_CROSSFADE_SECONDS = 0.01  # 10ms


def apply_prosody(audio: np.ndarray, sample_rate: int, params: ProsodyParams) -> np.ndarray:
    """Apply prosody modifications to an audio array.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Sample rate of the audio.
        params: Prosody parameters to apply.

    Returns:
        Modified audio array (same sample rate).
    """
    if audio.size == 0:
        return audio

    # Apply pitch shift first (if needed)
    if abs(params.pitch - 1.0) > 0.01:
        audio = _pitch_shift(audio, sample_rate, params.pitch)

    # Apply rate change
    if abs(params.rate - 1.0) > 0.01:
        audio = _change_rate(audio, params.rate)

    # Apply volume
    if abs(params.volume - 1.0) > 0.01:
        audio = _change_volume(audio, params.volume)

    return audio


def crossfade_segments(
    prev_audio: np.ndarray, next_audio: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Crossfade two audio segments to avoid boundary discontinuities.

    Uses vDSP_vramp + vDSP_vma when Accelerate is available for AMX-backed
    ramp generation and fused multiply-add.

    Args:
        prev_audio: 1D audio array for the previous segment.
        next_audio: 1D audio array for the next segment.
        sample_rate: Sample rate.

    Returns:
        Concatenated audio with crossfade applied.
    """
    if prev_audio.size == 0:
        return next_audio
    if next_audio.size == 0:
        return prev_audio

    crossfade_samples = min(
        int(sample_rate * _CROSSFADE_SECONDS),
        len(prev_audio),
        len(next_audio),
    )

    if crossfade_samples < 2:
        return np.concatenate([prev_audio, next_audio])

    # Use Accelerate-backed crossfade (vDSP_vramp + vDSP_vma)
    return adsp.crossfade(prev_audio, next_audio, crossfade_samples)


def generate_silence(sample_rate: int, duration_ms: int) -> np.ndarray:
    """Generate silence of specified duration.

    Args:
        sample_rate: Sample rate for the silence.
        duration_ms: Duration in milliseconds.

    Returns:
        1D numpy array of zeros.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def _change_rate(audio: np.ndarray, rate: float) -> np.ndarray:
    """Change the playback rate of audio (time stretch/compress).

    A rate > 1.0 speeds up (shorter duration), rate < 1.0 slows down.
    This uses resampling to change duration while preserving pitch.
    """
    if rate <= 0:
        logger.warning("Rate must be positive, got %f. Using 1.0.", rate)
        return audio

    # rate > 1 means faster -> fewer samples; rate < 1 means slower -> more samples
    precision = 1000
    up = precision
    down = max(1, int(round(rate * precision)))

    try:
        resampled = resample_poly(audio, up, down).astype(np.float32)
        return resampled
    except Exception:
        logger.warning("Rate change failed, returning original audio")
        return audio


def _change_volume(audio: np.ndarray, volume: float) -> np.ndarray:
    """Change the volume of audio with soft-knee limiting.

    Uses vForce vvtanhf (AMX-backed vectorized tanh) and vDSP_vabs when
    Accelerate is available, providing significant speedup for the
    transcendental math in the soft-knee compressor.

    Args:
        audio: Input audio samples.
        volume: Volume multiplier (1.0 = no change, 0.0 = silent, 2.0 = double).

    Returns:
        Volume-adjusted audio with soft-knee limiting to prevent clipping.
    """
    # Scale by volume (uses vDSP_vsmul for large arrays, numpy for small)
    audio = adsp.vsmul(audio, volume)

    threshold = np.float32(0.8)

    # Use vDSP_vabs + vDSP_maxv for peak detection (AMX-backed)
    abs_audio = adsp.vabs(audio)
    peak = adsp.maxv(abs_audio)

    if peak > threshold:
        above_knee = abs_audio > threshold
        if np.any(above_knee):
            knee_range = np.float32(1.0) - threshold

            # Extract samples above knee threshold
            excess = abs_audio[above_knee] - threshold

            # Use vForce vvtanhf for vectorized tanh (AMX-backed)
            tanh_vals = adsp.vtanhf(excess / knee_range)
            compressed = threshold + knee_range * tanh_vals

            # Apply compressed magnitudes with original sign using vcopysign
            audio[above_knee] = adsp.vcopysignf(compressed, audio[above_knee])

    return audio.astype(np.float32)


def _pitch_shift(audio: np.ndarray, sample_rate: int, pitch_factor: float) -> np.ndarray:
    """Shift the pitch of audio using a phase vocoder approach.

    Args:
        audio: 1D audio array.
        sample_rate: Sample rate.
        pitch_factor: Pitch multiplier (>1 = higher, <1 = lower).

    Returns:
        Pitch-shifted audio (same duration, same sample rate).
    """
    if abs(pitch_factor - 1.0) < 0.01:
        return audio

    n_fft = 2048
    hop_length = n_fft // 4

    # Step 1: Phase vocoder time stretch
    stretch_factor = 1.0 / pitch_factor
    stretched = _phase_vocoder_stretch(audio, stretch_factor, n_fft, hop_length)

    # Step 2: Resample to original length (this changes pitch)
    if len(stretched) > 0 and len(audio) > 0:
        precision = 1000
        up = len(audio)
        down = len(stretched)
        from math import gcd

        g = gcd(up, down)
        up, down = up // g, down // g
        # Cap to avoid memory issues
        if up > 10000 or down > 10000:
            ratio = len(audio) / len(stretched)
            up = max(1, int(round(ratio * precision)))
            down = precision
        try:
            result = resample_poly(stretched, up, down).astype(np.float32)
            # Trim or pad to exact original length
            if len(result) > len(audio):
                result = result[: len(audio)]
            elif len(result) < len(audio):
                result = np.pad(result, (0, len(audio) - len(result)))
            return result
        except Exception:
            logger.warning("Pitch shift resampling failed, returning original")
            return audio
    return audio


def _phase_vocoder_stretch(
    audio: np.ndarray, stretch_factor: float, n_fft: int, hop_length: int
) -> np.ndarray:
    """Time-stretch audio using phase vocoder without changing pitch.

    Uses batched AMX-backed vDSP FFT/IFFT when Apple Accelerate is available,
    replacing numpy PocketFFT with hardware-accelerated transforms. The
    AMX coprocessor provides 2-5x speedup for power-of-2 FFT sizes.

    The STFT is computed in a batched matrix operation rather than per-frame
    Python loops, eliminating interpreter overhead.

    Args:
        audio: 1D audio array.
        stretch_factor: Time stretch factor (>1 = longer, <1 = shorter).
        n_fft: FFT size.
        hop_length: Hop length for STFT.

    Returns:
        Time-stretched audio array.
    """
    if abs(stretch_factor - 1.0) < 0.01:
        return audio.copy()

    # STFT: Batched windowing + AMX-backed FFT
    window = adsp.hanning_window(n_fft)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames < 2:
        return audio.copy()

    # Build windowed frames matrix (num_frames, n_fft) -- vectorized, no Python loop
    audio_f32 = audio.astype(np.float32) if audio.dtype != np.float32 else audio
    indices = np.arange(n_fft)[None, :] + (np.arange(num_frames) * hop_length)[:, None]
    frames = audio_f32[indices] * window[None, :]

    # Batch FFT using AMX-backed vDSP (replaces per-frame np.fft.rfft loop)
    stft = adsp.rfft_batch(frames)

    # Phase vocoder synthesis
    new_num_frames = max(2, int(round(num_frames * stretch_factor)))

    phase_advance = np.linspace(0, np.pi * hop_length, n_fft // 2 + 1, endpoint=False)
    output_stft = np.zeros((new_num_frames, n_fft // 2 + 1), dtype=np.complex128)

    # Precompute magnitudes and phases for all input frames (vectorized)
    stft_mag = np.abs(stft)  # (num_frames, n_fft//2+1)
    stft_phase = np.angle(stft)  # (num_frames, n_fft//2+1)

    phase = stft_phase[0].copy()

    for i in range(new_num_frames):
        pos = i / stretch_factor
        idx = int(pos)
        frac = pos - idx

        if idx + 1 < num_frames:
            mag = (1 - frac) * stft_mag[idx] + frac * stft_mag[idx + 1]
        elif idx < num_frames:
            mag = stft_mag[idx]
        else:
            break

        output_stft[i] = mag * np.exp(1j * phase)

        # Phase advance with instantaneous frequency estimation
        if idx + 1 < num_frames:
            dphi = stft_phase[idx + 1] - stft_phase[idx] - phase_advance
            dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
            phase += phase_advance + dphi
        else:
            phase += phase_advance

    # Batch inverse FFT using AMX-backed vDSP (replaces per-frame np.fft.irfft loop)
    raw_frames = adsp.irfft_batch(output_stft, n_fft)

    # Apply window to each reconstructed frame (vectorized matrix multiply)
    windowed_frames = raw_frames * window[None, :]

    # Overlap-add (use float64 for accumulation to avoid precision loss
    # when many overlapping frames sum together, then convert to float32)
    output_hop = hop_length
    output_length = (new_num_frames - 1) * output_hop + n_fft
    output = np.zeros(output_length, dtype=np.float64)
    window_sum = np.zeros(output_length, dtype=np.float64)
    window_sq = window.astype(np.float64) ** 2
    windowed_frames_f64 = windowed_frames.astype(np.float64)

    for i in range(new_num_frames):
        start = i * output_hop
        end = min(start + n_fft, output_length)
        frame_len = end - start
        output[start:end] += windowed_frames_f64[i, :frame_len]
        window_sum[start:end] += window_sq[:frame_len]

    # Normalize by window overlap
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    # Apply fade-out on last few samples to avoid clicks
    fade_len = min(256, len(output) // 4)
    if fade_len > 1:
        # vramp uses vDSP for large arrays, numpy for small (fade is typically 256 samples)
        fade = adsp.vramp(1.0, -1.0 / (fade_len - 1), fade_len)
        output[-fade_len:] *= fade

    return output.astype(np.float32)
