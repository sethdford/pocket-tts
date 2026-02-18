"""Native NEON SIMD-accelerated audio processing.

Compiles a small C library on first import using the system C compiler,
then loads it via ctypes. Falls back to numpy if compilation fails
(e.g., on non-ARM or missing compiler).

The C library provides:
  - float32_to_pcm16: NEON-vectorized clip+scale+cast (8 samples/cycle)
  - float32_to_pcm16_bytes: Same but writes directly to a byte buffer
"""

import ctypes
import logging
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_LIB_DIR = Path(__file__).parent
_C_SOURCE = _LIB_DIR / "simd_audio.c"
_LIB_NAME = "libsimd_audio.dylib" if sys.platform == "darwin" else "libsimd_audio.so"
_LIB_PATH = _LIB_DIR / _LIB_NAME

_native_lib = None
_native_available = False


def _compile_native_lib() -> bool:
    """Compile the NEON SIMD C library using the system compiler."""
    if not _C_SOURCE.exists():
        return False

    if platform.machine() not in ("arm64", "aarch64"):
        logger.info("Native SIMD: skipping, not ARM64 (machine=%s)", platform.machine())
        return False

    try:
        cmd = [
            "cc",
            "-O3",
            "-shared",
            "-fPIC",
            "-arch",
            "arm64",
            "-o",
            str(_LIB_PATH),
            str(_C_SOURCE),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("Native SIMD compilation failed: %s", result.stderr.strip())
            return False
        logger.info("Native SIMD library compiled: %s", _LIB_PATH)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("Native SIMD compilation unavailable: %s", e)
        return False


def _load_native_lib():
    """Load the compiled native library."""
    global _native_lib, _native_available

    if _native_available:
        return

    # Compile if needed
    if not _LIB_PATH.exists():
        if not _compile_native_lib():
            return

    try:
        _native_lib = ctypes.CDLL(str(_LIB_PATH))

        # Set up float32_to_pcm16 signature
        _native_lib.float32_to_pcm16.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_int16),  # output
            ctypes.c_size_t,  # n
        ]
        _native_lib.float32_to_pcm16.restype = None

        # Set up float32_to_pcm16_bytes signature
        _native_lib.float32_to_pcm16_bytes.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_uint8),  # output_bytes
            ctypes.c_size_t,  # n
        ]
        _native_lib.float32_to_pcm16_bytes.restype = None

        _native_available = True
        logger.info("Native SIMD library loaded successfully")
    except OSError as e:
        logger.warning("Failed to load native SIMD library: %s", e)


# Attempt to load on import
_load_native_lib()


def pcm_convert(audio_chunk: np.ndarray) -> bytes:
    """Convert float32 audio to int16 PCM bytes.

    Uses NEON SIMD when available (single-pass, zero intermediate allocations).
    Falls back to numpy (3 passes, 3 allocations) otherwise.

    Args:
        audio_chunk: 1D float32 numpy array of audio samples.

    Returns:
        bytes: PCM int16 data ready for WAV writing.
    """
    if _native_available and audio_chunk.dtype == np.float32:
        n = audio_chunk.size
        # Ensure contiguous
        if not audio_chunk.flags["C_CONTIGUOUS"]:
            audio_chunk = np.ascontiguousarray(audio_chunk)

        # Allocate output buffer (int16 = 2 bytes per sample)
        out_bytes = bytearray(n * 2)
        out_ptr = (ctypes.c_uint8 * len(out_bytes)).from_buffer(out_bytes)
        in_ptr = audio_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _native_lib.float32_to_pcm16_bytes(in_ptr, out_ptr, n)
        return bytes(out_bytes)

    # Fallback: numpy path (3 allocations, 3 passes)
    chunk_int16 = np.clip(audio_chunk, -1.0, 1.0)
    chunk_int16 = (chunk_int16 * 32767).astype(np.int16)
    return chunk_int16.tobytes()


def pcm_convert_to_array(audio_chunk: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int16 PCM numpy array.

    Uses NEON SIMD when available.

    Args:
        audio_chunk: 1D float32 numpy array.

    Returns:
        np.ndarray: int16 PCM array.
    """
    if _native_available and audio_chunk.dtype == np.float32:
        n = audio_chunk.size
        if not audio_chunk.flags["C_CONTIGUOUS"]:
            audio_chunk = np.ascontiguousarray(audio_chunk)

        out = np.empty(n, dtype=np.int16)
        in_ptr = audio_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

        _native_lib.float32_to_pcm16(in_ptr, out_ptr, n)
        return out

    chunk_int16 = np.clip(audio_chunk, -1.0, 1.0)
    return (chunk_int16 * 32767).astype(np.int16)


def is_available() -> bool:
    """Check if native SIMD acceleration is available."""
    return _native_available
