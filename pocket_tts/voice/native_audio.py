"""Python wrapper for the native C voice engine (libpocket_voice).

Provides ultra-low-latency audio I/O via CoreAudio VoiceProcessingIO
with built-in echo cancellation, lock-free ring buffers, and energy VAD.
Falls back to sounddevice if compilation fails (non-ARM or missing frameworks).

The C engine handles the entire audio path on CoreAudio's real-time thread —
no Python, no GIL, no allocations in the hot path.
"""

import ctypes
import logging
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_LIB_DIR = Path(__file__).resolve().parent.parent / "native"
_C_SOURCE = _LIB_DIR / "pocket_voice.c"
_LIB_NAME = "libpocket_voice.dylib" if sys.platform == "darwin" else "libpocket_voice.so"
_LIB_PATH = _LIB_DIR / _LIB_NAME

_native_lib = None
_native_available = False

VAD_SILENCE = 0
VAD_SPEECH_START = 1
VAD_SPEECH = 2
VAD_SPEECH_END = 3


def _compile_lib() -> bool:
    if not _C_SOURCE.exists():
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        logger.info("pocket_voice: skipping, not ARM64 (machine=%s)", platform.machine())
        return False
    if sys.platform != "darwin":
        logger.info("pocket_voice: skipping, CoreAudio requires macOS")
        return False
    try:
        cmd = [
            "cc",
            "-O3",
            "-shared",
            "-fPIC",
            "-arch",
            "arm64",
            "-framework",
            "Accelerate",
            "-framework",
            "CoreAudio",
            "-framework",
            "AudioToolbox",
            "-o",
            str(_LIB_PATH),
            str(_C_SOURCE),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("pocket_voice compilation failed: %s", result.stderr.strip())
            return False
        logger.info("pocket_voice library compiled: %s", _LIB_PATH)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("pocket_voice compilation unavailable: %s", e)
        return False


def _load_lib():
    global _native_lib, _native_available
    if _native_available:
        return
    if not _LIB_PATH.exists():
        if not _compile_lib():
            return
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))

        _void_p = ctypes.c_void_p
        _float_p = ctypes.POINTER(ctypes.c_float)
        _int = ctypes.c_int
        _uint32 = ctypes.c_uint32
        _float = ctypes.c_float

        lib.voice_engine_create.argtypes = [_uint32, _uint32]
        lib.voice_engine_create.restype = _void_p

        lib.voice_engine_start.argtypes = [_void_p]
        lib.voice_engine_start.restype = _int

        lib.voice_engine_stop.argtypes = [_void_p]
        lib.voice_engine_stop.restype = None

        lib.voice_engine_destroy.argtypes = [_void_p]
        lib.voice_engine_destroy.restype = None

        lib.voice_engine_read_capture.argtypes = [_void_p, _float_p, _int]
        lib.voice_engine_read_capture.restype = _int

        lib.voice_engine_write_playback.argtypes = [_void_p, _float_p, _int]
        lib.voice_engine_write_playback.restype = _int

        lib.voice_engine_flush_playback.argtypes = [_void_p]
        lib.voice_engine_flush_playback.restype = None

        lib.voice_engine_is_playing.argtypes = [_void_p]
        lib.voice_engine_is_playing.restype = _int

        lib.voice_engine_get_vad_state.argtypes = [_void_p]
        lib.voice_engine_get_vad_state.restype = _int

        lib.voice_engine_get_barge_in.argtypes = [_void_p]
        lib.voice_engine_get_barge_in.restype = _int

        lib.voice_engine_clear_barge_in.argtypes = [_void_p]
        lib.voice_engine_clear_barge_in.restype = None

        lib.voice_engine_set_vad_thresholds.argtypes = [_void_p, _float, _float]
        lib.voice_engine_set_vad_thresholds.restype = None

        lib.voice_engine_capture_available.argtypes = [_void_p]
        lib.voice_engine_capture_available.restype = _int

        lib.voice_engine_playback_available.argtypes = [_void_p]
        lib.voice_engine_playback_available.restype = _int

        lib.voice_engine_resample_48_to_24.argtypes = [_float_p, _float_p, _int]
        lib.voice_engine_resample_48_to_24.restype = None

        lib.voice_engine_resample_24_to_48.argtypes = [_float_p, _float_p, _int]
        lib.voice_engine_resample_24_to_48.restype = None

        _native_lib = lib
        _native_available = True
        logger.info("pocket_voice native library loaded")
    except OSError as e:
        logger.warning("Failed to load pocket_voice library: %s", e)


_load_lib()

_float_p = ctypes.POINTER(ctypes.c_float)


def _as_fp(arr: np.ndarray):
    return arr.ctypes.data_as(_float_p)


class NativeVoiceEngine:
    """High-level wrapper around the C voice engine.

    Provides CoreAudio VoiceProcessingIO with built-in AEC,
    lock-free ring buffers, energy-based VAD, and audio resampling.
    Falls back to sounddevice if the native engine is unavailable.
    """

    def __init__(self, sample_rate: int = 48000, buffer_frames: int = 256):
        self._sample_rate = sample_rate
        self._buffer_frames = buffer_frames
        self._engine = None
        self._fallback = False
        self._sd_stream = None

    @property
    def native_available(self) -> bool:
        return _native_available

    def start(self):
        if _native_available:
            self._engine = _native_lib.voice_engine_create(
                ctypes.c_uint32(self._sample_rate), ctypes.c_uint32(self._buffer_frames)
            )
            if not self._engine:
                raise RuntimeError("Failed to create native voice engine")
            ret = _native_lib.voice_engine_start(self._engine)
            if ret != 0:
                _native_lib.voice_engine_destroy(self._engine)
                self._engine = None
                raise RuntimeError("Failed to start native voice engine")
            logger.info(
                "Native voice engine started: %dHz, %d-frame buffer",
                self._sample_rate,
                self._buffer_frames,
            )
        else:
            self._start_fallback()

    def _start_fallback(self):
        """Fallback to sounddevice for audio I/O."""
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "Native voice engine unavailable and sounddevice not installed. "
                "Install with: pip install sounddevice"
            ) from None

        self._fallback = True
        self._capture_buf = np.zeros(65536, dtype=np.float32)
        self._capture_write = 0
        self._capture_read = 0
        self._playback_buf = np.zeros(65536, dtype=np.float32)
        self._playback_write = 0
        self._playback_read = 0

        def capture_cb(indata, frames, time_info, status):
            data = indata[:, 0].astype(np.float32)
            n = len(data)
            start = self._capture_write % len(self._capture_buf)
            end = start + n
            if end <= len(self._capture_buf):
                self._capture_buf[start:end] = data
            else:
                first = len(self._capture_buf) - start
                self._capture_buf[start:] = data[:first]
                self._capture_buf[: n - first] = data[first:]
            self._capture_write += n

        def playback_cb(outdata, frames, time_info, status):
            avail = self._playback_write - self._playback_read
            n = min(frames, avail)
            if n > 0:
                start = self._playback_read % len(self._playback_buf)
                end = start + n
                if end <= len(self._playback_buf):
                    outdata[:n, 0] = self._playback_buf[start:end]
                else:
                    first = len(self._playback_buf) - start
                    outdata[:first, 0] = self._playback_buf[start:]
                    outdata[first:n, 0] = self._playback_buf[: n - first]
                self._playback_read += n
            if n < frames:
                outdata[n:, 0] = 0.0

        self._sd_input = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._buffer_frames,
            callback=capture_cb,
        )
        self._sd_output = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._buffer_frames,
            callback=playback_cb,
        )
        self._sd_input.start()
        self._sd_output.start()
        logger.info("Fallback sounddevice engine started: %dHz", self._sample_rate)

    def stop(self):
        if self._engine:
            _native_lib.voice_engine_stop(self._engine)
        elif self._fallback:
            self._sd_input.stop()
            self._sd_output.stop()

    def destroy(self):
        if self._engine:
            _native_lib.voice_engine_destroy(self._engine)
            self._engine = None
        elif self._fallback:
            self._sd_input.close()
            self._sd_output.close()
            self._fallback = False

    def read_capture(self, max_frames: int = 4096) -> np.ndarray:
        """Read captured mic audio. Returns float32 array (may be empty)."""
        buf = np.empty(max_frames, dtype=np.float32)
        if self._engine:
            n = _native_lib.voice_engine_read_capture(self._engine, _as_fp(buf), max_frames)
            return buf[:n]
        elif self._fallback:
            avail = self._capture_write - self._capture_read
            n = min(max_frames, avail)
            if n <= 0:
                return np.array([], dtype=np.float32)
            start = self._capture_read % len(self._capture_buf)
            end = start + n
            if end <= len(self._capture_buf):
                buf[:n] = self._capture_buf[start:end]
            else:
                first = len(self._capture_buf) - start
                buf[:first] = self._capture_buf[start:]
                buf[first:n] = self._capture_buf[: n - first]
            self._capture_read += n
            return buf[:n]
        return np.array([], dtype=np.float32)

    def write_playback(self, audio: np.ndarray) -> int:
        """Write TTS audio for playback. Returns 0 on success."""
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if self._engine:
            return _native_lib.voice_engine_write_playback(self._engine, _as_fp(audio), len(audio))
        elif self._fallback:
            n = len(audio)
            start = self._playback_write % len(self._playback_buf)
            end = start + n
            if end <= len(self._playback_buf):
                self._playback_buf[start:end] = audio
            else:
                first = len(self._playback_buf) - start
                self._playback_buf[start:] = audio[:first]
                self._playback_buf[: n - first] = audio[first:]
            self._playback_write += n
            return 0
        return -1

    def flush_playback(self):
        """Instantly silence playback (for barge-in)."""
        if self._engine:
            _native_lib.voice_engine_flush_playback(self._engine)
        elif self._fallback:
            self._playback_read = self._playback_write

    def is_playing(self) -> bool:
        if self._engine:
            return _native_lib.voice_engine_is_playing(self._engine) != 0
        elif self._fallback:
            return self._playback_write > self._playback_read
        return False

    def get_vad_state(self) -> int:
        if self._engine:
            return _native_lib.voice_engine_get_vad_state(self._engine)
        return VAD_SILENCE

    def get_barge_in(self) -> bool:
        if self._engine:
            return _native_lib.voice_engine_get_barge_in(self._engine) != 0
        return False

    def clear_barge_in(self):
        if self._engine:
            _native_lib.voice_engine_clear_barge_in(self._engine)

    def set_vad_thresholds(self, energy: float = 0.01, silence: float = 0.005):
        if self._engine:
            _native_lib.voice_engine_set_vad_thresholds(
                self._engine, ctypes.c_float(energy), ctypes.c_float(silence)
            )

    def capture_available(self) -> int:
        if self._engine:
            return _native_lib.voice_engine_capture_available(self._engine)
        elif self._fallback:
            return self._capture_write - self._capture_read
        return 0

    def resample_48_to_24(self, audio: np.ndarray) -> np.ndarray:
        """Downsample 48kHz audio to 24kHz using vDSP."""
        if _native_available:
            out = np.empty(len(audio) // 2, dtype=np.float32)
            inp = np.ascontiguousarray(audio, dtype=np.float32)
            _native_lib.voice_engine_resample_48_to_24(_as_fp(inp), _as_fp(out), len(inp))
            return out
        from scipy.signal import resample_poly

        return resample_poly(audio, 1, 2).astype(np.float32)

    def resample_24_to_48(self, audio: np.ndarray) -> np.ndarray:
        """Upsample 24kHz audio to 48kHz using vDSP."""
        if _native_available:
            out = np.empty(len(audio) * 2, dtype=np.float32)
            inp = np.ascontiguousarray(audio, dtype=np.float32)
            _native_lib.voice_engine_resample_24_to_48(_as_fp(inp), _as_fp(out), len(inp))
            return out
        from scipy.signal import resample_poly

        return resample_poly(audio, 2, 1).astype(np.float32)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.destroy()
