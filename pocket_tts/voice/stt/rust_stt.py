"""Rust-native STT backend using libpocket_stt (candle + Metal).

Zero Python in the inference hot path. The Rust cdylib wraps Kyutai's
moshi::asr::State with candle Metal GPU acceleration, exposed through
C FFI functions called via ctypes.

Overhead per frame: ~2us ctypes dispatch + ~80ms Metal inference.
Compared to the PyTorch backend which adds ~1-2ms Python/executor overhead.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import platform
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from .base import STTBackend

logger = logging.getLogger(__name__)

_NATIVE_DIR = Path(__file__).resolve().parent.parent.parent / "native"
_CRATE_DIR = _NATIVE_DIR / "pocket_stt"
_LIB_NAME = "libpocket_stt.dylib" if sys.platform == "darwin" else "libpocket_stt.so"
_LIB_PATH = _NATIVE_DIR / _LIB_NAME

_stt_lib = None
_stt_available = False

DEFAULT_HF_REPO = "kyutai/stt-1b-en_fr-candle"
DEFAULT_MODEL_PATH = "model.safetensors"


def _build_stt_lib() -> bool:
    """Build the Rust STT cdylib if cargo is available."""
    if not _CRATE_DIR.exists() or not (_CRATE_DIR / "Cargo.toml").exists():
        logger.warning("pocket_stt: Rust crate not found at %s", _CRATE_DIR)
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        logger.info("pocket_stt: skipping build, not ARM64 (machine=%s)", platform.machine())
        return False

    try:
        subprocess.run(["cargo", "--version"], capture_output=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        logger.warning("pocket_stt: Rust toolchain (cargo) not found")
        return False

    logger.info("pocket_stt: building Rust STT library (first build takes ~60s)...")
    try:
        result = subprocess.run(
            ["cargo", "build", "--release"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_CRATE_DIR),
        )
        if result.returncode != 0:
            logger.warning("pocket_stt: cargo build failed:\n%s", result.stderr[-2000:])
            return False

        built = _CRATE_DIR / "target" / "release" / _LIB_NAME
        if built.exists():
            import shutil

            shutil.copy2(str(built), str(_LIB_PATH))
            logger.info("pocket_stt: library built and copied to %s", _LIB_PATH)
            return True

        logger.warning("pocket_stt: build succeeded but library not found at %s", built)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("pocket_stt: cargo build timed out (600s)")
        return False


def _load_stt_lib():
    global _stt_lib, _stt_available
    if _stt_available:
        return
    if not _LIB_PATH.exists():
        if not _build_stt_lib():
            return
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))

        _vp = ctypes.c_void_p
        _cp = ctypes.c_char_p
        _fp = ctypes.POINTER(ctypes.c_float)
        _dp = ctypes.POINTER(ctypes.c_double)
        _i = ctypes.c_int
        _f = ctypes.c_float
        _d = ctypes.c_double

        lib.pocket_stt_create.argtypes = [_cp, _cp, _i]
        lib.pocket_stt_create.restype = _vp

        lib.pocket_stt_destroy.argtypes = [_vp]
        lib.pocket_stt_destroy.restype = None

        lib.pocket_stt_process_frame.argtypes = [_vp, _fp, _i]
        lib.pocket_stt_process_frame.restype = _i

        lib.pocket_stt_flush.argtypes = [_vp]
        lib.pocket_stt_flush.restype = _i

        lib.pocket_stt_get_word.argtypes = [_vp, _i, _cp, _i, _dp, _dp]
        lib.pocket_stt_get_word.restype = _i

        lib.pocket_stt_get_all_text.argtypes = [_vp, _cp, _i]
        lib.pocket_stt_get_all_text.restype = _i

        lib.pocket_stt_get_vad_prob.argtypes = [_vp, _i]
        lib.pocket_stt_get_vad_prob.restype = _f

        lib.pocket_stt_has_vad.argtypes = [_vp]
        lib.pocket_stt_has_vad.restype = _i

        lib.pocket_stt_reset.argtypes = [_vp]
        lib.pocket_stt_reset.restype = None

        lib.pocket_stt_frame_size.argtypes = []
        lib.pocket_stt_frame_size.restype = _i

        lib.pocket_stt_sample_rate.argtypes = []
        lib.pocket_stt_sample_rate.restype = _i

        lib.pocket_stt_audio_delay.argtypes = [_vp]
        lib.pocket_stt_audio_delay.restype = _d

        _stt_lib = lib
        _stt_available = True
        logger.info("pocket_stt: native library loaded from %s", _LIB_PATH)
    except OSError as e:
        logger.warning("pocket_stt: failed to load library: %s", e)


_load_stt_lib()

_TEXT_BUF_SIZE = 4096
_float_p = ctypes.POINTER(ctypes.c_float)
_double_p = ctypes.POINTER(ctypes.c_double)


def _as_fp(arr: np.ndarray):
    return arr.ctypes.data_as(_float_p)


class RustSTT(STTBackend):
    """Streaming STT using the Rust-native candle + Metal backend.

    The model runs entirely in Rust on the Metal GPU. Python's only role
    is dispatching ctypes calls (~2us each) and routing text output.
    """

    def __init__(
        self,
        hf_repo: str = DEFAULT_HF_REPO,
        model_path: str = DEFAULT_MODEL_PATH,
        enable_vad: bool = True,
    ):
        if not _stt_available:
            raise RuntimeError(
                "Rust STT library not available. "
                "Ensure Rust toolchain is installed and run: "
                "cd pocket_tts/native/pocket_stt && cargo build --release"
            )
        self._hf_repo = hf_repo
        self._model_path = model_path
        self._enable_vad = enable_vad
        self._engine = None
        self._frame_size_samples = _stt_lib.pocket_stt_frame_size()
        self._sample_rate_hz = _stt_lib.pocket_stt_sample_rate()

    async def load(self):
        if self._engine is not None:
            return

        def _create():
            repo_bytes = self._hf_repo.encode("utf-8")
            path_bytes = self._model_path.encode("utf-8")
            engine = _stt_lib.pocket_stt_create(
                repo_bytes,
                path_bytes,
                1 if self._enable_vad else 0,
            )
            if not engine:
                raise RuntimeError(
                    f"Failed to create Rust STT engine for {self._hf_repo}. "
                    "Check stderr for details."
                )
            return engine

        self._engine = await asyncio.get_running_loop().run_in_executor(None, _create)
        logger.info(
            "Rust STT loaded: %s, frame_size=%d, sample_rate=%d, vad=%s",
            self._hf_repo,
            self._frame_size_samples,
            self._sample_rate_hz,
            self._enable_vad,
        )

    async def transcribe_stream(
        self, audio_frames: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        if self._engine is None:
            await self.load()

        buf = np.array([], dtype=np.float32)
        text_buf = ctypes.create_string_buffer(_TEXT_BUF_SIZE)

        async for frame in audio_frames:
            buf = np.concatenate([buf, frame])

            while len(buf) >= self._frame_size_samples:
                chunk = np.ascontiguousarray(buf[: self._frame_size_samples], dtype=np.float32)
                buf = buf[self._frame_size_samples :]

                n_words = _stt_lib.pocket_stt_process_frame(
                    self._engine, _as_fp(chunk), len(chunk)
                )

                if n_words > 0:
                    n = _stt_lib.pocket_stt_get_all_text(
                        self._engine, text_buf, _TEXT_BUF_SIZE
                    )
                    if n > 0:
                        yield text_buf.value.decode("utf-8", errors="replace")

    async def flush_remaining(self) -> str | None:
        """Feed silence to extract text delayed by the model's audio delay."""
        if self._engine is None:
            return None

        text_buf = ctypes.create_string_buffer(_TEXT_BUF_SIZE)

        def _flush():
            n_words = _stt_lib.pocket_stt_flush(self._engine)
            if n_words > 0:
                n = _stt_lib.pocket_stt_get_all_text(self._engine, text_buf, _TEXT_BUF_SIZE)
                if n > 0:
                    return text_buf.value.decode("utf-8", errors="replace")
            return None

        return await asyncio.get_running_loop().run_in_executor(None, _flush)

    def get_vad_prob(self, horizon: int = 2) -> float:
        """Get semantic VAD probability for the given time horizon.

        Horizons: 0=0.5s, 1=1.0s, 2=2.0s, 3=3.0s
        Returns probability of NO voice activity (higher = more silent).
        """
        if self._engine is None or not self._enable_vad:
            return -1.0
        return _stt_lib.pocket_stt_get_vad_prob(self._engine, horizon)

    @property
    def has_semantic_vad(self) -> bool:
        if self._engine is None:
            return self._enable_vad
        return _stt_lib.pocket_stt_has_vad(self._engine) != 0

    def reset(self):
        if self._engine is not None:
            _stt_lib.pocket_stt_reset(self._engine)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_hz

    @property
    def frame_size(self) -> int:
        return self._frame_size_samples

    def destroy(self):
        if self._engine is not None:
            _stt_lib.pocket_stt_destroy(self._engine)
            self._engine = None

    def __del__(self):
        self.destroy()


def is_available() -> bool:
    """Check if the Rust STT backend is available."""
    return _stt_available
