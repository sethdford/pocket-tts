"""Apple Accelerate framework bindings for AMX-backed DSP operations.

Wraps Apple's Accelerate framework (vDSP, vForce, BLAS) via ctypes to leverage
the AMX coprocessor (M1-M3) or ARM SME (M4+) for audio processing operations.

Size-aware routing: For small arrays (< _MIN_ACCEL_SIZE), numpy is faster due
to ctypes marshalling overhead (~10-20us per call). The AMX advantage emerges
for larger arrays (n > ~8192) where compute dominates over call overhead.

Functions:
  - vDSP FFT:   AMX-backed FFT for large transforms (n >= 8192)
  - vDSP ops:   Vectorized multiply, add, ramp, clip (large arrays)
  - vForce:     Vectorized transcendentals (tanh, abs) for large arrays
  - Convolution: vDSP_conv, vDSP_desamp for signal processing

All functions transparently fall back to numpy for small arrays or non-macOS.
"""

import ctypes
import ctypes.util
import logging
import sys
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Accelerate framework
# ---------------------------------------------------------------------------
_accelerate = None
_vforce = None
_available = False

if sys.platform == "darwin":
    try:
        _accel_path = ctypes.util.find_library("Accelerate")
        if _accel_path:
            _accelerate = ctypes.CDLL(_accel_path)
        else:
            _accelerate = ctypes.CDLL(
                "/System/Library/Frameworks/Accelerate.framework/Accelerate"
            )

        _vforce_path = ctypes.util.find_library("vecLib")
        if _vforce_path:
            _vforce = ctypes.CDLL(_vforce_path)
        else:
            _vforce = ctypes.CDLL(
                "/System/Library/Frameworks/Accelerate.framework/"
                "Frameworks/vecLib.framework/vecLib"
            )

        _available = True
        logger.info("Apple Accelerate framework loaded (AMX-backed DSP enabled)")
    except OSError as e:
        logger.info("Apple Accelerate not available: %s", e)


def is_available() -> bool:
    """Check if Apple Accelerate DSP acceleration is available."""
    return _available


# Minimum array size for Accelerate to outperform numpy (due to ctypes overhead).
# Below this threshold, numpy's optimized C internals are faster because they
# avoid the Python->ctypes->C marshalling cost (~10-20us per call).
# Above this, AMX compute advantage dominates.
_MIN_ACCEL_SIZE = 8192


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_uint = ctypes.c_uint
_c_ulong = ctypes.c_ulong  # vDSP_Length
_c_long = ctypes.c_long  # vDSP_Stride

# vDSP_DFT_Direction enum values
_DFT_FORWARD = ctypes.c_int(1)  # vDSP_DFT_FORWARD
_DFT_INVERSE = ctypes.c_int(-1)  # vDSP_DFT_INVERSE


# ---------------------------------------------------------------------------
# vDSP function signatures
# ---------------------------------------------------------------------------
def _setup_vdsp_signatures():
    """Configure ctypes function signatures for vDSP calls."""
    if not _available:
        return

    # vDSP_DFT_zop_CreateSetup(previous, length, direction) -> setup
    _accelerate.vDSP_DFT_zop_CreateSetup.argtypes = [
        ctypes.c_void_p,  # previous setup (NULL for new)
        _c_ulong,  # length
        ctypes.c_int,  # direction
    ]
    _accelerate.vDSP_DFT_zop_CreateSetup.restype = ctypes.c_void_p

    # vDSP_DFT_Execute(setup, ir, ii, or, oi)
    _accelerate.vDSP_DFT_Execute.argtypes = [
        ctypes.c_void_p,  # setup
        _c_float_p,  # input real
        _c_float_p,  # input imag
        _c_float_p,  # output real
        _c_float_p,  # output imag
    ]
    _accelerate.vDSP_DFT_Execute.restype = None

    # vDSP_DFT_DestroySetup(setup)
    _accelerate.vDSP_DFT_DestroySetup.argtypes = [ctypes.c_void_p]
    _accelerate.vDSP_DFT_DestroySetup.restype = None

    # vDSP_vmul: vector multiply C[i] = A[i] * B[i]
    _accelerate.vDSP_vmul.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p, _c_long,  # B, stride
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vmul.restype = None

    # vDSP_vma: vector multiply-add D[i] = A[i]*B[i] + C[i]
    _accelerate.vDSP_vma.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p, _c_long,  # B, stride
        _c_float_p, _c_long,  # C, stride
        _c_float_p, _c_long,  # D, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vma.restype = None

    # vDSP_vadd: vector add C[i] = A[i] + B[i]
    _accelerate.vDSP_vadd.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p, _c_long,  # B, stride
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vadd.restype = None

    # vDSP_vsub: vector subtract C[i] = B[i] - A[i]   (note: B-A, not A-B)
    _accelerate.vDSP_vsub.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p, _c_long,  # B, stride
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vsub.restype = None

    # vDSP_vsmul: vector-scalar multiply C[i] = A[i] * scalar
    _accelerate.vDSP_vsmul.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p,  # scalar (by reference)
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vsmul.restype = None

    # vDSP_vramp: generate ramp O[i] = start + i * step
    _accelerate.vDSP_vramp.argtypes = [
        _c_float_p,  # start (by reference)
        _c_float_p,  # step (by reference)
        _c_float_p, _c_long,  # output, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vramp.restype = None

    # vDSP_vclip: clip vector to [low, high]
    _accelerate.vDSP_vclip.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p,  # low (by reference)
        _c_float_p,  # high (by reference)
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vclip.restype = None

    # vDSP_maxv: maximum of vector
    _accelerate.vDSP_maxv.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p,  # result (by reference)
        _c_ulong,  # N
    ]
    _accelerate.vDSP_maxv.restype = None

    # vDSP_vabs: absolute value C[i] = |A[i]|
    _accelerate.vDSP_vabs.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vabs.restype = None

    # vDSP_vdiv: vector divide C[i] = B[i] / A[i]  (note: B/A, not A/B)
    _accelerate.vDSP_vdiv.argtypes = [
        _c_float_p, _c_long,  # A (divisor), stride
        _c_float_p, _c_long,  # B (dividend), stride
        _c_float_p, _c_long,  # C (result), stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vdiv.restype = None

    # vDSP_sve: sum of vector elements
    _accelerate.vDSP_sve.argtypes = [
        _c_float_p, _c_long,  # A, stride
        _c_float_p,  # result (by reference)
        _c_ulong,  # N
    ]
    _accelerate.vDSP_sve.restype = None

    # vDSP_vfill: fill vector with scalar
    _accelerate.vDSP_vfill.argtypes = [
        _c_float_p,  # scalar (by reference)
        _c_float_p, _c_long,  # C, stride
        _c_ulong,  # N
    ]
    _accelerate.vDSP_vfill.restype = None

    # vDSP_conv: 1D convolution
    _accelerate.vDSP_conv.argtypes = [
        _c_float_p, _c_long,  # signal, stride
        _c_float_p, _c_long,  # filter, stride
        _c_float_p, _c_long,  # result, stride
        _c_ulong,  # result length
        _c_ulong,  # filter length
    ]
    _accelerate.vDSP_conv.restype = None

    # vDSP_desamp: decimation with FIR filter
    _accelerate.vDSP_desamp.argtypes = [
        _c_float_p,  # input
        _c_long,  # decimation factor I
        _c_float_p,  # filter coefficients
        _c_float_p,  # output
        _c_ulong,  # output count N
        _c_ulong,  # filter length P
    ]
    _accelerate.vDSP_desamp.restype = None


def _setup_blas_signatures():
    """Configure ctypes function signatures for BLAS calls (AMX-backed GEMV/GEMM)."""
    if not _available:
        return

    # CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112
    _c_enum = ctypes.c_int
    _c_blasint = ctypes.c_int  # CBLAS uses int for dimensions

    # cblas_sgemv: matrix-vector multiply  y = alpha*A*x + beta*y
    _accelerate.cblas_sgemv.argtypes = [
        _c_enum,  # order (CblasRowMajor=101)
        _c_enum,  # trans (CblasNoTrans=111, CblasTrans=112)
        _c_blasint,  # M (rows of A)
        _c_blasint,  # N (cols of A)
        ctypes.c_float,  # alpha
        _c_float_p,  # A
        _c_blasint,  # lda
        _c_float_p,  # x
        _c_blasint,  # incx
        ctypes.c_float,  # beta
        _c_float_p,  # y
        _c_blasint,  # incy
    ]
    _accelerate.cblas_sgemv.restype = None

    # cblas_sgemm: matrix-matrix multiply  C = alpha*A*B + beta*C
    _accelerate.cblas_sgemm.argtypes = [
        _c_enum,  # order
        _c_enum,  # transA
        _c_enum,  # transB
        _c_blasint,  # M
        _c_blasint,  # N
        _c_blasint,  # K
        ctypes.c_float,  # alpha
        _c_float_p,  # A
        _c_blasint,  # lda
        _c_float_p,  # B
        _c_blasint,  # ldb
        ctypes.c_float,  # beta
        _c_float_p,  # C
        _c_blasint,  # ldc
    ]
    _accelerate.cblas_sgemm.restype = None

    # cblas_saxpy: y = alpha*x + y (vector add with scalar)
    _accelerate.cblas_saxpy.argtypes = [
        _c_blasint,  # N
        ctypes.c_float,  # alpha
        _c_float_p,  # x
        _c_blasint,  # incx
        _c_float_p,  # y
        _c_blasint,  # incy
    ]
    _accelerate.cblas_saxpy.restype = None


def _setup_vforce_signatures():
    """Configure ctypes function signatures for vForce calls."""
    if not _vforce:
        return

    # vvtanhf(output, input, &n) -- vectorized tanh for float32
    _vforce.vvtanhf.argtypes = [_c_float_p, _c_float_p, _c_int_p]
    _vforce.vvtanhf.restype = None

    # vvfabsf(output, input, &n) -- vectorized absolute value for float32
    _vforce.vvfabsf.argtypes = [_c_float_p, _c_float_p, _c_int_p]
    _vforce.vvfabsf.restype = None

    # vvcopysignf(output, magnitude, sign, &n) -- copy sign
    _vforce.vvcopysignf.argtypes = [_c_float_p, _c_float_p, _c_float_p, _c_int_p]
    _vforce.vvcopysignf.restype = None


if _available:
    _setup_vdsp_signatures()
    _setup_blas_signatures()
    _setup_vforce_signatures()


# ---------------------------------------------------------------------------
# Helper: numpy array -> ctypes pointer
# ---------------------------------------------------------------------------
def _fp(arr: np.ndarray) -> _c_float_p:
    """Get a float pointer from a contiguous float32 numpy array."""
    return arr.ctypes.data_as(_c_float_p)


def _ensure_f32_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is contiguous float32."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


# ---------------------------------------------------------------------------
# DFT Setup cache: reuse setups for repeated FFT sizes (AMX-backed)
# ---------------------------------------------------------------------------
# ctypes returns c_void_p values as plain Python int
_dft_setups: dict[tuple[int, int], int] = {}


def _get_dft_setup(length: int, forward: bool = True) -> int:
    """Get or create a cached vDSP DFT setup object (as ctypes void pointer).

    The setup is expensive to create but reusable for repeated FFT calls
    of the same size. Apple's DFT internally uses the AMX coprocessor
    for power-of-2 sizes.
    """
    direction = 1 if forward else -1
    key = (length, direction)
    if key not in _dft_setups:
        setup = _accelerate.vDSP_DFT_zop_CreateSetup(
            None,
            _c_ulong(length),
            ctypes.c_int(direction),
        )
        if not setup:
            raise RuntimeError(f"vDSP_DFT_zop_CreateSetup failed for length={length}")
        _dft_setups[key] = setup
    return _dft_setups[key]


# ---------------------------------------------------------------------------
# Public API: FFT operations (AMX-backed via vDSP)
# ---------------------------------------------------------------------------
def rfft(signal: np.ndarray) -> np.ndarray:
    """Compute real FFT using Apple Accelerate vDSP (AMX-backed).

    For large transforms (n >= 8192), uses the AMX coprocessor which
    outperforms numpy PocketFFT. For smaller transforms, delegates to
    numpy which avoids ctypes marshalling overhead.

    Args:
        signal: 1D float32 array of length N (should be power of 2 for vDSP).

    Returns:
        Complex128 array of length N//2 + 1 (matching numpy rfft output format).
    """
    n = len(signal)
    # Use numpy for small transforms (ctypes overhead dominates) or non-power-of-2
    if not _available or n < _MIN_ACCEL_SIZE or n & (n - 1) != 0:
        return np.fft.rfft(signal)

    signal = _ensure_f32_contiguous(signal)

    # vDSP DFT works on complex split format (separate real/imag arrays)
    # For real input, imag part is zero
    in_real = signal.copy()
    in_imag = np.zeros(n, dtype=np.float32)
    out_real = np.empty(n, dtype=np.float32)
    out_imag = np.empty(n, dtype=np.float32)

    setup = _get_dft_setup(n, forward=True)
    _accelerate.vDSP_DFT_Execute(
        setup, _fp(in_real), _fp(in_imag), _fp(out_real), _fp(out_imag)
    )

    # Convert to numpy complex format (only first N//2+1 bins for rfft)
    half = n // 2 + 1
    result = np.empty(half, dtype=np.complex128)
    result.real = out_real[:half]
    result.imag = out_imag[:half]
    return result


def irfft(spectrum: np.ndarray, n: int | None = None) -> np.ndarray:
    """Compute inverse real FFT using Apple Accelerate vDSP (AMX-backed).

    For large transforms (n >= 8192), uses the AMX coprocessor.
    For smaller transforms, delegates to numpy.

    Args:
        spectrum: Complex array of length N//2 + 1.
        n: Output length. If None, inferred as 2*(len(spectrum)-1).

    Returns:
        Float32 array of real samples.
    """
    if n is None:
        n = 2 * (len(spectrum) - 1)

    if not _available or n < _MIN_ACCEL_SIZE or n & (n - 1) != 0:
        return np.fft.irfft(spectrum, n=n).astype(np.float32)

    # Reconstruct full complex spectrum from rfft output (Hermitian symmetry)
    in_real = np.zeros(n, dtype=np.float32)
    in_imag = np.zeros(n, dtype=np.float32)

    half = len(spectrum)
    in_real[:half] = spectrum.real.astype(np.float32)
    in_imag[:half] = spectrum.imag.astype(np.float32)
    # Mirror conjugate for bins half..n-1
    if half > 1:
        in_real[half:] = spectrum[-2:0:-1].real.astype(np.float32)
        in_imag[half:] = -spectrum[-2:0:-1].imag.astype(np.float32)

    out_real = np.empty(n, dtype=np.float32)
    out_imag = np.empty(n, dtype=np.float32)

    setup = _get_dft_setup(n, forward=False)
    _accelerate.vDSP_DFT_Execute(
        setup, _fp(in_real), _fp(in_imag), _fp(out_real), _fp(out_imag)
    )

    # vDSP inverse DFT is unnormalized; divide by N
    result = out_real / n
    return result


def rfft_batch(frames: np.ndarray) -> np.ndarray:
    """Batch real FFT on multiple frames (rows).

    Uses numpy's highly optimized batch FFT (single C call for all frames).
    For large individual transforms (n_fft >= 8192), uses AMX-backed vDSP
    per-frame which amortizes ctypes overhead over the larger compute.

    Args:
        frames: 2D float32 array of shape (num_frames, n_fft).

    Returns:
        Complex128 array of shape (num_frames, n_fft//2 + 1).
    """
    num_frames, n_fft = frames.shape

    # numpy's batch FFT is a single C call -- faster than per-frame ctypes
    # for typical audio FFT sizes (n_fft=2048). Only use vDSP for large transforms.
    if not _available or n_fft < _MIN_ACCEL_SIZE or n_fft & (n_fft - 1) != 0:
        return np.fft.rfft(frames, axis=-1)

    frames = _ensure_f32_contiguous(frames)
    half = n_fft // 2 + 1
    result = np.empty((num_frames, half), dtype=np.complex128)

    setup = _get_dft_setup(n_fft, forward=True)
    in_imag = np.zeros(n_fft, dtype=np.float32)
    out_real = np.empty(n_fft, dtype=np.float32)
    out_imag = np.empty(n_fft, dtype=np.float32)

    for i in range(num_frames):
        in_real = np.ascontiguousarray(frames[i])
        _accelerate.vDSP_DFT_Execute(
            setup, _fp(in_real), _fp(in_imag), _fp(out_real), _fp(out_imag)
        )
        result[i].real = out_real[:half]
        result[i].imag = out_imag[:half]

    return result


def irfft_batch(spectra: np.ndarray, n: int) -> np.ndarray:
    """Batch inverse real FFT on multiple spectra (rows).

    Uses numpy's batch IFFT for typical audio sizes. AMX-backed vDSP for
    large transforms (n >= 8192).

    Args:
        spectra: Complex array of shape (num_frames, n_fft//2+1).
        n: Output length per frame (n_fft).

    Returns:
        Float32 array of shape (num_frames, n).
    """
    if not _available or n < _MIN_ACCEL_SIZE or n & (n - 1) != 0:
        return np.fft.irfft(spectra, n=n, axis=-1).astype(np.float32)

    num_frames = spectra.shape[0]
    half = spectra.shape[1]
    result = np.empty((num_frames, n), dtype=np.float32)
    scale = np.float32(1.0 / n)

    setup = _get_dft_setup(n, forward=False)
    in_real = np.empty(n, dtype=np.float32)
    in_imag = np.empty(n, dtype=np.float32)
    out_real = np.empty(n, dtype=np.float32)
    out_imag = np.empty(n, dtype=np.float32)

    for i in range(num_frames):
        spec = spectra[i]
        in_real[:half] = spec.real.astype(np.float32)
        in_imag[:half] = spec.imag.astype(np.float32)
        if half > 1:
            in_real[half:] = spec[-2:0:-1].real.astype(np.float32)
            in_imag[half:] = -spec[-2:0:-1].imag.astype(np.float32)

        _accelerate.vDSP_DFT_Execute(
            setup, _fp(in_real), _fp(in_imag), _fp(out_real), _fp(out_imag)
        )
        result[i] = out_real * scale

    return result


# ---------------------------------------------------------------------------
# Public API: vDSP vector operations (AMX/NEON optimized)
# ---------------------------------------------------------------------------
def vmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiply. Uses vDSP_vmul for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return (a * b).astype(np.float32) if a.dtype == np.float32 else a * b
    a = _ensure_f32_contiguous(a)
    b = _ensure_f32_contiguous(b)
    out = np.empty_like(a)
    n = _c_ulong(len(a))
    _accelerate.vDSP_vmul(_fp(a), _c_long(1), _fp(b), _c_long(1), _fp(out), _c_long(1), n)
    return out


def vadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add. Uses vDSP_vadd for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return (a + b).astype(np.float32) if a.dtype == np.float32 else a + b
    a = _ensure_f32_contiguous(a)
    b = _ensure_f32_contiguous(b)
    out = np.empty_like(a)
    n = _c_ulong(len(a))
    _accelerate.vDSP_vadd(_fp(a), _c_long(1), _fp(b), _c_long(1), _fp(out), _c_long(1), n)
    return out


def vma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vector multiply-add: D[i] = A[i]*B[i] + C[i]. Uses vDSP_vma for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return a * b + c
    a = _ensure_f32_contiguous(a)
    b = _ensure_f32_contiguous(b)
    c = _ensure_f32_contiguous(c)
    out = np.empty_like(a)
    n = _c_ulong(len(a))
    _accelerate.vDSP_vma(
        _fp(a), _c_long(1), _fp(b), _c_long(1),
        _fp(c), _c_long(1), _fp(out), _c_long(1), n,
    )
    return out


def vsmul(a: np.ndarray, scalar: float) -> np.ndarray:
    """Vector-scalar multiply. Uses vDSP_vsmul for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return (a * scalar).astype(np.float32) if a.dtype == np.float32 else a * scalar
    a = _ensure_f32_contiguous(a)
    out = np.empty_like(a)
    s = ctypes.c_float(scalar)
    n = _c_ulong(len(a))
    _accelerate.vDSP_vsmul(_fp(a), _c_long(1), ctypes.byref(s), _fp(out), _c_long(1), n)
    return out


def vramp(start: float, step: float, n: int) -> np.ndarray:
    """Generate linear ramp. Uses vDSP_vramp for large arrays.

    Returns array where out[i] = start + i * step.
    """
    if not _available or n < _MIN_ACCEL_SIZE:
        return np.arange(n, dtype=np.float32) * np.float32(step) + np.float32(start)
    out = np.empty(n, dtype=np.float32)
    s = ctypes.c_float(start)
    st = ctypes.c_float(step)
    _accelerate.vDSP_vramp(ctypes.byref(s), ctypes.byref(st), _fp(out), _c_long(1), _c_ulong(n))
    return out


def vclip(a: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip vector to [low, high]. Uses vDSP_vclip for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return np.clip(a, low, high)
    a = _ensure_f32_contiguous(a)
    out = np.empty_like(a)
    lo = ctypes.c_float(low)
    hi = ctypes.c_float(high)
    _accelerate.vDSP_vclip(
        _fp(a), _c_long(1), ctypes.byref(lo), ctypes.byref(hi), _fp(out), _c_long(1),
        _c_ulong(len(a)),
    )
    return out


def vabs(a: np.ndarray) -> np.ndarray:
    """Absolute value. Uses vDSP_vabs for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return np.abs(a).astype(np.float32) if a.dtype == np.float32 else np.abs(a)
    a = _ensure_f32_contiguous(a)
    out = np.empty_like(a)
    _accelerate.vDSP_vabs(_fp(a), _c_long(1), _fp(out), _c_long(1), _c_ulong(len(a)))
    return out


def maxv(a: np.ndarray) -> float:
    """Maximum value in vector. Uses vDSP_maxv for large arrays."""
    if not _available or len(a) < _MIN_ACCEL_SIZE:
        return float(np.max(a))
    a = _ensure_f32_contiguous(a)
    result = ctypes.c_float(0.0)
    _accelerate.vDSP_maxv(_fp(a), _c_long(1), ctypes.byref(result), _c_ulong(len(a)))
    return result.value


# ---------------------------------------------------------------------------
# Public API: vForce operations (vectorized transcendentals)
# ---------------------------------------------------------------------------
def vtanhf(a: np.ndarray) -> np.ndarray:
    """Vectorized tanh. Uses vForce vvtanhf for large arrays."""
    if not _vforce or not _available or len(a) < _MIN_ACCEL_SIZE:
        return np.tanh(a).astype(np.float32) if a.dtype == np.float32 else np.tanh(a)
    a = _ensure_f32_contiguous(a)
    out = np.empty_like(a)
    n = ctypes.c_int(len(a))
    _vforce.vvtanhf(_fp(out), _fp(a), ctypes.byref(n))
    return out


def vabsf(a: np.ndarray) -> np.ndarray:
    """Vectorized absolute value. Uses vForce vvfabsf for large arrays."""
    if not _vforce or not _available or len(a) < _MIN_ACCEL_SIZE:
        return np.abs(a).astype(np.float32) if a.dtype == np.float32 else np.abs(a)
    a = _ensure_f32_contiguous(a)
    out = np.empty_like(a)
    n = ctypes.c_int(len(a))
    _vforce.vvfabsf(_fp(out), _fp(a), ctypes.byref(n))
    return out


def vcopysignf(magnitude: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Copy sign from one array to another. Uses vForce vvcopysignf for large arrays.

    Returns array where out[i] = |magnitude[i]| * sign(sign[i]).
    Equivalent to np.copysign(magnitude, sign).
    """
    if not _vforce or not _available or len(magnitude) < _MIN_ACCEL_SIZE:
        return np.copysign(magnitude, sign).astype(np.float32)
    magnitude = _ensure_f32_contiguous(magnitude)
    sign = _ensure_f32_contiguous(sign)
    out = np.empty_like(magnitude)
    n = ctypes.c_int(len(magnitude))
    _vforce.vvcopysignf(_fp(out), _fp(magnitude), _fp(sign), ctypes.byref(n))
    return out


# ---------------------------------------------------------------------------
# Public API: Convolution / Resampling (AMX-backed)
# ---------------------------------------------------------------------------
def conv(signal: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
    """1D convolution using vDSP_conv (AMX-backed).

    Computes the correlation/convolution of signal with filter_coeffs.
    The filter is applied in reverse (as per vDSP_conv convention -- true
    convolution requires the caller to reverse the filter first).

    Args:
        signal: Input signal (1D float32).
        filter_coeffs: Filter kernel (1D float32, should be reversed for
                       true convolution).

    Returns:
        Convolution result array.
    """
    if not _available:
        return np.convolve(signal, filter_coeffs, mode="full").astype(np.float32)

    signal = _ensure_f32_contiguous(signal)
    filter_coeffs = _ensure_f32_contiguous(filter_coeffs)

    sig_len = len(signal)
    filt_len = len(filter_coeffs)
    result_len = sig_len + filt_len - 1

    # vDSP_conv needs the signal padded with filt_len-1 zeros at both ends
    padded = np.zeros(sig_len + 2 * (filt_len - 1), dtype=np.float32)
    padded[filt_len - 1 : filt_len - 1 + sig_len] = signal

    result = np.empty(result_len, dtype=np.float32)
    _accelerate.vDSP_conv(
        _fp(padded), _c_long(1),
        _fp(filter_coeffs), _c_long(1),
        _fp(result), _c_long(1),
        _c_ulong(result_len),
        _c_ulong(filt_len),
    )
    return result


def desamp(signal: np.ndarray, decimation: int, filter_coeffs: np.ndarray) -> np.ndarray:
    """Decimation with FIR anti-aliasing filter using vDSP_desamp.

    Downsamples signal by factor `decimation`, applying FIR filter to
    prevent aliasing. This is the core of polyphase resampling.

    Args:
        signal: Input signal (1D float32).
        decimation: Decimation factor (integer >= 1).
        filter_coeffs: FIR anti-aliasing filter (1D float32).

    Returns:
        Decimated signal.
    """
    if not _available:
        from scipy.signal import resample_poly
        return resample_poly(signal, 1, decimation).astype(np.float32)

    signal = _ensure_f32_contiguous(signal)
    filter_coeffs = _ensure_f32_contiguous(filter_coeffs)
    filt_len = len(filter_coeffs)

    # Output count: number of complete decimated samples
    output_count = max(0, (len(signal) - filt_len) // decimation + 1)
    if output_count == 0:
        return np.array([], dtype=np.float32)

    result = np.empty(output_count, dtype=np.float32)
    _accelerate.vDSP_desamp(
        _fp(signal),
        _c_long(decimation),
        _fp(filter_coeffs),
        _fp(result),
        _c_ulong(output_count),
        _c_ulong(filt_len),
    )
    return result


# ---------------------------------------------------------------------------
# Public API: High-level audio operations
# ---------------------------------------------------------------------------
def hanning_window(n: int) -> np.ndarray:
    """Generate Hann window. Cached for repeated use."""
    return _cached_hanning(n)


@lru_cache(maxsize=4)
def _cached_hanning(n: int) -> np.ndarray:
    """Cached Hann window generation."""
    return np.hanning(n).astype(np.float32)


def crossfade(
    prev: np.ndarray, next_audio: np.ndarray, crossfade_samples: int
) -> np.ndarray:
    """Crossfade two audio segments.

    For large crossfades (>= 8192 samples), uses vDSP_vramp + vDSP_vma for
    AMX-backed ramp generation and fused multiply-add. For typical audio
    crossfades (~240 samples), uses numpy which is faster due to avoiding
    ctypes overhead.

    Args:
        prev: Previous audio segment (1D float32).
        next_audio: Next audio segment (1D float32).
        crossfade_samples: Number of samples in crossfade region.

    Returns:
        Concatenated audio with smooth crossfade transition.
    """
    # For typical audio crossfades (~240 samples at 24kHz), numpy is faster
    if not _available or crossfade_samples < _MIN_ACCEL_SIZE:
        fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        overlap = prev[-crossfade_samples:] * fade_out + next_audio[:crossfade_samples] * fade_in
        return np.concatenate([prev[:-crossfade_samples], overlap, next_audio[crossfade_samples:]])

    prev = _ensure_f32_contiguous(prev)
    next_audio = _ensure_f32_contiguous(next_audio)

    # Generate ramps using vDSP (AMX-backed)
    step = 1.0 / (crossfade_samples - 1) if crossfade_samples > 1 else 1.0
    fade_in = vramp(0.0, step, crossfade_samples)
    fade_out = vramp(1.0, -step, crossfade_samples)

    # Get overlap regions (ensure contiguous copies)
    prev_tail = np.ascontiguousarray(prev[-crossfade_samples:])
    next_head = np.ascontiguousarray(next_audio[:crossfade_samples])

    # Fused multiply-add: overlap = prev_tail * fade_out + next_head * fade_in
    weighted_prev = vmul(prev_tail, fade_out)
    overlap = vma(next_head, fade_in, weighted_prev)

    return np.concatenate([prev[:-crossfade_samples], overlap, next_audio[crossfade_samples:]])


# ---------------------------------------------------------------------------
# Public API: BLAS operations (AMX-backed GEMV/GEMM)
# ---------------------------------------------------------------------------
# CBLAS enum constants
_CBLAS_ROW_MAJOR = 101
_CBLAS_NO_TRANS = 111
_CBLAS_TRANS = 112


def gemv(
    A: np.ndarray, x: np.ndarray, y: np.ndarray | None = None,
    alpha: float = 1.0, beta: float = 0.0, trans: bool = False,
) -> np.ndarray:
    """Matrix-vector multiply using cblas_sgemv (AMX-backed).

    Computes y = alpha * A @ x + beta * y  (or A^T @ x if trans=True).
    This uses the AMX coprocessor for the matrix-vector product, which is
    faster than GPU for small matrices (dims <= ~1536).

    Args:
        A: 2D float32 array of shape (M, N).
        x: 1D float32 array of length N (or M if trans).
        y: Optional output array. If None, allocated and beta=0.
        alpha: Scalar multiplier for A @ x.
        beta: Scalar multiplier for y (0 means overwrite).
        trans: If True, compute A^T @ x instead.

    Returns:
        Result vector y of length M (or N if trans).
    """
    if not _available:
        if trans:
            result = A.T @ x
        else:
            result = A @ x
        if y is not None:
            return (alpha * result + beta * y).astype(np.float32)
        return (alpha * result).astype(np.float32)

    A = _ensure_f32_contiguous(A)
    x = _ensure_f32_contiguous(x)
    M, N = A.shape
    out_len = N if trans else M

    if y is None:
        y = np.empty(out_len, dtype=np.float32)
        beta = 0.0
    else:
        y = _ensure_f32_contiguous(y)

    _accelerate.cblas_sgemv(
        _CBLAS_ROW_MAJOR,
        _CBLAS_TRANS if trans else _CBLAS_NO_TRANS,
        M, N,
        ctypes.c_float(alpha),
        _fp(A), N,
        _fp(x), 1,
        ctypes.c_float(beta),
        _fp(y), 1,
    )
    return y


def gemm(
    A: np.ndarray, B: np.ndarray, C: np.ndarray | None = None,
    alpha: float = 1.0, beta: float = 0.0,
    transA: bool = False, transB: bool = False,
) -> np.ndarray:
    """Matrix-matrix multiply using cblas_sgemm (AMX-backed).

    Computes C = alpha * op(A) @ op(B) + beta * C.
    Uses the AMX coprocessor for the computation.

    Args:
        A: 2D float32 array.
        B: 2D float32 array.
        C: Optional output array.
        alpha: Scalar for A @ B.
        beta: Scalar for C.
        transA: Transpose A.
        transB: Transpose B.

    Returns:
        Result matrix C.
    """
    if not _available:
        opA = A.T if transA else A
        opB = B.T if transB else B
        result = opA @ opB
        if C is not None:
            return (alpha * result + beta * C).astype(np.float32)
        return (alpha * result).astype(np.float32)

    A = _ensure_f32_contiguous(A)
    B = _ensure_f32_contiguous(B)

    M = A.shape[1] if transA else A.shape[0]
    K = A.shape[0] if transA else A.shape[1]
    N = B.shape[0] if transB else B.shape[1]
    lda = A.shape[1]
    ldb = B.shape[1]

    if C is None:
        C = np.empty((M, N), dtype=np.float32)
        beta = 0.0
    else:
        C = _ensure_f32_contiguous(C)

    _accelerate.cblas_sgemm(
        _CBLAS_ROW_MAJOR,
        _CBLAS_TRANS if transA else _CBLAS_NO_TRANS,
        _CBLAS_TRANS if transB else _CBLAS_NO_TRANS,
        M, N, K,
        ctypes.c_float(alpha),
        _fp(A), lda,
        _fp(B), ldb,
        ctypes.c_float(beta),
        _fp(C), N,
    )
    return C
