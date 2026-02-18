"""Custom Metal kernels for pocket-tts inference optimization.

All kernels use mx.fast.metal_kernel() for JIT-compiled Metal Shading Language
code that runs on Apple Silicon GPUs.
"""
