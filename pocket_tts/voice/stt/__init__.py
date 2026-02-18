"""Speech-to-text backends for the voice pipeline.

Available backends:
  - DsmSTT:    MLX-native via moshi_mlx with semantic VAD (recommended)
  - RustSTT:   Zero-Python Rust/candle/Metal backend (requires Rust toolchain)
  - KyutaiSTT: Python/PyTorch backend (fallback)
"""
