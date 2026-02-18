import mlx.core as mx
import mlx.nn as nn


class DummyQuantizer(nn.Module):
    """Simplified quantizer that only provides output projection for TTS.

    This removes all unnecessary quantization logic since we don't use actual quantization.
    Uses NLC (batch, length, channels) format for MLX.
    """

    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_proj(x)
