import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array):
        return self.scale * x
