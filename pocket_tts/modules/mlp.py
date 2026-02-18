"""
Taken from
https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py

"""

import math

import mlx.core as mx
import mlx.nn as nn
from typing_extensions import Self

from pocket_tts.utils.config import FlowLMConfig


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """RMSNorm using mx.fast.rms_norm fused Metal kernel.
    Uses 'alpha' as weight name for PyTorch weight compatibility.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.ones((dim,))

    def __call__(self, x: mx.array):
        return mx.fast.rms_norm(x, self.alpha, self.eps)


class LayerNorm(nn.Module):
    """LayerNorm using mx.fast.layer_norm fused Metal kernel."""

    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))

    def __call__(self, x):
        if self.elementwise_affine:
            return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
        return mx.fast.layer_norm(x, None, None, self.eps)


def _run_sequential(layers: list, x: mx.array) -> mx.array:
    """Run a list of layers sequentially, matching PyTorch nn.Sequential behavior."""
    for layer in layers:
        x = layer(x)
    return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000
    ):
        super().__init__()
        # Store as numbered dict keys to match PyTorch nn.Sequential naming (0, 1, 2, ...)
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            RMSNorm(hidden_size),
        ]
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.freqs = mx.exp(-math.log(max_period) * mx.arange(0, half).astype(mx.float32) / half)

    def __call__(self, t):
        args = t * self.freqs
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        t_emb = _run_sequential(self.mlp, embedding)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = [
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        ]

        self.adaLN_modulation = [
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        ]

    def __call__(self, x, y):
        mod_out = _run_sequential(self.adaLN_modulation, y)
        shift_mlp, scale_mlp, gate_mlp = mx.split(mod_out, 3, axis=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = _run_sequential(self.mlp, h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = [
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        ]

    def __call__(self, x, c):
        mod_out = _run_sequential(self.adaLN_modulation, c)
        shift, scale = mx.split(mod_out, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """Taken from https://arxiv.org/abs/2406.11838.

    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param cond_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        cond_channels,
        num_res_blocks,
        num_time_conds=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        assert num_time_conds != 1
        self.time_embed = [TimestepEmbedder(model_channels) for _ in range(num_time_conds)]
        self.cond_embed = nn.Linear(cond_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = [ResBlock(model_channels) for _ in range(num_res_blocks)]
        self.final_layer = FinalLayer(model_channels, out_channels)
        # Default to uncompiled; call _init_compiled() after freeze for fused kernels
        self._compiled_forward = self._forward

    @classmethod
    def from_pydantic_config(cls, cfg: FlowLMConfig, latent_dim: int, cond_dim: int) -> Self:
        config = cfg.flow

        flow_dim = config.dim
        flow_depth = config.depth
        num_time_conds = 2
        return SimpleMLPAdaLN(
            latent_dim, flow_dim, latent_dim, cond_dim, flow_depth, num_time_conds=num_time_conds
        )

    def __call__(
        self, c: mx.array, s: mx.array, t: mx.array, x: mx.array
    ) -> mx.array:
        """
        Apply the model to an input batch.
        :param c: conditioning from AR transformer.
        :param s: start time tensor.
        :param t: target time tensor.
        :param x: an [N x C] Tensor of inputs.
        :return: an [N x C] Tensor of outputs.
        """
        return self._compiled_forward(c, s, t, x)

    def _forward(
        self, c: mx.array, s: mx.array, t: mx.array, x: mx.array
    ) -> mx.array:
        x = self.input_proj(x)
        # Fuse both timestep embeddings + conditioning in minimal ops
        t_combined = (self.time_embed[0](s) + self.time_embed[1](t)) * 0.5
        y = t_combined + self.cond_embed(c)

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

    def _init_compiled(self):
        """Initialize compiled forward pass (call after model is loaded and frozen)."""
        self._compiled_forward = mx.compile(self._forward)
