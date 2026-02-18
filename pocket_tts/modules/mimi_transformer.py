import mlx.core as mx
import mlx.nn as nn
from typing_extensions import Self

from pocket_tts.modules.layer_scale import LayerScale
from pocket_tts.modules.mlp import LayerNorm
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.utils.config import FlowLMTransformerConfig


class Identity(nn.Module):
    def __call__(self, x):
        return x


class MimiStreamingMultiheadAttention(StatefulModule):
    """Windowed multi-head attention for Mimi codec with streaming KV cache.
    KV cache stored in (B, H, T, D) format for efficient attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        out_dim = 3 * embed_dim

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, mx.array]:
        dim_per_head = self.embed_dim // self.num_heads
        return dict(
            offset=mx.array(0, dtype=mx.int32),
            k_cache=mx.zeros((batch_size, self.num_heads, 0, dim_per_head)),
            v_cache=mx.zeros((batch_size, self.num_heads, 0, dim_per_head)),
        )

    def increment_step(self, state, increment: int = 1):
        state["offset"] = state["offset"] + increment

    def __call__(self, query: mx.array, model_state: dict | None) -> mx.array:
        B, T = query.shape[:2]
        d = self.embed_dim // self.num_heads

        state = self.get_state(model_state) if model_state is not None else None
        offset = state["offset"] if state is not None else mx.zeros((B,), dtype=mx.int32)
        if offset.ndim == 0:
            offset = mx.broadcast_to(offset, (B,))

        projected = self.in_proj(query)

        # Reshape: (B, T, 3*H*D) -> (B, T, 3, H, D)
        packed = projected.reshape(B, T, 3, self.num_heads, d)
        q = packed[:, :, 0, :, :]  # (B, T, H, D)
        k = packed[:, :, 1, :, :]
        v = packed[:, :, 2, :, :]

        # Transpose to (B, H, T, D) for fused RoPE and SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply fused RoPE on (B, H, T, D)
        q, k = self.rope(q, k, offset)

        if state is not None:
            k_cache = state["k_cache"]
            v_cache = state["v_cache"]
            cache_len = k_cache.shape[2]

            # Ring-buffer: when at context capacity, drop oldest instead of concat+trim
            if cache_len >= self.context:
                # Drop oldest entry, append new one (avoids growing past context)
                k_full = mx.concatenate([k_cache[:, :, 1:, :], k], axis=2)
                v_full = mx.concatenate([v_cache[:, :, 1:, :], v], axis=2)
            else:
                k_full = mx.concatenate([k_cache, k], axis=2)
                v_full = mx.concatenate([v_cache, v], axis=2)

            total_kv = k_full.shape[2]

            # Optimized mask for streaming (T=1 common case)
            if T == 1:
                # All positions within context window are valid
                pos_k_start = offset - cache_len
                pos_k = pos_k_start.reshape(-1, 1) + mx.arange(total_kv).reshape(1, -1)
                attn_mask = (pos_k >= 0) & (pos_k <= offset.reshape(-1, 1))
                attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, Kv)
            else:
                pos_q = offset.reshape(-1, 1) + mx.arange(T).reshape(1, -1)
                start_pos = offset - cache_len
                pos_k = start_pos.reshape(-1, 1) + mx.arange(total_kv).reshape(1, -1)
                delta = pos_q[:, :, None] - pos_k[:, None, :]
                attn_mask = (delta >= 0) & (delta < self.context) & (pos_k[:, None, :] >= 0)
                attn_mask = attn_mask[:, None, :, :]

            scale = d ** -0.5
            x = mx.fast.scaled_dot_product_attention(
                q, k_full, v_full, scale=scale, mask=attn_mask
            )

            state["k_cache"] = k_full
            state["v_cache"] = v_full
        else:
            scale = d ** -0.5
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # (B, H, T, D) -> (B, T, H*D)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        context: int | None = None,
        rope: RotaryEmbedding | None = None,
        layer_scale: float | None = None,
    ):
        super().__init__()

        if context is not None:
            self.self_attn = MimiStreamingMultiheadAttention(
                d_model, num_heads, context=context, rope=rope
            )
        else:
            self.self_attn = StreamingMultiheadAttention(d_model, num_heads, rope=rope)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self._has_layer_scale = layer_scale is not None
        if self._has_layer_scale:
            self.layer_scale_1 = LayerScale(d_model, init=layer_scale)
            self.layer_scale_2 = LayerScale(d_model, init=layer_scale)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

    def _ff_block(self, x: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear1(x)))

    def __call__(self, src: mx.array, model_state: dict | None) -> mx.array:
        attn_out = self.self_attn(self.norm1(src), model_state)
        if self._has_layer_scale:
            attn_out = self.layer_scale_1(attn_out)
        x = src + attn_out
        ff_out = self._ff_block(self.norm2(x))
        if self._has_layer_scale:
            ff_out = self.layer_scale_2(ff_out)
        x = x + ff_out
        return x


class StreamingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 5,
        dim_feedforward: int = 2048,
        context: int | None = None,
        max_period: float | int = 10000.0,
        layer_scale: float | None = None,
    ):
        super().__init__()
        rope = RotaryEmbedding(max_period=max_period)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=rope,
                    layer_scale=layer_scale,
                )
            )

    @classmethod
    def from_pydantic_config(cls, cfg: FlowLMTransformerConfig) -> Self:
        return cls(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.d_model * cfg.hidden_scale,
            max_period=cfg.max_period,
        )

    def __call__(self, x: mx.array, model_state: dict | None) -> mx.array:
        for layer in self.layers:
            x = layer(x, model_state)
        return x

    def forward_layers(
        self, x: mx.array, model_state: dict | None, end_layer: int | None = None
    ) -> mx.array:
        """Forward through a subset of layers (for speculative decoding draft model).

        Args:
            x: Input tensor.
            model_state: Model state dict.
            end_layer: Stop after this many layers (exclusive). None = all layers.
        """
        layers = self.layers[:end_layer] if end_layer is not None else self.layers
        for layer in layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: list[int] | tuple[int, ...],
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 5,
        dim_feedforward: int = 2048,
        context: int | None = None,
        max_period: float = 10000.0,
        layer_scale: float | None = None,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dimension, d_model, bias=False) if (
            input_dimension != d_model
        ) else Identity()

        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            context=context,
            max_period=max_period,
            layer_scale=layer_scale,
        )

        self.output_projs = []
        for output_dimension in output_dimensions:
            if output_dimension == d_model:
                self.output_projs.append(Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    @classmethod
    def from_pydantic_config(cls, cfg, input_dimension: int, output_dimensions: list[int]) -> Self:
        return cls(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            context=cfg.context,
            max_period=cfg.max_period,
            layer_scale=cfg.layer_scale,
            input_dimension=input_dimension,
            output_dimensions=output_dimensions,
        )

    def __call__(self, x: mx.array, model_state: dict | None) -> tuple[mx.array, ...]:
        # x is NLC: (B, T, C) — no transpose needed
        x = self.input_proj(x)
        x = self.transformer(x, model_state)
        outs = tuple(proj(x) for proj in self.output_projs)
        return outs
