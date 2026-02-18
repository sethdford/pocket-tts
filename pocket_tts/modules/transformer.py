import mlx.core as mx
import mlx.nn as nn

from pocket_tts.kernels.fused_post_attn import fused_post_attention
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule

# Custom Metal kernel: fused dual KV cache append.
# Combines two mx.concatenate operations (k_cache + k, v_cache + v) into a
# single GPU kernel dispatch. For each attention layer, this saves one kernel
# launch (~10-20μs). With 6 FlowLM layers at 12.5 Hz, that's ~600-1200μs/s
# of pure overhead eliminated.
_FUSED_KV_APPEND_SOURCE = """
    // Thread grid: (total_elements, 1, 1)
    // total_elements = B * H * (T_old + 1) * D * 2  (for both k and v)
    uint elem = thread_position_in_grid.x;

    // Layout: [k_out (B,H,T_new,D) | v_out (B,H,T_new,D)]
    // T_new = T_old + 1
    uint half_size = k_cache_shape[0] * k_cache_shape[1] * (k_cache_shape[2] + 1) * k_cache_shape[3];
    bool is_v = elem >= half_size;
    uint local_elem = is_v ? (elem - half_size) : elem;

    uint D = k_cache_shape[3];
    uint T_new = k_cache_shape[2] + 1;
    uint H = k_cache_shape[1];

    uint d_idx = local_elem % D;
    uint t_idx = (local_elem / D) % T_new;
    uint h_idx = (local_elem / D / T_new) % H;
    uint b_idx = local_elem / D / T_new / H;

    uint T_old = k_cache_shape[2];
    uint old_stride = D;
    uint old_h_stride = T_old * old_stride;
    uint old_b_stride = H * old_h_stride;

    if (t_idx < T_old) {
        // Copy from existing cache
        uint src_idx = b_idx * old_b_stride + h_idx * old_h_stride + t_idx * old_stride + d_idx;
        if (is_v) {
            k_v_out[elem] = v_cache[src_idx];
        } else {
            k_v_out[elem] = k_cache[src_idx];
        }
    } else {
        // Append new entry (t_idx == T_old)
        uint new_stride = D;
        uint new_h_stride = new_stride;  // T=1, so just D
        uint new_b_stride = H * new_h_stride;
        uint src_idx = b_idx * new_b_stride + h_idx * new_h_stride + d_idx;
        if (is_v) {
            k_v_out[elem] = new_v[src_idx];
        } else {
            k_v_out[elem] = new_k[src_idx];
        }
    }
"""

_fused_kv_append_kernel = mx.fast.metal_kernel(
    name="fused_kv_cache_append",
    input_names=["k_cache", "v_cache", "new_k", "new_v"],
    output_names=["k_v_out"],
    source=_FUSED_KV_APPEND_SOURCE,
)


def _fused_kv_cache_append(
    k_cache: mx.array, v_cache: mx.array, k: mx.array, v: mx.array
) -> tuple[mx.array, mx.array]:
    """Append new k/v entries to caches in a single Metal kernel dispatch.

    Instead of two separate mx.concatenate operations, this fuses both into
    one GPU kernel, saving one kernel launch overhead per attention layer.

    Args:
        k_cache: (B, H, T_old, D) existing key cache
        v_cache: (B, H, T_old, D) existing value cache
        k: (B, H, 1, D) new key to append
        v: (B, H, 1, D) new value to append

    Returns:
        Tuple of (k_new, v_new) each with shape (B, H, T_old+1, D)
    """
    B, H, T_old, D = k_cache.shape
    T_new = T_old + 1
    out_shape = (B, H, T_new, D)
    total_elements = B * H * T_new * D * 2  # *2 for both k and v

    outputs = _fused_kv_append_kernel(
        inputs=[k_cache, v_cache, k, v],
        template=[("T", k_cache.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(min(256, total_elements), 1, 1),
        output_shapes=[(B * H * T_new * D * 2,)],
        output_dtypes=[k_cache.dtype],
    )
    # Split the interleaved output into k and v
    flat = outputs[0]
    half = B * H * T_new * D
    k_new = flat[:half].reshape(out_shape)
    v_new = flat[half:].reshape(out_shape)
    return k_new, v_new


def _make_compiled_pre_attn(num_heads: int, head_dim: int, max_period: float | int):
    """Create a compiled function for pre-attention processing.

    Fuses: in_proj matmul → split Q/K/V → transpose → dual RoPE
    into a single optimized computation graph. Since input shapes are
    fixed during generation (B=1, T=1), this compiles once and reuses.
    """
    # mx.fast.rope requires base to be float
    base = float(max_period)

    def _pre_attn(query, in_proj_weight, offset):
        projected = query @ in_proj_weight.T
        packed = projected.reshape(1, 1, 3, num_heads, head_dim)
        q = packed[:, :, 0, :, :].transpose(0, 2, 1, 3)
        k = packed[:, :, 1, :, :].transpose(0, 2, 1, 3)
        v = packed[:, :, 2, :, :].transpose(0, 2, 1, 3)
        q = mx.fast.rope(
            q, head_dim, traditional=True, base=base, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, head_dim, traditional=True, base=base, scale=1.0, offset=offset
        )
        return q, k, v

    return mx.compile(_pre_attn)


class StreamingMultiheadAttention(StatefulModule):
    """Streaming multihead attention with growing KV cache and fused RoPE/SDPA.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        rope (`RotaryEmbedding`): Rotary embedding module.
    """

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads
        self._head_dim = embed_dim // num_heads

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Compiled pre-attention: fuses in_proj → split → transpose → RoPE
        self._compiled_pre_attn = _make_compiled_pre_attn(
            num_heads, self._head_dim, rope.max_period
        )

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, mx.array]:
        dim_per_head = self.embed_dim // self.num_heads
        return dict(
            offset=mx.array(0, dtype=mx.int32),
            k_cache=mx.zeros((batch_size, self.num_heads, 0, dim_per_head)),
            v_cache=mx.zeros((batch_size, self.num_heads, 0, dim_per_head)),
        )

    def increment_step(self, state: dict, increment: int = 1):
        state["offset"] = state["offset"] + increment

    def __call__(self, query: mx.array, model_state: dict | None):
        if model_state is None:
            raise ValueError("model_state must be provided")
        state = self.get_state(model_state)
        b, t, _ = query.shape
        d = self._head_dim
        offset = state["offset"]

        if t == 1:
            # Compiled fast path: fused in_proj → split → transpose → RoPE
            q, k, v = self._compiled_pre_attn(query, self.in_proj.weight, offset)
        else:
            # Prefill path: full processing
            projected = self.in_proj(query)
            packed = projected.reshape(b, t, 3, self.num_heads, d)
            q = packed[:, :, 0, :, :]
            k = packed[:, :, 1, :, :]
            v = packed[:, :, 2, :, :]
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            q, k = self.rope(q, k, offset=offset)

        # Update KV cache (stored in B, H, T, D format)
        if t == 1:
            # Fused dual KV cache append: single Metal kernel for both k and v
            k_cache, v_cache = _fused_kv_cache_append(state["k_cache"], state["v_cache"], k, v)
        else:
            k_cache = mx.concatenate([state["k_cache"], k], axis=2)
            v_cache = mx.concatenate([state["v_cache"], v], axis=2)
        state["k_cache"] = k_cache
        state["v_cache"] = v_cache

        scale = d ** -0.5

        if t == 1:
            # Streaming: query attends to all keys — causal mask is trivially all-True
            # Skip mask construction entirely for a significant speedup
            x = mx.fast.scaled_dot_product_attention(q, k_cache, v_cache, scale=scale)
            # Fused post-attention: transpose + reshape + out_proj in 1 Metal dispatch
            # Saves 2 kernel launches per attention layer per step
            x = fused_post_attention(x, self.out_proj.weight, self.embed_dim)
        else:
            # Prefill: construct full causal mask
            total_len = k_cache.shape[2]
            row_idx = mx.arange(t).reshape(-1, 1) + (total_len - t)
            col_idx = mx.arange(total_len).reshape(1, -1)
            causal_mask = col_idx <= row_idx
            x = mx.fast.scaled_dot_product_attention(
                q, k_cache, v_cache, scale=scale, mask=causal_mask
            )
            # Standard post-attention path for prefill
            x = x.transpose(0, 2, 1, 3)
            x = x.reshape(b, t, self.embed_dim)
            x = self.out_proj(x)

        return x
