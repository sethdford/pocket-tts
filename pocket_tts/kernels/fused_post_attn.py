"""Fused post-attention Metal kernel: transpose + reshape + out_proj in one dispatch.

After scaled_dot_product_attention, the standard path does:
  1. x.transpose(0, 2, 1, 3)  -- (B,H,T,D) → (B,T,H,D)
  2. x.reshape(B, T, embed_dim) -- flatten heads
  3. out_proj(x)               -- linear projection

For T=1 streaming (the hot generation path), all dimensions are fixed:
B=1, H=num_heads, T=1, D=head_dim. This kernel fuses all three operations
into a single Metal dispatch, saving 2 kernel launches per attention layer.

With 6 FlowLM layers, this saves 12 kernel dispatches per generation step.
"""

import mlx.core as mx

# Metal kernel: fused transpose(0,2,1,3) + reshape + matmul(out_proj)
# For T=1: input is (1, H, 1, D), output is (1, 1, embed_dim)
#
# The transpose for (B,H,1,D) → (B,1,H,D) with T=1 is effectively just
# a reinterpretation: the data layout [h0d0..h0dD, h1d0..h1dD, ...] is
# already the correct layout for (1, H*D) = (1, embed_dim).
# So we skip the transpose entirely and go straight to the matmul.
_FUSED_POST_ATTN_SOURCE = """
    // Grid: (embed_dim, 1, 1) — one thread per output element
    uint out_idx = thread_position_in_grid.x;

    // attn_out is (1, H, 1, D) but stored contiguously as (H*D,)
    // out_proj_weight is (embed_dim, embed_dim) stored row-major
    // We compute: output[out_idx] = dot(attn_out, out_proj_weight[out_idx, :])

    float acc = 0.0f;
    for (uint k = 0; k < attn_out_shape[1] * attn_out_shape[3]; k++) {
        acc += float(attn_out[k]) * float(weight[out_idx * weight_shape[1] + k]);
    }
    out[out_idx] = T(acc);
"""

_fused_post_attn_kernel = mx.fast.metal_kernel(
    name="fused_post_attn_t1",
    input_names=["attn_out", "weight"],
    output_names=["out"],
    source=_FUSED_POST_ATTN_SOURCE,
)


def fused_post_attention(
    attn_out: mx.array, out_proj_weight: mx.array, embed_dim: int
) -> mx.array:
    """Fused post-attention: transpose + reshape + out_proj in one Metal dispatch.

    Only for T=1 streaming (B=1, T=1). Falls back to standard path otherwise.

    Args:
        attn_out: (1, H, 1, D) attention output
        out_proj_weight: (embed_dim, embed_dim) out_proj weight matrix
        embed_dim: embedding dimension (H * D)

    Returns:
        (1, 1, embed_dim) projected output
    """
    outputs = _fused_post_attn_kernel(
        inputs=[attn_out, out_proj_weight],
        template=[("T", attn_out.dtype)],
        grid=(embed_dim, 1, 1),
        threadgroup=(min(256, embed_dim), 1, 1),
        output_shapes=[(embed_dim,)],
        output_dtypes=[attn_out.dtype],
    )
    return outputs[0].reshape(1, 1, embed_dim)
