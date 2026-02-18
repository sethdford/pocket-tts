import mlx.core as mx
import mlx.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding using mx.fast.rope fused Metal kernel.

    For T=1 generation, uses a batched strategy to apply RoPE to both Q and K
    in a single kernel dispatch (halving the number of RoPE kernel launches).

    Args:
        max_period (float): Maximum period of the rotation frequencies (base).
    """

    def __init__(self, max_period: float | int = 10000.0):
        super().__init__()
        self.max_period = float(max_period)

    def __call__(self, q: mx.array, k: mx.array, offset: mx.array | int):
        """Apply RoPE to query and key tensors.

        Args:
            q: shape (B, H, T, D) - already transposed for attention
            k: shape (B, H, T, D)
            offset: position offset for streaming
        """
        D = q.shape[-1]
        B = q.shape[0]
        T = q.shape[2]

        if T == 1 and B == 1:
            # Fused dual RoPE: stack Q and K along batch dim, apply once, split
            # Reduces 2 kernel dispatches to 1 for the common generation case
            qk = mx.concatenate([q, k], axis=0)  # (2, H, 1, D)
            # Broadcast offset for both Q and K (same position)
            if isinstance(offset, mx.array) and offset.ndim == 0:
                dual_offset = mx.broadcast_to(offset, (2,))
            elif isinstance(offset, mx.array):
                dual_offset = mx.concatenate([offset, offset], axis=0)
            else:
                dual_offset = offset
            qk = mx.fast.rope(
                qk, D, traditional=True, base=self.max_period, scale=1.0, offset=dual_offset
            )
            q = qk[:1]
            k = qk[1:]
        else:
            q = mx.fast.rope(
                q, D, traditional=True, base=self.max_period, scale=1.0, offset=offset
            )
            k = mx.fast.rope(
                k, D, traditional=True, base=self.max_period, scale=1.0, offset=offset
            )
        return q, k
