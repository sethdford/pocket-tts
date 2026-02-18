import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing_extensions import Self

from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import LayerNorm, SimpleMLPAdaLN
from pocket_tts.utils.config import FlowLMConfig

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def lsd_decode(
    flow_net, conditioning: mx.array, x_0: mx.array, num_steps: int = 1
) -> mx.array:
    """Rebuilds the data sample from starting point x_0.

    Lagrangian Self Distillation (https://arxiv.org/pdf/2505.18825)

    Args:
        flow_net: Flow network callable (conditioning, s, t, x) -> flow direction.
        conditioning: Conditioning from the transformer backbone.
        x_0: Starting point from the known distribution.
        num_steps: Number of steps to take.

    Returns:
        x_1_hat: (B, D) Reconstructed data sample.
    """
    current = x_0
    # Pre-create scalar broadcast template once (avoids ones_like per step)
    scalar_shape = x_0[..., :1]
    inv_steps = 1.0 / num_steps
    for i in range(num_steps):
        s_val = i * inv_steps
        t_val = (i + 1) * inv_steps
        flow_dir = flow_net(
            conditioning, s_val * scalar_shape, t_val * scalar_shape, current
        )
        current = current + flow_dir * inv_steps
    return current


class FlowLMModel(nn.Module):
    """Transformer-based flow language model on multiple streams of latents.

    Args:
        conditioner (LUTConditioner): Text conditioner for processing text inputs.
        flow: Flow module that defines the flow loss and sampling strategy.
        flow_net: Trainable function (cond, t, x_t) -> u_t.
        dim (int): Dimension of the transformer encoder.
        ldim (int): Latent dimension.
        stats_ema_decay (float): Decay for the EMA of the latent statistics.
    """

    def __init__(
        self,
        conditioner: LUTConditioner,
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: int = 128,
        ldim: int = 64,
        stats_ema_decay: float = 0.999,
        text_padding_weight: float = 1.0,
        dtype=None,
    ):
        super().__init__()
        self.conditioner = conditioner
        self.ldim = ldim
        self.stats_ema_decay = stats_ema_decay
        self.dim = dim
        self.text_padding_weight = text_padding_weight
        self.dtype = dtype or mx.float32

        self.flow_net = flow_net
        self.emb_std = mx.ones((ldim,), dtype=self.dtype)
        self.emb_mean = mx.zeros((ldim,), dtype=self.dtype)
        self.bos_emb = mx.random.normal((ldim,)).astype(self.dtype)

        self.input_linear = nn.Linear(self.ldim, dim, bias=False)
        self.transformer = transformer
        self.out_norm = LayerNorm(dim, eps=1e-5)
        self.out_eos = nn.Linear(dim, 1)

        # AMX hybrid flow network (initialized by init_amx_flow_net())
        self._amx_flow_net = None

    def init_amx_flow_net(self):
        """Initialize the AMX-backed CPU flow network from current weights.

        Call after the model is frozen and weights are loaded. This copies
        flow_net weights to CPU numpy arrays for AMX execution, using ~6MB
        additional memory. If Accelerate BLAS is unavailable, falls back to
        numpy BLAS (still avoids GPU dispatch overhead).
        """
        from pocket_tts.native.amx_flow_net import AMXFlowNet

        self._amx_flow_net = AMXFlowNet.from_mlx_model(self.flow_net)
        logger.info("AMX hybrid flow network enabled")

    def __call__(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        model_state: dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, S, ldim], returns the
        reconstructed latent in generation mode.
        """
        # NaN values signal a BOS position (only on first step)
        is_nan = mx.isnan(sequence)
        sequence = mx.where(is_nan, self.bos_emb, sequence)
        input_ = self.input_linear(sequence)

        transformer_out = self.backbone(input_, text_embeddings, sequence, model_state=model_state)
        # Skip cast if already float32
        if transformer_out.dtype != mx.float32:
            transformer_out = transformer_out.astype(mx.float32)

        transformer_out = transformer_out[:, -1]
        out_eos = self.out_eos(transformer_out) > eos_threshold

        noise_shape = transformer_out.shape[:-1] + (self.ldim,)
        # Cache std computation — temp is constant across steps
        std = temp**0.5
        noise = mx.random.normal(noise_shape, dtype=mx.float32) * std
        if noise_clamp is not None:
            noise = mx.clip(noise, -noise_clamp, noise_clamp)

        # Hybrid AMX path: run LSD decode on CPU/AMX instead of GPU
        if self._amx_flow_net is not None:
            return self._amx_lsd_decode(
                transformer_out, noise, lsd_decode_steps
            ), out_eos

        # Pass conditioning directly to avoid partial() allocation each step
        return lsd_decode(self.flow_net, transformer_out, noise, lsd_decode_steps), out_eos

    def _amx_lsd_decode(
        self, conditioning: mx.array, noise: mx.array, num_steps: int
    ) -> mx.array:
        """Run LSD decode on CPU/AMX, bridging MLX <-> numpy.

        Syncs transformer_out and noise to CPU, runs the full Euler
        integration loop on AMX, then converts the result back to MLX.
        The sync point is essentially free since we already need the
        result on CPU for the Mimi decoder input anyway.
        """
        from pocket_tts.native.amx_flow_net import amx_lsd_decode

        # Sync MLX arrays to CPU numpy (forces GPU eval)
        cond_np = np.array(conditioning, copy=False)
        noise_np = np.array(noise, copy=False)

        # Flatten batch dim for the flow network (B=1 always)
        cond_flat = cond_np.reshape(-1)
        noise_flat = noise_np.reshape(-1)

        # Run full LSD decode on CPU/AMX
        result_np = amx_lsd_decode(
            self._amx_flow_net, cond_flat, noise_flat, num_steps
        )

        # Convert back to MLX (reshaping to original layout)
        return mx.array(result_np.reshape(conditioning.shape[:-1] + (self.ldim,)))

    def backbone(
        self, input_, text_embeddings: mx.array, sequence, model_state: dict
    ) -> mx.array:
        # Skip concatenation when text_embeddings is empty (generation steps)
        if text_embeddings.shape[1] > 0:
            input_ = mx.concatenate([text_embeddings, input_], axis=1)
        transformer_out = self.transformer(input_, model_state)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        if text_embeddings.shape[1] > 0:
            transformer_out = transformer_out[:, -sequence.shape[1]:]
        return transformer_out

    def forward_draft(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        model_state: dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
        num_draft_layers: int = 2,
    ) -> tuple[mx.array, mx.array]:
        """Draft forward pass using only the first N transformer layers.

        Used for speculative frame generation. Produces approximate latents
        much faster than the full model.
        """
        is_nan = mx.isnan(sequence)
        sequence = mx.where(is_nan, self.bos_emb, sequence)
        input_ = self.input_linear(sequence)

        if text_embeddings.shape[1] > 0:
            input_ = mx.concatenate([text_embeddings, input_], axis=1)
        transformer_out = self.transformer.forward_layers(
            input_, model_state, end_layer=num_draft_layers
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        if text_embeddings.shape[1] > 0:
            transformer_out = transformer_out[:, -sequence.shape[1]:]

        if transformer_out.dtype != mx.float32:
            transformer_out = transformer_out.astype(mx.float32)
        transformer_out = transformer_out[:, -1]
        out_eos = self.out_eos(transformer_out) > eos_threshold

        noise_shape = transformer_out.shape[:-1] + (self.ldim,)
        std = temp**0.5
        noise = mx.random.normal(noise_shape, dtype=mx.float32) * std
        if noise_clamp is not None:
            noise = mx.clip(noise, -noise_clamp, noise_clamp)

        if self._amx_flow_net is not None:
            return self._amx_lsd_decode(
                transformer_out, noise, lsd_decode_steps
            ), out_eos
        return lsd_decode(self.flow_net, transformer_out, noise, lsd_decode_steps), out_eos

    def forward_verify_batch(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        model_state: dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        """Batch verification pass: process N positions at once, return all N outputs.

        Used for speculative decoding verification. The full model processes
        all draft positions in parallel using causal attention.
        """
        is_nan = mx.isnan(sequence)
        sequence = mx.where(is_nan, self.bos_emb, sequence)
        input_ = self.input_linear(sequence)

        if text_embeddings.shape[1] > 0:
            input_ = mx.concatenate([text_embeddings, input_], axis=1)
        transformer_out = self.transformer(input_, model_state)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        if text_embeddings.shape[1] > 0:
            transformer_out = transformer_out[:, -sequence.shape[1]:]

        if transformer_out.dtype != mx.float32:
            transformer_out = transformer_out.astype(mx.float32)

        # Process ALL N positions (not just last)
        n_positions = transformer_out.shape[1]
        eos_flags = self.out_eos(transformer_out).squeeze(-1) > eos_threshold  # (B, N)

        # Run flow_net on each position
        all_latents = []
        std = temp**0.5
        for i in range(n_positions):
            pos_out = transformer_out[:, i]
            noise_shape = pos_out.shape[:-1] + (self.ldim,)
            noise = mx.random.normal(noise_shape, dtype=mx.float32) * std
            if noise_clamp is not None:
                noise = mx.clip(noise, -noise_clamp, noise_clamp)
            if self._amx_flow_net is not None:
                latent = self._amx_lsd_decode(pos_out, noise, lsd_decode_steps)
            else:
                latent = lsd_decode(self.flow_net, pos_out, noise, lsd_decode_steps)
            all_latents.append(latent[:, None, :])

        all_latents = mx.concatenate(all_latents, axis=1)  # (B, N, ldim)
        return all_latents, eos_flags

    def _sample_next_latent(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        model_state: dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[mx.array, mx.array]:
        """Sample next latent from the model given a sequence and a set of conditions."""
        result = self(
            sequence=sequence,
            text_embeddings=text_embeddings,
            lsd_decode_steps=lsd_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            model_state=model_state,
        )

        return result

    @classmethod
    def from_pydantic_config(cls, config: FlowLMConfig, latent_dim: int) -> Self:
        d_model = config.transformer.d_model
        flow_mlp = SimpleMLPAdaLN.from_pydantic_config(config, latent_dim, d_model)

        conditioner = LUTConditioner(
            n_bins=config.lookup_table.n_bins,
            tokenizer_path=str(config.lookup_table.tokenizer_path),
            dim=config.lookup_table.dim,
            output_dim=d_model,
        )

        transformer = StreamingTransformer.from_pydantic_config(config.transformer)

        mlx_dtype = _DTYPE_MAP.get(config.dtype, mx.float32)
        return cls(
            flow_net=flow_mlp,
            transformer=transformer,
            dim=d_model,
            conditioner=conditioner,
            ldim=latent_dim,
            dtype=mlx_dtype,
        )
