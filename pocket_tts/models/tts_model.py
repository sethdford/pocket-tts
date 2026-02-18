import hashlib
import logging
import math
import os
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing_extensions import Self

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.ssml.parser import is_ssml, parse_ssml
from pocket_tts.ssml.prosody import apply_prosody, crossfade_segments, generate_silence
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import (
    StatefulModule,
    _named_modules,
    increment_steps,
    init_states,
)
from pocket_tts.utils.config import Config, load_config
from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    display_execution_time,
    download_if_necessary,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import (
    get_flow_lm_state_dict,
    get_mimi_state_dict,
    load_safetensors_as_mlx,
)

logger = logging.getLogger(__name__)

VOICE_CLONING_UNSUPPORTED = (
    f"We could not download the weights for the model with voice cloning, "
    f"but you're trying to use voice cloning. "
    f"Without voice cloning, you can use our catalog of voices {list(PREDEFINED_VOICES)}. "
    f"If you want access to the model with voice cloning, go to "
    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
    f"then make sure you're logged in locally with `uvx hf auth login`."
)


def _load_weights_from_dict(model: nn.Module, state_dict: dict[str, mx.array]):
    """Load weights into an MLX model from a flat state dict."""
    weights_list = list(state_dict.items())
    model.load_weights(weights_list)


def _cast_module_weights(module: nn.Module, dtype: mx.Dtype):
    """Cast all float32 weights in a module to the target dtype.

    Reduces memory bandwidth for the module's forward pass. Useful for
    decoder paths where reduced precision is acceptable.
    """
    new_weights = []
    for name, param in module.parameters().items():
        if isinstance(param, mx.array) and param.dtype == mx.float32:
            new_weights.append((name, param.astype(dtype)))
        elif isinstance(param, dict):
            for k, v in param.items():
                if isinstance(v, mx.array) and v.dtype == mx.float32:
                    new_weights.append((f"{name}.{k}", v.astype(dtype)))
    if new_weights:
        module.load_weights(new_weights)


class TTSModel(nn.Module):
    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold,
        config: Config,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True
        # Dict-based LRU cache for text prefill states (avoids lru_cache on unhashable self)
        self._prefill_cache: dict[str, tuple[dict, int]] = {}
        self._PREFILL_CACHE_SIZE = 32

    @property
    def device(self) -> str:
        return "gpu"  # MLX runs on Apple Silicon GPU

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls, config: Config, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm, latent_dim=config.mimi.quantizer.dimension
        )
        tts_model = cls(flow_lm, temp, lsd_decode_steps, noise_clamp, eos_threshold, config)
        return tts_model

    @classmethod
    def _from_pydantic_config_with_weights(
        cls, config: Config, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        tts_model = cls._from_pydantic_config(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.flow_lm.speaker_proj_weight = mx.zeros((1024, 512), dtype=mx.float32)

        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
            state_dict_flowlm = get_flow_lm_state_dict(
                download_if_necessary(config.flow_lm.weights_path)
            )
            _load_weights_from_dict(tts_model.flow_lm, state_dict_flowlm)

        mimi_config = config.mimi.model_dump()

        encoder = SEANetEncoder(**mimi_config["seanet"])
        decoder = SEANetDecoder(**mimi_config["seanet"])

        encoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        decoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        quantizer = DummyQuantizer(**mimi_config["quantizer"])

        tts_model.mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        )

        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
            mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
            _load_weights_from_dict(tts_model.mimi, mimi_state)

        if config.weights_path is not None:
            logger.info(f"Loading TTSModel weights from {config.weights_path}")
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            state_dict = load_safetensors_as_mlx(weights_file)
            _load_weights_from_dict(tts_model, state_dict)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )

        logging.info("TTS Model loaded successfully.")

        for top_module in (tts_model.flow_lm, tts_model.mimi):
            for module_name, module in _named_modules(top_module):
                if not isinstance(module, StatefulModule):
                    continue
                module._module_absolute_name = module_name

        # Inference optimizations: freeze params, set eval mode, materialize eagerly
        tts_model.eval()
        tts_model.freeze()
        mx.eval(tts_model.parameters())

        # Initialize compiled forward passes for fused Metal kernels
        tts_model.flow_lm.flow_net._init_compiled()

        # Initialize AMX hybrid flow network with fused C kernel.
        # The fused kernel runs the entire LSD decode in a single C call,
        # beating GPU's mx.compile by ~18% (1.19ms vs 1.39ms) by eliminating
        # all Python overhead and keeping intermediates in L1 cache.
        try:
            tts_model.flow_lm.init_amx_flow_net()
        except Exception:
            logger.warning("AMX flow network init failed, using GPU path", exc_info=True)

        # Pre-allocate constants reused every generation call
        ldim = tts_model.flow_lm.ldim
        dim = tts_model.flow_lm.dim
        dtype = tts_model.flow_lm.dtype
        tts_model._bos_nan_input = mx.full((1, 1, ldim), vals=float("nan"), dtype=dtype)
        tts_model._empty_text_tokens = mx.zeros((1, 0), dtype=mx.int32)
        tts_model._empty_latents = mx.zeros((1, 0, ldim), dtype=dtype)
        tts_model._empty_conditioning = mx.zeros((1, 0, dim), dtype=dtype)
        # Empty text embeddings for the generation fast path (skips conditioner + concat)
        tts_model._empty_text_embeddings = mx.zeros((1, 0, dim), dtype=dtype)
        mx.eval(
            tts_model._bos_nan_input,
            tts_model._empty_text_tokens,
            tts_model._empty_latents,
            tts_model._empty_conditioning,
            tts_model._empty_text_embeddings,
        )

        return tts_model

    @classmethod
    def load_model(
        cls,
        config: str | Path = DEFAULT_VARIANT,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        quantize: int | None = None,
        mimi_dtype: str | None = None,
    ) -> Self:
        """Load a pre-trained TTS model with specified configuration.

        This class method loads a complete TTS model including the flow language model
        and Mimi compression model from pre-trained weights. The model is initialized
        with the specified generation parameters and ready for inference.

        Args:
            config: a path to a custom YAML config file saved locally (e.g., C://pocket_tts/pocket_tts_config.yaml)
                or a model variant identifier (e.g., '610b0b2c'; must match a YAML file in the config directory).
            temp: Sampling temperature for generation. Higher values produce more
                diverse but potentially lower quality output.
            lsd_decode_steps: Number of steps for Lagrangian Self Distillation
                decoding. More steps can improve quality but increase computation.
            noise_clamp: Maximum value for noise sampling. If None, no clamping
                is applied. Helps prevent extreme values in generation.
            eos_threshold: Threshold for end-of-sequence detection. Higher values
                make the model more likely to continue generating.
            quantize: Optional quantization bit width (4 or 8). Quantizes Linear
                layers to reduce memory usage. None means no quantization (float32).
            mimi_dtype: Optional dtype for Mimi decoder ('bfloat16' or 'float16').
                Reduces memory bandwidth for the audio decoder path while keeping
                FlowLM in float32 for quality. None keeps float32.

        Returns:
            TTSModel: Fully initialized model with loaded weights, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.

        Example:
            ```python
            from pocket_tts import TTSModel

            # Load with default settings
            model = TTSModel.load_model()

            # Load with 4-bit quantization for lower memory usage
            model = TTSModel.load_model(quantize=4)

            # Load with bfloat16 Mimi decoder for lower latency
            model = TTSModel.load_model(mimi_dtype='bfloat16')
            ```
        """
        if str(config).endswith(".yaml"):
            config_path = Path(config)
            config = load_config(config_path)
            logger.info(f"Loading model from config at {config_path}...")
        else:
            config = load_config(Path(__file__).parents[1] / f"config/{config}.yaml")

        tts_model = TTSModel._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        if mimi_dtype is not None:
            dtype_map = {"bfloat16": mx.bfloat16, "float16": mx.float16}
            if mimi_dtype not in dtype_map:
                raise ValueError(f"mimi_dtype must be 'bfloat16' or 'float16', got {mimi_dtype}")
            target_dtype = dtype_map[mimi_dtype]
            logger.info("Casting Mimi decoder to %s for lower memory bandwidth", mimi_dtype)
            # Cast Mimi decoder weights to reduced precision
            _cast_module_weights(tts_model.mimi.decoder, target_dtype)
            _cast_module_weights(tts_model.mimi.decoder_transformer, target_dtype)
            _cast_module_weights(tts_model.mimi.upsample, target_dtype)

        if quantize is not None:
            if quantize not in (4, 8):
                raise ValueError(f"quantize must be 4 or 8, got {quantize}")
            logger.info("Quantizing model to %d bits...", quantize)
            nn.quantize(tts_model.flow_lm.transformer, bits=quantize)
            nn.quantize(tts_model.flow_lm.flow_net, bits=quantize)
            nn.quantize(tts_model.mimi.decoder_transformer, bits=quantize)
            nn.quantize(tts_model.mimi.encoder_transformer, bits=quantize)

        if mimi_dtype is not None or quantize is not None:
            tts_model.eval()
            tts_model.freeze()
            mx.eval(tts_model.parameters())
            # Re-initialize compiled forward after weight changes
            tts_model.flow_lm.flow_net._init_compiled()

        # Quantized models use nn.QuantizedLinear, incompatible with AMX weight extraction
        if quantize is not None and tts_model.flow_lm._amx_flow_net is not None:
            tts_model.flow_lm._amx_flow_net = None
            logger.info("AMX flow network disabled (quantized model uses GPU)")

        # Warm up all Metal shaders with a dummy forward pass.
        # Metal JIT-compiles GPU kernels on first use, adding 2-5s cold-start
        # latency. By doing this during load, the first real generation is fast.
        tts_model._warmup_metal_shaders()

        return tts_model

    def _warmup_metal_shaders(self):
        """Run a dummy forward pass to JIT-compile all Metal shaders.

        Metal kernels (SDPA, RoPE, LayerNorm, RMSNorm, matmul, etc.) are
        compiled on first invocation, adding 2-5s cold-start latency. By
        executing a minimal generation pass during model load, all shader
        variants are cached in the Metal pipeline state cache, making the
        first real generation instant.

        This is a critical UX advantage: no other on-device TTS (MLX-Audio,
        ChipChat, Kokoro) pre-warms shaders.
        """
        logger.info("Warming up Metal shaders...")
        t0 = time.monotonic()

        # Create throwaway states (not saved anywhere)
        warmup_flow_state = init_states(self.flow_lm, batch_size=1, sequence_length=1)
        warmup_mimi_state = init_states(
            self.mimi, batch_size=1, sequence_length=self.config.mimi.transformer.context
        )

        # 1) Trigger text conditioner + prefill path (covers in_proj, RoPE, SDPA with T>1)
        warmup_text = self.flow_lm.conditioner.prepare("Hello.")
        self._run_flow_lm_and_increment_step(
            model_state=warmup_flow_state, text_tokens=warmup_text.tokens
        )

        # 2) Trigger streaming path (covers compiled pre-attn, SDPA with T=1, flow_net)
        warmup_output, _ = self._run_flow_lm_generation_step(
            warmup_flow_state, self._bos_nan_input
        )
        mx.eval(warmup_output)

        # 3) Trigger Mimi decode path (covers SEANet decoder, conv, upsample)
        mimi_input = warmup_output * self.flow_lm.emb_std + self.flow_lm.emb_mean
        quantized = self.mimi.quantizer(mimi_input)
        audio_frame = self.mimi.decode_from_latent(quantized, warmup_mimi_state)
        mx.eval(audio_frame)

        # Release warmup memory
        del warmup_flow_state, warmup_mimi_state, warmup_output, audio_frame
        mx.clear_cache()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info("Metal shader warmup complete in %d ms", elapsed_ms)

    def _run_flow_lm_and_increment_step(
        self,
        model_state: dict,
        text_tokens: mx.array | None = None,
        backbone_input_latents: mx.array | None = None,
        audio_conditioning: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """First one is the backbone output, second one is the audio decoding output."""
        if text_tokens is None:
            text_tokens = self._empty_text_tokens
        if backbone_input_latents is None:
            backbone_input_latents = self._empty_latents
        if audio_conditioning is None:
            audio_conditioning = self._empty_conditioning

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm_generation_step(
        self,
        model_state: dict,
        backbone_input: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Fast path for autoregressive generation steps.

        Avoids empty array creation, unnecessary concatenations, and extra function
        call levels that the general _run_flow_lm_and_increment_step path has.
        """
        output_embeddings, is_eos = self.flow_lm(
            sequence=backbone_input,
            text_embeddings=self._empty_text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        increment_steps(self.flow_lm, model_state, increment=1)
        return output_embeddings[:, None, :], is_eos

    def _run_flow_lm(
        self,
        model_state: dict,
        text_tokens: mx.array,
        backbone_input_latents: mx.array,
        audio_conditioning: mx.array,
    ) -> tuple[mx.array, mx.array]:
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        if audio_conditioning.shape[1] > 0:
            text_embeddings = mx.concatenate([text_embeddings, audio_conditioning], axis=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _encode_audio(self, audio: mx.array) -> mx.array:
        # audio is NLC: (B, T, C)
        encoded = self.mimi.encode_to_latent(audio)
        # encoded is NLC: (B, T, C) - no transpose needed
        latents = encoded.astype(mx.float32)
        # F.linear(latents, weight) = latents @ weight.T
        conditioning = latents @ self.flow_lm.speaker_proj_weight.T
        return conditioning

    def _flow_lm_current_offset(self, model_state: dict) -> int:
        """Get the current offset from the FlowLM model state."""
        for module_state in model_state.values():
            offset = module_state.get("offset")
            if offset is not None:
                return int(offset.item())
        raise ValueError(
            "Could not find offset in model state, please open an issue "
            "at https://github.com/kyutai-labs/pocket-tts/issues"
        )

    def generate_audio(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        speculative_tokens: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate complete audio from text input (supports SSML).

        This method generates the full audio output for the given text prompt
        and returns it as a single numpy array. If the text starts with `<speak>`,
        it is parsed as SSML and processed accordingly.

        Args:
            model_state: Model state dictionary containing hidden states.
            text_to_generate: Input text or SSML document to convert to speech.
            frames_after_eos: Number of additional frames to generate after EOS.
            copy_state: Whether to create a deep copy of the model state.
            speculative_tokens: If set, use speculative frame generation.
            seed: Random seed for reproducible generation. When set, the MLX
                PRNG is seeded before each chunk's generation loop, producing
                bit-identical output across runs with the same parameters.

        Returns:
            np.ndarray: Generated audio array with shape [samples] at the model's
                sample rate (typically 24kHz).
        """
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
            speculative_tokens=speculative_tokens,
            seed=seed,
        ):
            audio_chunks.append(chunk)
        return np.concatenate(audio_chunks, axis=0)

    def generate_audio_stream(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        speculative_tokens: int | None = None,
        seed: int | None = None,
    ):
        """Generate audio streaming chunks from text input (supports SSML).

        If the text starts with `<speak>`, it is parsed as W3C SSML and each
        segment is synthesized with the appropriate voice, prosody, breaks, etc.
        Otherwise, plain text is processed as before.

        Args:
            speculative_tokens: If set, use speculative frame generation with
                this many draft tokens per round (e.g., 4). Uses a lightweight
                draft model to generate candidates, then verifies in batch.
            seed: Random seed for reproducible generation. When set, the MLX
                PRNG is seeded before each chunk's generation loop, producing
                bit-identical output across runs with the same parameters.

        Yields:
            np.ndarray: Audio chunks with shape [samples] at the model's sample rate.
        """
        if is_ssml(text_to_generate):
            yield from self._generate_audio_stream_ssml(
                model_state=model_state,
                ssml_text=text_to_generate,
                max_tokens=max_tokens,
                frames_after_eos=frames_after_eos,
                copy_state=copy_state,
            )
        else:
            yield from self._generate_audio_stream_plain(
                model_state=model_state,
                text_to_generate=text_to_generate,
                max_tokens=max_tokens,
                frames_after_eos=frames_after_eos,
                copy_state=copy_state,
                speculative_tokens=speculative_tokens,
                seed=seed,
            )

    def _generate_audio_stream_plain(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        skip_text_prep: bool = False,
        speculative_tokens: int | None = None,
        seed: int | None = None,
    ):
        """Plain text generation (original pipeline).

        Args:
            skip_text_prep: If True, skip prepare_text_prompt (used for SSML
                segments where text has already been normalized).
            speculative_tokens: If set, use speculative frame generation with
                this many draft tokens per round.
            seed: Random seed for reproducible generation.
        """
        if skip_text_prep:
            chunks = _split_into_token_chunks(
                self.flow_lm.conditioner.tokenizer, text_to_generate, max_tokens
            )
        else:
            chunks = split_into_best_sentences(
                self.flow_lm.conditioner.tokenizer, text_to_generate, max_tokens
            )

        for chunk_idx, chunk in enumerate(chunks):
            _, frames_after_eos_guess = prepare_text_prompt(chunk)
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            chunk_seed = (seed + chunk_idx) if seed is not None else None
            if speculative_tokens and speculative_tokens > 1:
                yield from self._generate_speculative(
                    model_state=model_state,
                    text_to_generate=chunk,
                    frames_after_eos=effective_frames,
                    copy_state=copy_state,
                    num_draft_tokens=speculative_tokens,
                    seed=chunk_seed,
                )
            else:
                yield from self._generate_audio_stream_short_text(
                    model_state=model_state,
                    text_to_generate=chunk,
                    frames_after_eos=effective_frames,
                    copy_state=copy_state,
                    seed=chunk_seed,
                )

    def _generate_audio_stream_ssml(
        self,
        model_state: dict,
        ssml_text: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ):
        """SSML-aware generation pipeline with CPU/GPU pipelining.

        Parses SSML into segments and generates audio for each segment with
        appropriate voice switching, prosody adjustments, breaks, and audio mixing.

        CPU/GPU Pipelining: Prosody post-processing (phase vocoder FFT, resampling,
        volume limiting) runs on CPU/AMX via Apple Accelerate while GPU generates
        the next segment. Since AMX is a separate hardware unit from the GPU, they
        execute concurrently without contention, hiding prosody latency entirely.
        """
        segments = parse_ssml(ssml_text)
        voice_states: dict[str | None, dict] = {None: model_state}
        current_voice = None
        prev_segment_audio: np.ndarray | None = None
        total_samples_yielded = 0

        # Single-thread executor for CPU/AMX prosody processing overlapped with GPU
        prosody_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prosody")
        pending_prosody_future: Future | None = None
        pending_prosody_chunks: list[np.ndarray] | None = None

        def _collect_prosody_result() -> np.ndarray | None:
            """Collect the result from a pending prosody future."""
            nonlocal pending_prosody_future, pending_prosody_chunks
            if pending_prosody_future is not None:
                segment_audio = pending_prosody_future.result()
                pending_prosody_future = None
                pending_prosody_chunks = None
                return segment_audio
            if pending_prosody_chunks is not None:
                segment_audio = np.concatenate(pending_prosody_chunks, axis=0)
                pending_prosody_chunks = None
                return segment_audio
            return None

        try:
            for segment in segments:
                # Insert silence for break_before
                if segment.break_before_ms > 0:
                    # Must flush any pending prosody + previous audio first
                    prosody_result = _collect_prosody_result()
                    if prosody_result is not None:
                        if prev_segment_audio is not None:
                            combined = crossfade_segments(
                                prev_segment_audio, prosody_result, self.sample_rate
                            )
                            prev_segment_audio = combined
                        else:
                            prev_segment_audio = prosody_result

                    silence = generate_silence(self.sample_rate, segment.break_before_ms)
                    if prev_segment_audio is not None:
                        yield prev_segment_audio
                        total_samples_yielded += len(prev_segment_audio)
                        prev_segment_audio = None
                    yield silence
                    total_samples_yielded += len(silence)

                # Handle voice switching
                if segment.voice and segment.voice != current_voice:
                    if segment.voice not in voice_states:
                        try:
                            voice_states[segment.voice] = self.get_state_for_audio_prompt(
                                segment.voice
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not load voice '%s': %s. Using current voice.",
                                segment.voice,
                                e,
                            )
                            voice_states[segment.voice] = model_state
                    current_voice = segment.voice

                active_state = voice_states.get(current_voice, model_state)

                # Handle audio segments (from <audio> tag)
                if segment.is_audio:
                    # Collect any pending prosody before handling audio
                    prosody_result = _collect_prosody_result()
                    if prosody_result is not None:
                        if prev_segment_audio is not None:
                            combined = crossfade_segments(
                                prev_segment_audio, prosody_result, self.sample_rate
                            )
                            prev_segment_audio = combined
                        else:
                            prev_segment_audio = prosody_result

                    try:
                        audio_data, sr = audio_read(
                            download_if_necessary(segment.text)
                        )
                        if sr != self.sample_rate:
                            audio_data = convert_audio(audio_data, sr, self.sample_rate, 1)
                        segment_audio = audio_data[0]
                    except Exception as e:
                        logger.warning("Could not load audio '%s': %s", segment.text, e)
                        continue

                    # Crossfade with previous segment
                    if prev_segment_audio is not None:
                        combined = crossfade_segments(
                            prev_segment_audio, segment_audio, self.sample_rate
                        )
                        yield combined
                        total_samples_yielded += len(combined)
                        prev_segment_audio = None
                    else:
                        yield segment_audio
                        total_samples_yielded += len(segment_audio)
                    continue

                # Populate mark events with sample offsets
                for mark in segment.marks:
                    mark.offset_samples = total_samples_yielded

                # Handle text segments with CPU/GPU pipelining
                if segment.has_text:
                    # Collect previous prosody result before starting new generation.
                    # This ensures the previous segment's prosody (running on CPU/AMX)
                    # has finished before we integrate it.
                    prosody_result = _collect_prosody_result()
                    if prosody_result is not None:
                        if prev_segment_audio is not None:
                            combined = crossfade_segments(
                                prev_segment_audio, prosody_result, self.sample_rate
                            )
                            prev_segment_audio = combined
                        else:
                            prev_segment_audio = prosody_result

                    # Generate audio on GPU
                    segment_audio_chunks = []
                    for chunk in self._generate_audio_stream_plain(
                        model_state=active_state,
                        text_to_generate=segment.text,
                        max_tokens=max_tokens,
                        frames_after_eos=frames_after_eos,
                        copy_state=copy_state,
                        skip_text_prep=True,
                    ):
                        segment_audio_chunks.append(chunk)

                    if not segment_audio_chunks:
                        continue

                    # Determine if prosody processing is needed
                    needs_prosody = (
                        abs(segment.prosody.rate - 1.0) > 0.01
                        or abs(segment.prosody.pitch - 1.0) > 0.01
                        or abs(segment.prosody.volume - 1.0) > 0.01
                    )

                    if needs_prosody:
                        # Submit prosody processing to background thread (CPU/AMX).
                        # This runs concurrently with GPU generation of the NEXT
                        # segment, overlapping CPU prosody with GPU inference.
                        full_audio = np.concatenate(segment_audio_chunks, axis=0)
                        sample_rate = self.sample_rate
                        prosody_params = segment.prosody

                        pending_prosody_future = prosody_executor.submit(
                            apply_prosody, full_audio, sample_rate, prosody_params
                        )
                    else:
                        # No prosody needed -- just stash chunks for later integration
                        pending_prosody_chunks = segment_audio_chunks

                # Insert silence for break_after (flush previous segment first)
                if segment.break_after_ms > 0:
                    # Collect pending prosody
                    prosody_result = _collect_prosody_result()
                    if prosody_result is not None:
                        if prev_segment_audio is not None:
                            combined = crossfade_segments(
                                prev_segment_audio, prosody_result, self.sample_rate
                            )
                            prev_segment_audio = combined
                        else:
                            prev_segment_audio = prosody_result

                    if prev_segment_audio is not None:
                        yield prev_segment_audio
                        total_samples_yielded += len(prev_segment_audio)
                        prev_segment_audio = None
                    silence = generate_silence(self.sample_rate, segment.break_after_ms)
                    yield silence
                    total_samples_yielded += len(silence)

            # Flush any pending prosody + remaining audio
            prosody_result = _collect_prosody_result()
            if prosody_result is not None:
                if prev_segment_audio is not None:
                    combined = crossfade_segments(
                        prev_segment_audio, prosody_result, self.sample_rate
                    )
                    prev_segment_audio = combined
                else:
                    prev_segment_audio = prosody_result

            if prev_segment_audio is not None:
                yield prev_segment_audio
        finally:
            prosody_executor.shutdown(wait=True)

    def _make_prefill_cache_key(self, voice_state: dict, text: str) -> str:
        """Create a cache key from the voice state identity and text content.

        The voice state is identified by the offset of its first module (which
        encodes the voice prompt length), combined with a hash of the text.
        """
        voice_id = 0
        for module_state in voice_state.values():
            offset = module_state.get("offset")
            if offset is not None:
                voice_id = int(offset.item())
                break
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"{voice_id}:{text_hash}"

    def _get_prefilled_state(self, voice_state: dict, text: str) -> tuple[dict, int]:
        """Get a prefilled model state, using cache when available.

        Caches the FlowLM state after text prefill for repeated text patterns.
        For server use, the same text is often generated multiple times (e.g.,
        UI notifications, greetings). Caching skips tokenization + embedding +
        transformer prefill (~50-200ms).

        Returns a fresh deep copy of the cached prefill state and the token count.
        """
        cache_key = self._make_prefill_cache_key(voice_state, text)

        if cache_key in self._prefill_cache:
            cached_state, token_count = self._prefill_cache[cache_key]
            return _copy_model_state(cached_state), token_count

        # Cache miss: compute the prefill
        model_state = _copy_model_state(voice_state)
        prepared = self.flow_lm.conditioner.prepare(text)
        token_count = prepared.tokens.shape[1]
        self._run_flow_lm_and_increment_step(
            model_state=model_state, text_tokens=prepared.tokens
        )

        # Store in cache (evict oldest if full)
        if len(self._prefill_cache) >= self._PREFILL_CACHE_SIZE:
            oldest_key = next(iter(self._prefill_cache))
            del self._prefill_cache[oldest_key]
        self._prefill_cache[cache_key] = (model_state, token_count)

        return _copy_model_state(model_state), token_count

    def _generate_audio_stream_short_text(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        seed: int | None = None,
    ):
        """Single-threaded generation with async eval pipelining.

        Uses mx.async_eval to overlap compute: while one step's latent is being
        evaluated on the GPU, the next step's computation graph is being built on CPU.
        """
        debug = logger.isEnabledFor(logging.DEBUG)
        t_generating = time.monotonic()

        # Text prefill with caching — skip tokenization + embedding + transformer
        # prefill when the same voice+text combination is requested again.
        prefilled_state, token_count = self._get_prefilled_state(
            model_state, text_to_generate
        )
        model_state = prefilled_state

        # Seed AFTER prefill so the generation loop is deterministic regardless
        # of whether the prefill was cached (which would skip RNG calls).
        # Force-eval the model state first to materialize any pending lazy random
        # ops from the prefill — otherwise those ops could consume seed state.
        if seed is not None:
            mx.eval(*[v for s in model_state.values() for v in s.values() if isinstance(v, mx.array)])
            mx.random.seed(seed)

        # Initialize Mimi decoder state
        mimi_context = self.config.mimi.transformer.context
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=mimi_context)

        # Pre-allocated BOS NaN marker — reused every call instead of re-creating
        backbone_input = self._bos_nan_input
        total_generated_samples = 0
        max_gen_len = self._estimate_max_gen_len(token_count)
        steps_times = [] if debug else None
        eos_step = None
        pending_audio_frame = None

        # Pre-fetch scaling constants to avoid repeated attribute lookups in hot loop
        emb_std = self.flow_lm.emb_std
        emb_mean = self.flow_lm.emb_mean
        mimi_quantizer = self.mimi.quantizer
        mimi_decode = self.mimi.decode_from_latent
        _increment_mimi = self._increment_mimi_steps

        # Use fast path that skips empty array creation and extra function calls
        _gen_step = self._run_flow_lm_generation_step

        for generation_step in range(max_gen_len):
            if debug:
                with display_execution_time("Generating latent", print_output=False) as timer:
                    next_latent, is_eos = _gen_step(model_state, backbone_input)
                    mx.eval(next_latent, is_eos)
                    if is_eos.item() and eos_step is None:
                        eos_step = generation_step
                    if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                        break
                steps_times.append(timer.elapsed_time_ms)
            else:
                next_latent, is_eos = _gen_step(model_state, backbone_input)
                mx.eval(next_latent, is_eos)
                if is_eos.item() and eos_step is None:
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

            # Decode latent to audio — fused scale+shift, async eval for pipelining
            mimi_decoding_input = next_latent * emb_std + emb_mean
            quantized = mimi_quantizer(mimi_decoding_input)
            audio_frame = mimi_decode(quantized, mimi_state)
            _increment_mimi(mimi_state)

            # Yield the *previous* frame while current decode runs on GPU
            if pending_audio_frame is not None:
                chunk_np = np.array(pending_audio_frame[0, :, 0])
                total_generated_samples += chunk_np.shape[0]
                yield chunk_np

            # Schedule current audio_frame for async evaluation
            mx.async_eval(audio_frame)
            pending_audio_frame = audio_frame

            backbone_input = next_latent

            # Periodic memory cleanup (bitwise AND avoids modulo overhead)
            if not (generation_step & 63):
                mx.clear_cache()

        # Yield the final pending frame
        if pending_audio_frame is not None:
            chunk_np = np.array(pending_audio_frame[0, :, 0])
            total_generated_samples += chunk_np.shape[0]
            yield chunk_np
        else:
            if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, "
                "this very often indicates an error."
            )

        if debug and steps_times:
            logger.debug("Average generation step time: %d ms", int(statistics.mean(steps_times)))

        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / max(generation_time, 1)

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    def _generate_speculative(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        num_draft_tokens: int = 4,
        num_draft_layers: int = 2,
        acceptance_threshold: float = 1.0,
        seed: int | None = None,
    ):
        """Speculative frame generation for faster inference.

        Uses a lightweight "draft" model (first N transformer layers) to generate
        multiple candidate frames, then verifies them in parallel with the full
        model. Accepted frames are decoded through Mimi in batches for better
        GPU utilization.

        Based on Speech Speculative Decoding (arxiv:2505.15380) adapted for
        continuous latent flow matching.

        Args:
            num_draft_tokens: Number of candidate frames per speculation round.
            num_draft_layers: Number of transformer layers for the draft model.
            acceptance_threshold: L2 distance threshold for accepting draft frames.
                Lower = stricter (higher quality), higher = more accepted (faster).
            seed: Random seed for reproducible generation.
        """
        if copy_state:
            model_state = _copy_model_state(model_state)

        t_generating = time.monotonic()

        # Prepare text
        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.tokens.shape[1]
        self._run_flow_lm_and_increment_step(
            model_state=model_state, text_tokens=prepared.tokens
        )

        # Seed AFTER prefill for deterministic generation.
        # Force-eval pending lazy ops from the prefill first.
        if seed is not None:
            mx.eval(*[v for s in model_state.values() for v in s.values() if isinstance(v, mx.array)])
            mx.random.seed(seed)

        mimi_context = self.config.mimi.transformer.context
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=mimi_context)

        backbone_input = self._bos_nan_input
        total_generated_samples = 0
        max_gen_len = self._estimate_max_gen_len(token_count)
        emb_std = self.flow_lm.emb_std
        emb_mean = self.flow_lm.emb_mean
        empty_text = self._empty_text_embeddings
        eos_found = False
        generation_step = 0
        total_accepted = 0
        total_speculated = 0

        while generation_step < max_gen_len and not eos_found:
            remaining = max_gen_len - generation_step
            n_draft = min(num_draft_tokens, remaining)

            # --- Phase 1: Draft generation (lightweight, first N layers) ---
            draft_state = _copy_draft_state(
                model_state, self.flow_lm, num_draft_layers
            )
            draft_latents = []
            draft_input = backbone_input
            for _ in range(n_draft):
                latent, d_eos = self.flow_lm.forward_draft(
                    sequence=draft_input,
                    text_embeddings=empty_text,
                    model_state=draft_state,
                    lsd_decode_steps=self.lsd_decode_steps,
                    temp=self.temp,
                    noise_clamp=self.noise_clamp,
                    eos_threshold=self.eos_threshold,
                    num_draft_layers=num_draft_layers,
                )
                latent = latent[:, None, :]
                draft_latents.append(latent)
                increment_steps(self.flow_lm, draft_state, increment=1)
                draft_input = latent

            # --- Phase 2: Full model verification (batch) ---
            # Input sequence: [current backbone_input, draft_1, ..., draft_{N-1}]
            verify_inputs = mx.concatenate(
                [backbone_input] + draft_latents[:-1], axis=1
            )  # (1, N, ldim)

            full_latents, eos_flags = self.flow_lm.forward_verify_batch(
                sequence=verify_inputs,
                text_embeddings=empty_text,
                model_state=model_state,
                lsd_decode_steps=self.lsd_decode_steps,
                temp=self.temp,
                noise_clamp=self.noise_clamp,
                eos_threshold=self.eos_threshold,
            )
            # full_latents: (1, N, ldim), eos_flags: (1, N)

            # --- Phase 3: Accept/reject ---
            draft_stack = mx.concatenate(draft_latents, axis=1)  # (1, N, ldim)
            distances = mx.sqrt(
                mx.sum((draft_stack - full_latents) ** 2, axis=-1)
            )  # (1, N)
            mx.eval(distances, full_latents, eos_flags)

            num_accepted = 0
            for i in range(n_draft):
                if distances[0, i].item() < acceptance_threshold:
                    num_accepted += 1
                else:
                    break

            # Always produce at least 1 frame (the full model's correction)
            total_frames = min(num_accepted + 1, n_draft)
            total_accepted += num_accepted
            total_speculated += n_draft

            # Roll back KV cache if we didn't use all N positions
            if total_frames < n_draft:
                _truncate_kv_caches(model_state, n_draft - total_frames)

            increment_steps(self.flow_lm, model_state, increment=total_frames)

            # --- Phase 4: Batched Mimi decode ---
            accepted = full_latents[:, :total_frames, :]
            mimi_input = accepted * emb_std + emb_mean
            quantized = self.mimi.quantizer(mimi_input)
            audio_frames = self.mimi.decode_from_latent(quantized, mimi_state)
            increment_steps(self.mimi, mimi_state, increment=16 * total_frames)
            mx.eval(audio_frames)

            # Yield all decoded audio
            chunk_np = np.array(audio_frames[0, :, 0])
            total_generated_samples += chunk_np.shape[0]
            yield chunk_np

            # Check EOS in any accepted frame
            for i in range(total_frames):
                if eos_flags[0, i].item():
                    eos_found = True
                    break

            # Set up next iteration
            backbone_input = full_latents[:, total_frames - 1:total_frames, :]
            generation_step += total_frames

            if not (generation_step & 63):
                mx.clear_cache()

        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / max(generation_time, 1)
        acceptance_rate = total_accepted / max(total_speculated, 1)

        logger.info(
            "Speculative: %d ms audio in %d ms (%.2fx RT), "
            "acceptance rate: %.0f%% (%d/%d)",
            duration_generated_audio,
            generation_time,
            real_time_factor,
            acceptance_rate * 100,
            total_accepted,
            total_speculated,
        )

    @lru_cache(maxsize=16)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str, truncate: bool = False
    ) -> dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | np.ndarray, truncate: bool = False
    ) -> dict:
        """Create model state conditioned on audio prompt for continuation.

        Args:
            audio_conditioning: Audio prompt to condition. Can be:
                - Path: Local file path to audio file (or .safetensors)
                - str: URL to download audio file (or .safetensors) from
                - np.ndarray: Pre-loaded audio array with shape [channels, samples]
            truncate: Whether to truncate long audio prompts to 30 seconds.

        Returns:
            dict: Model state dictionary conditioned on the audio prompt.

        Example:
            ```python
            from pocket_tts import TTSModel

            model = TTSModel.load_model()
            voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
            ```
        """
        if isinstance(audio_conditioning, (str, Path)) and str(audio_conditioning).endswith(
            ".safetensors"
        ):
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            return _import_model_state(audio_conditioning)

        elif isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICES:
            return _import_model_state(download_if_necessary(PREDEFINED_VOICES[audio_conditioning]))

        if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
            raise ValueError(VOICE_CLONING_UNSUPPORTED)

        if isinstance(audio_conditioning, str):
            audio_conditioning = download_if_necessary(audio_conditioning)

        if isinstance(audio_conditioning, Path):
            audio, conditioning_sample_rate = audio_read(audio_conditioning)

            if truncate:
                max_samples = int(30 * conditioning_sample_rate)
                if audio.shape[-1] > max_samples:
                    audio = audio[..., :max_samples]
                    logger.info(f"Audio truncated to first 30 seconds ({max_samples} samples)")

            audio_conditioning = convert_audio(
                audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
            )

        # Convert numpy to MLX: audio_conditioning is (channels, samples), need NLC: (1, samples, channels)
        if isinstance(audio_conditioning, np.ndarray):
            # audio_conditioning shape: (channels, samples) -> (1, samples, channels)
            audio_mx = mx.array(audio_conditioning.T[np.newaxis, :, :])
        else:
            audio_mx = audio_conditioning

        with display_execution_time("Encoding audio prompt"):
            prompt = self._encode_audio(audio_mx)

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=prompt.shape[1])

        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        logger.info(
            "Size of the model state for audio prompt: %d MB", size_of_dict(model_state) // 1_000_000
        )

        return model_state

    @staticmethod
    def list_voices() -> list[str]:
        """Return a list of available predefined voice names.

        Returns:
            list[str]: Names of predefined voices that can be passed directly
                to get_state_for_audio_prompt().

        Example:
            ```python
            from pocket_tts import TTSModel

            voices = TTSModel.list_voices()
            print(voices)  # ['alba', 'marius', 'javert', ...]
            ```
        """
        return list(PREDEFINED_VOICES.keys())

    def _increment_mimi_steps(self, mimi_state: dict):
        """Fast path for incrementing Mimi state by 16 (one frame's worth of samples)."""
        increment_steps(self.mimi, mimi_state, increment=16)

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        frame_rate = self.config.mimi.frame_rate
        return math.ceil(gen_len_sec * frame_rate)


def prepare_text_prompt(text: str) -> tuple[str, int]:
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    if text[-1].isalnum():
        text = text + "."

    if len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


def split_into_best_sentences(tokenizer, text_to_generate: str, max_tokens: int) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(text_to_generate)
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = np.array(tokens.tokens[0]).tolist()

    _, *end_of_sentence_tokens = np.array(tokenizer(".!...?").tokens[0]).tolist()

    end_of_sentences_indices = [0]
    previous_was_end_of_sentence_token = False

    for token_idx, token in enumerate(list_of_tokens):
        if token in end_of_sentence_tokens:
            previous_was_end_of_sentence_token = True
        else:
            if previous_was_end_of_sentence_token:
                end_of_sentences_indices.append(token_idx)
            previous_was_end_of_sentence_token = False
    end_of_sentences_indices.append(len(list_of_tokens))

    nb_tokens_and_sentences = []
    for i in range(len(end_of_sentences_indices) - 1):
        start = end_of_sentences_indices[i]
        end = end_of_sentences_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        nb_tokens_and_sentences.append((end - start, text))

    max_nb_tokens_in_a_chunk = max_tokens
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in nb_tokens_and_sentences:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    return chunks


def _copy_model_state(model_state: dict) -> dict:
    """Efficiently copy model state by only copying mutable arrays (KV caches, offsets).

    This is much faster than copy.deepcopy since it avoids copying the entire
    nested dict structure and only duplicates the arrays that will be mutated
    during generation (caches and state buffers).
    """
    result = {}
    for module_name, module_state in model_state.items():
        copied = {}
        for key, val in module_state.items():
            if isinstance(val, mx.array):
                # Copy arrays that are mutated during generation
                copied[key] = mx.array(val)
            else:
                copied[key] = val
        result[module_name] = copied
    return result


def _copy_draft_state(
    full_state: dict, flow_lm, num_draft_layers: int
) -> dict:
    """Create a model state for the draft model (first N layers only).

    Copies only the KV caches for the first `num_draft_layers` transformer
    layers, which is all the draft model needs.
    """
    from pocket_tts.modules.stateful_module import _get_stateful_modules

    draft_state = {}
    draft_layer_prefixes = set()
    for i in range(num_draft_layers):
        draft_layer_prefixes.add(f"transformer.layers.{i}.")

    for name, _ in _get_stateful_modules(flow_lm):
        if name not in full_state:
            continue
        # Include only layers used by the draft model
        is_draft_layer = any(name.startswith(p) for p in draft_layer_prefixes)
        if is_draft_layer:
            # Deep copy the state for draft layers (they'll be mutated)
            orig = full_state[name]
            draft_state[name] = {
                k: mx.array(v) if isinstance(v, mx.array) else v
                for k, v in orig.items()
            }

    return draft_state


def _truncate_kv_caches(model_state: dict, num_to_remove: int):
    """Roll back KV caches by removing the last N entries.

    Used after speculative decoding when not all draft positions were accepted.
    """
    for module_state in model_state.values():
        if "k_cache" in module_state:
            if num_to_remove > 0 and module_state["k_cache"].shape[2] >= num_to_remove:
                module_state["k_cache"] = module_state["k_cache"][:, :, :-num_to_remove, :]
                module_state["v_cache"] = module_state["v_cache"][:, :, :-num_to_remove, :]


def _split_into_token_chunks(tokenizer, text: str, max_tokens: int) -> list[str]:
    """Split text into chunks by token count without applying prepare_text_prompt.

    Used for SSML segments where the text has already been normalized and
    should not have capitalization, periods, or padding applied.
    """
    text = text.strip()
    if not text:
        return []

    tokens = tokenizer(text)
    list_of_tokens = np.array(tokens.tokens[0]).tolist()

    if len(list_of_tokens) <= max_tokens:
        return [text]

    # Split at sentence-ending punctuation boundaries
    _, *end_of_sentence_tokens = np.array(tokenizer(".!...?").tokens[0]).tolist()
    end_of_sentences_indices = [0]
    previous_was_end_of_sentence_token = False
    for token_idx, token in enumerate(list_of_tokens):
        if token in end_of_sentence_tokens:
            previous_was_end_of_sentence_token = True
        else:
            if previous_was_end_of_sentence_token:
                end_of_sentences_indices.append(token_idx)
            previous_was_end_of_sentence_token = False
    end_of_sentences_indices.append(len(list_of_tokens))

    chunks = []
    current_chunk = ""
    current_nb = 0
    for i in range(len(end_of_sentences_indices) - 1):
        start = end_of_sentences_indices[i]
        end = end_of_sentences_indices[i + 1]
        nb = end - start
        sentence = tokenizer.sp.decode(list_of_tokens[start:end])
        if current_chunk == "":
            current_chunk = sentence
            current_nb = nb
            continue
        if current_nb + nb > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb = nb
        else:
            current_chunk += " " + sentence
            current_nb += nb
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def export_model_state(model_state: dict[str, dict[str, mx.array]], dest: str | Path):
    dict_to_store = {}
    for module_name, module_state in model_state.items():
        for key, array_value in module_state.items():
            if not isinstance(array_value, mx.array):
                array_value = mx.array(np.array(array_value))
            dict_to_store[f"{module_name}/{key}"] = array_value
    mx.save_safetensors(str(dest), dict_to_store)


def _import_model_state(source: str | Path) -> dict[str, dict[str, mx.array]]:
    result = {}
    tensors = mx.load(str(source))
    for key, arr in tensors.items():
        module_name, tensor_key = key.split("/")
        result.setdefault(module_name, {})
        result[module_name][tensor_key] = arr

    # Convert PyTorch-format attention states to MLX format.
    # PyTorch used: cache (2, B, T, H, D) + current_end (N,)
    # MLX uses: offset (scalar) + k_cache (B, H, T, D) + v_cache (B, H, T, D)
    for module_name, state in result.items():
        if "cache" in state and "current_end" in state:
            cache = state.pop("cache")
            current_end = state.pop("current_end")
            state["offset"] = mx.array(int(current_end[0].item()), dtype=mx.int32)
            # PyTorch cache is (2, B, T, H, D) -> transpose to (B, H, T, D)
            state["k_cache"] = cache[0].transpose(0, 2, 1, 3)
            state["v_cache"] = cache[1].transpose(0, 2, 1, 3)
    return result
