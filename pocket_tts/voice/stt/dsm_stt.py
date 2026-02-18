"""DSM STT 1B backend using moshi_mlx for streaming speech-to-text on Apple Silicon.

Wraps Kyutai's STT model (stt-1b-en_fr or stt-2.6b-en) via moshi_mlx for
native MLX inference. Uses rustymimi for audio tokenization (Rust-native Mimi
encoder) and moshi_mlx.models.LmGen for streaming text generation.

Key advantages over the Rust cdylib and PyTorch backends:
  - No Rust toolchain or cargo build required (rustymimi is a pip wheel)
  - Native MLX Metal acceleration on Apple Silicon
  - Built-in semantic VAD via model's extra heads (1B model only)
  - Same moshi_mlx package used for DSM TTS, sharing code paths
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import numpy as np

from pocket_tts.voice.stt.base import STTBackend

logger = logging.getLogger(__name__)

DEFAULT_STT_HF_REPO = "kyutai/stt-1b-en_fr-mlx"

_HF_REPO_MAP = {
    "kyutai/stt-1b-en_fr": "kyutai/stt-1b-en_fr-mlx",
    "kyutai/stt-2.6b-en": "kyutai/stt-2.6b-en-mlx",
}


class DsmSTT(STTBackend):
    """Streaming STT using Kyutai's DSM model on MLX with semantic VAD.

    Audio frames are encoded with rustymimi (Rust Mimi encoder) and fed
    to the LmGen language model which produces text tokens and optional
    VAD probabilities in real time.
    """

    def __init__(
        self,
        hf_repo: str = DEFAULT_STT_HF_REPO,
        enable_vad: bool = True,
        max_steps: int = 4096,
    ):
        self._hf_repo = _HF_REPO_MAP.get(hf_repo, hf_repo)
        self._enable_vad = enable_vad
        self._max_steps = max_steps

        self._model = None
        self._gen = None
        self._audio_tokenizer = None
        self._text_tokenizer = None
        self._lm_config = None
        self._other_codebooks = 0
        self._frame_size_samples = 1920  # 80ms at 24kHz
        self._sample_rate_hz = 24000
        self._loaded = False
        self._vad_prob = 0.0
        self._end_of_turn = False

    async def load(self):
        if self._loaded:
            return

        def _load_sync():
            try:
                import mlx.core as mx
                import mlx.nn as nn
                import rustymimi
                import sentencepiece
                from huggingface_hub import hf_hub_download
                from moshi_mlx import models, utils
            except ImportError:
                raise ImportError(
                    "DSM STT requires moshi_mlx, rustymimi, and sentencepiece.\n"
                    "Note: moshi_mlx requires mlx<0.27 which conflicts with pocket-tts's mlx>=0.30.\n"
                    "Install in a separate environment or use: pip install 'moshi_mlx>=0.3.0' rustymimi\n"
                    "Alternatively, use --stt-backend=rust or --stt-backend=pytorch."
                ) from None

            logger.info("Loading DSM STT model from %s...", self._hf_repo)

            lm_config_path = hf_hub_download(self._hf_repo, "config.json")
            with open(lm_config_path) as f:
                raw_config = json.load(f)

            mimi_weights = hf_hub_download(self._hf_repo, raw_config["mimi_name"])
            moshi_name = raw_config.get("moshi_name", "model.safetensors")
            moshi_weights = hf_hub_download(self._hf_repo, moshi_name)
            tokenizer_path = hf_hub_download(self._hf_repo, raw_config["tokenizer_name"])

            lm_config = models.LmConfig.from_config_dict(raw_config)
            model = models.Lm(lm_config)
            model.set_dtype(mx.bfloat16)

            is_candle_repo = self._hf_repo.endswith("-candle")
            if moshi_weights.endswith(".q4.safetensors"):
                nn.quantize(model, bits=4, group_size=32)
            elif moshi_weights.endswith(".q8.safetensors"):
                nn.quantize(model, bits=8, group_size=64)

            logger.info("Loading STT LM weights from %s", moshi_weights)
            if is_candle_repo:
                model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
            else:
                model.load_weights(moshi_weights, strict=True)

            logger.info("Loading text tokenizer from %s", tokenizer_path)
            text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

            logger.info("Loading Mimi audio tokenizer from %s", mimi_weights)
            generated_codebooks = lm_config.generated_codebooks
            other_codebooks = lm_config.other_codebooks
            mimi_codebooks = max(generated_codebooks, other_codebooks)
            audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)

            logger.info("Warming up model...")
            model.warmup()

            gen = models.LmGen(
                model=model,
                max_steps=self._max_steps,
                text_sampler=utils.Sampler(top_k=25, temp=0),
                audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                check=False,
            )

            self._model = model
            self._gen = gen
            self._audio_tokenizer = audio_tokenizer
            self._text_tokenizer = text_tokenizer
            self._lm_config = lm_config
            self._other_codebooks = other_codebooks
            self._loaded = True

            logger.info(
                "DSM STT loaded: %s, other_codebooks=%d, vad=%s",
                self._hf_repo,
                other_codebooks,
                self._enable_vad,
            )

        await asyncio.get_running_loop().run_in_executor(None, _load_sync)

    async def transcribe_stream(
        self, audio_frames: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        """Transcribe streaming audio frames into text chunks.

        Also updates internal VAD probability if semantic VAD is enabled.
        """
        if not self._loaded:
            await self.load()

        import mlx.core as mx

        buffer = np.array([], dtype=np.float32)

        async for frame in audio_frames:
            buffer = np.concatenate([buffer, frame])

            while len(buffer) >= self._frame_size_samples:
                chunk = buffer[: self._frame_size_samples]
                buffer = buffer[self._frame_size_samples :]

                text = await asyncio.get_running_loop().run_in_executor(
                    None, self._process_frame, chunk
                )
                if text:
                    yield text

    def _process_frame(self, chunk: np.ndarray) -> str | None:
        """Process one 80ms audio frame, return text if produced."""
        import mlx.core as mx

        audio_data = chunk[np.newaxis, :, np.newaxis]  # (1, samples, 1)
        other_audio_tokens = self._audio_tokenizer.encode_step(audio_data)
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
            :, :, : self._other_codebooks
        ]

        text_parts = []

        if self._enable_vad:
            text_token, vad_heads = self._gen.step_with_extra_heads(other_audio_tokens[0])
            if vad_heads and len(vad_heads) > 2:
                self._vad_prob = vad_heads[2][0, 0, 0].item()
                if self._vad_prob > 0.5:
                    self._end_of_turn = True
        else:
            text_token = self._gen.step(other_audio_tokens[0])

        text_token_val = text_token[0].item()
        if text_token_val not in (0, 3):
            piece = self._text_tokenizer.id_to_piece(text_token_val)
            piece = piece.replace("\u2581", " ")
            text_parts.append(piece)

        return "".join(text_parts) if text_parts else None

    async def flush_remaining(self) -> str | None:
        """Feed silence frames to extract any buffered text from the model's audio delay."""
        if not self._loaded:
            return None

        def _flush_sync():
            import mlx.core as mx

            silence = np.zeros((1, self._frame_size_samples, 1), dtype=np.float32)
            text_parts = []
            for _ in range(8):
                other_audio_tokens = self._audio_tokenizer.encode_step(silence)
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, : self._other_codebooks
                ]
                if self._enable_vad:
                    text_token, _ = self._gen.step_with_extra_heads(other_audio_tokens[0])
                else:
                    text_token = self._gen.step(other_audio_tokens[0])

                text_token_val = text_token[0].item()
                if text_token_val not in (0, 3):
                    piece = self._text_tokenizer.id_to_piece(text_token_val)
                    piece = piece.replace("\u2581", " ")
                    text_parts.append(piece)

            return "".join(text_parts) if text_parts else None

        return await asyncio.get_running_loop().run_in_executor(None, _flush_sync)

    def reset(self):
        """Reset streaming state for a new utterance."""
        if self._gen is not None:
            self._gen = None
            try:
                from moshi_mlx import models, utils

                self._gen = models.LmGen(
                    model=self._model,
                    max_steps=self._max_steps,
                    text_sampler=utils.Sampler(top_k=25, temp=0),
                    audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                    check=False,
                )
            except Exception:
                logger.warning("Failed to reset LmGen, recreating on next load")
        self._vad_prob = 0.0
        self._end_of_turn = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_hz

    @property
    def frame_size(self) -> int:
        return self._frame_size_samples

    @property
    def vad_probability(self) -> float:
        """Current semantic VAD probability (0.0–1.0). >0.5 = end of turn."""
        return self._vad_prob

    @property
    def end_of_turn_detected(self) -> bool:
        """Whether the semantic VAD has detected an end-of-turn."""
        return self._end_of_turn

    def clear_end_of_turn(self):
        """Reset the end-of-turn flag after handling it."""
        self._end_of_turn = False
        self._vad_prob = 0.0
