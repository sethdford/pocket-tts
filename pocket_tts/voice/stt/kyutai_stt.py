"""Kyutai STT backend using the moshi package.

Wraps Kyutai's streaming speech-to-text model for real-time transcription
on Apple Silicon. Uses the PyTorch-based `moshi` package with the
`LMGen` streaming interface.

Models:
  - kyutai/stt-1b-en_fr: 1B params, English+French, fast, has semantic VAD
  - kyutai/stt-2.6b-en: 2.6B params, English only, more accurate
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from .base import STTBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "kyutai/stt-1b-en_fr"


class KyutaiSTT(STTBackend):
    """Streaming STT using Kyutai's moshi LMGen.

    The model runs on the local device (CPU or MPS). Audio is encoded
    with Mimi (24kHz, 12.5Hz frame rate) and fed to the language model
    which produces text tokens in real time.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self._model_name = model_name
        self._device = device
        self._mimi = None
        self._lm_gen = None
        self._text_tokenizer = None
        self._frame_size_samples = 0
        self._sample_rate_hz = 24000
        self._padding_token_id = 3
        self._loaded = False

    async def load(self):
        if self._loaded:
            return

        def _load_sync():
            try:
                import torch
                from moshi.models import LMGen, loaders
            except ImportError:
                raise ImportError(
                    "Kyutai STT requires the 'moshi' package. Install with: pip install moshi"
                ) from None

            device = self._device
            if device is None:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            self._device = device

            logger.info("Loading Kyutai STT model %s on %s...", self._model_name, device)

            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(self._model_name)
            self._mimi = checkpoint_info.get_mimi(device=device)
            self._frame_size_samples = int(self._mimi.sample_rate / self._mimi.frame_rate)
            self._sample_rate_hz = int(self._mimi.sample_rate)

            moshi_model = checkpoint_info.get_moshi(device=device)
            self._lm_gen = LMGen(moshi_model, temp=0, temp_text=0)

            self._mimi.streaming_forever(1)
            self._lm_gen.streaming_forever(1)

            self._text_tokenizer = checkpoint_info.get_text_tokenizer()

            self._padding_token_id = checkpoint_info.raw_config.get("text_padding_token_id", 3)

            # Warmup
            import torch as _torch

            for _ in range(4):
                codes = self._mimi.encode(_torch.zeros(1, 1, self._frame_size_samples).to(device))
                for c in range(codes.shape[-1]):
                    self._lm_gen.step(codes[:, :, c : c + 1])

            logger.info(
                "Kyutai STT loaded: %s, frame_size=%d, sample_rate=%d",
                self._model_name,
                self._frame_size_samples,
                self._sample_rate_hz,
            )

        await asyncio.get_running_loop().run_in_executor(None, _load_sync)
        self._loaded = True

    async def transcribe_stream(
        self, audio_frames: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        if not self._loaded:
            await self.load()

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
        """Process one frame synchronously (runs in executor)."""
        import torch

        tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device=self._device)

        with torch.no_grad():
            codes = self._mimi.encode(tensor)
            text_parts = []

            for c in range(codes.shape[-1]):
                result = self._lm_gen.step_with_extra_heads(codes[:, :, c : c + 1])
                if result is None:
                    continue

                text_tokens = result[0] if isinstance(result, tuple) else result
                if text_tokens is None:
                    continue

                text_token = text_tokens[0, 0, 0].item()
                if text_token not in (0, self._padding_token_id):
                    piece = self._text_tokenizer.id_to_piece(text_token)
                    piece = piece.replace("\u2581", " ")
                    text_parts.append(piece)

            return "".join(text_parts) if text_parts else None

    def reset(self):
        """Reset streaming state for a new utterance."""
        if self._mimi is not None:
            self._mimi.reset_streaming()
        if self._lm_gen is not None:
            self._lm_gen.reset_streaming()
            self._mimi.streaming_forever(1)
            self._lm_gen.streaming_forever(1)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_hz

    @property
    def frame_size(self) -> int:
        return self._frame_size_samples
