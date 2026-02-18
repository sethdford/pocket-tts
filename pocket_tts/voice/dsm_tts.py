"""DSM TTS 1.6B backend using moshi_mlx for high-quality streaming text-to-speech.

Wraps Kyutai's Delayed Streams Modeling TTS model (1.8B params: 1B backbone +
600M depth transformer) for MLX inference on Apple Silicon. Supports:
  - Streaming audio output via on_frame callback
  - 4-bit and 8-bit quantization for speed vs quality tradeoff
  - Voice conditioning via pre-computed embeddings (safetensors)
  - CFG distillation (no inference-time overhead)
  - Streaming text input via StreamingTTSGen for ultra-low latency

The model uses the same Mimi codec (12.5 Hz, 24 kHz) as pocket-tts, making it
a drop-in replacement for the voice pipeline.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DSM_TTS_HF_REPO = "kyutai/tts-1.6b-en_fr"
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"
DEFAULT_DSM_N_Q = 24
DEFAULT_DSM_TEMP = 0.6
DEFAULT_DSM_CFG_COEF = 2.0
DEFAULT_DSM_PADDING_BETWEEN = 1


class DsmTTSBackend:
    """High-quality TTS using Kyutai's DSM TTS 1.6B model on MLX.

    Provides the same interface as the pocket-tts TTSModel for use in the
    voice pipeline: load(), generate_audio_stream(), and voice conditioning.
    """

    def __init__(
        self,
        hf_repo: str = DEFAULT_DSM_TTS_HF_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        quantize: int | None = 8,
        n_q: int = DEFAULT_DSM_N_Q,
        temp: float = DEFAULT_DSM_TEMP,
        cfg_coef: float = DEFAULT_DSM_CFG_COEF,
        padding_between: int = DEFAULT_DSM_PADDING_BETWEEN,
    ):
        self._hf_repo = hf_repo
        self._voice_repo = voice_repo
        self._quantize = quantize
        self._n_q = n_q
        self._temp = temp
        self._cfg_coef = cfg_coef
        self._padding_between = padding_between

        self._tts_model = None
        self._raw_config = None
        self._loaded = False
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self):
        """Load the DSM TTS model, Mimi codec, and text tokenizer."""
        if self._loaded:
            return

        try:
            import mlx.core as mx
            import mlx.nn as nn
            import sentencepiece
            from moshi_mlx import models
            from moshi_mlx.models.tts import TTSModel
            from moshi_mlx.utils.loaders import hf_get
        except ImportError:
            raise ImportError(
                "DSM TTS requires moshi_mlx and sentencepiece.\n"
                "Note: moshi_mlx requires mlx<0.27 which conflicts with pocket-tts's mlx>=0.30.\n"
                "Install in a separate environment or use: pip install 'moshi_mlx>=0.3.0'\n"
                "Alternatively, use --tts-backend=pocket to use the built-in 100M model."
            ) from None

        logger.info("Loading DSM TTS model from %s...", self._hf_repo)

        raw_config_path = hf_get("config.json", self._hf_repo)
        with open(hf_get(raw_config_path)) as f:
            self._raw_config = json.load(f)

        mimi_weights = hf_get(self._raw_config["mimi_name"], self._hf_repo)
        moshi_name = self._raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, self._hf_repo)
        tokenizer_path = hf_get(self._raw_config["tokenizer_name"], self._hf_repo)

        lm_config = models.LmConfig.from_config_dict(self._raw_config)
        # Workaround for ring KV cache bug in moshi_mlx <= 0.3.0
        lm_config.transformer.max_seq_len = lm_config.transformer.context

        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        logger.info("Loading LM weights from %s", moshi_weights)
        model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if self._quantize is not None:
            logger.info("Quantizing model to %d bits", self._quantize)
            nn.quantize(model.depformer, bits=self._quantize)
            for layer in model.transformer.layers:
                nn.quantize(layer.self_attn, bits=self._quantize)
                nn.quantize(layer.gating, bits=self._quantize)

        logger.info("Loading text tokenizer from %s", tokenizer_path)
        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

        logger.info("Loading Mimi audio tokenizer from %s", mimi_weights)
        generated_codebooks = lm_config.generated_codebooks
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        cfg_coef_conditioning = None
        tts_model = TTSModel(
            model,
            audio_tokenizer,
            text_tokenizer,
            voice_repo=self._voice_repo,
            temp=self._temp,
            cfg_coef=1,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=self._raw_config,
        )

        if tts_model.valid_cfg_conditionings:
            cfg_coef_conditioning = self._cfg_coef
            tts_model.cfg_coef = 1.0
            self._cfg_is_no_text = False
            self._cfg_is_no_prefix = False
        else:
            self._cfg_is_no_text = True
            self._cfg_is_no_prefix = True

        self._cfg_coef_conditioning = cfg_coef_conditioning
        self._tts_model = tts_model
        self._sample_rate = tts_model.mimi.sample_rate
        self._loaded = True

        logger.info(
            "DSM TTS loaded: %s, sample_rate=%d, n_q=%d, quantize=%s",
            self._hf_repo,
            self._sample_rate,
            self._n_q,
            self._quantize,
        )

    def get_voice_path(self, voice: str) -> str:
        """Resolve a voice name to a safetensors file path."""
        if not self._loaded:
            self.load()
        if voice.endswith(".safetensors"):
            return voice
        return str(self._tts_model.get_voice_path(voice))

    def make_condition_attributes(self, voice_paths: list[str]):
        """Create voice conditioning attributes for generation."""
        if not self._loaded:
            self.load()
        return self._tts_model.make_condition_attributes(
            voice_paths, self._cfg_coef_conditioning
        )

    def generate_audio_stream(
        self,
        text: str,
        voice: str | None = None,
        condition_attributes=None,
    ) -> Iterator[np.ndarray]:
        """Generate speech audio from text, yielding chunks as they are produced.

        Args:
            text: Text to synthesize.
            voice: Voice name or path. Used if condition_attributes is None.
            condition_attributes: Pre-computed voice conditioning. If None,
                uses ``voice`` to create one.

        Yields:
            float32 numpy arrays of audio samples at ``self.sample_rate``.
        """
        import mlx.core as mx

        if not self._loaded:
            self.load()

        if condition_attributes is None:
            if voice is None:
                raise ValueError("Either voice or condition_attributes must be provided")
            voice_path = self.get_voice_path(voice)
            condition_attributes = self.make_condition_attributes([voice_path])

        entries = [self._tts_model.prepare_script([text])]

        pcm_queue: queue.Queue[np.ndarray | None] = queue.Queue()

        def _on_frame(frame):
            if (frame == -1).any():
                return
            pcm = self._tts_model.mimi.decode_step(frame[:, :, None])
            pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
            pcm_queue.put_nowait(pcm)

        def _generate():
            try:
                self._tts_model.generate(
                    entries,
                    [condition_attributes],
                    cfg_is_no_prefix=self._cfg_is_no_prefix,
                    cfg_is_no_text=self._cfg_is_no_text,
                    on_frame=_on_frame,
                )
            finally:
                pcm_queue.put(None)

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()

        while True:
            pcm = pcm_queue.get()
            if pcm is None:
                break
            yield pcm

        gen_thread.join()

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        condition_attributes=None,
    ) -> np.ndarray:
        """Generate speech audio from text, returning a single concatenated array."""
        chunks = list(self.generate_audio_stream(text, voice, condition_attributes))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)


class StreamingTTSGen:
    """Streaming text-to-audio generator for DSM TTS.

    Ports the TTSGen pattern from tts_pytorch_streaming.py to work with
    the DsmTTSBackend. Text entries are appended incrementally as LLM
    tokens arrive; audio frames are generated as soon as enough text
    is available, eliminating the need for a sentence buffer.

    Usage:
        gen = StreamingTTSGen(backend, condition_attributes)
        # As LLM tokens arrive:
        gen.append_text("Hello, ")
        for pcm in gen.process():
            play(pcm)
        gen.append_text("how are you?")
        for pcm in gen.process():
            play(pcm)
        # After all text received:
        for pcm in gen.process_last():
            play(pcm)
    """

    def __init__(self, backend: DsmTTSBackend, condition_attributes, first_turn: bool = True):
        import mlx.core as mx
        from moshi_mlx.models.tts import TTSModel

        if not backend._loaded:
            backend.load()

        self._backend = backend
        self._tts_model: TTSModel = backend._tts_model
        self._mx = mx

        self._multi_speaker = first_turn and self._tts_model.multi_speaker

        if self._tts_model.valid_cfg_conditionings:
            cfg_is_no_text = False
            cfg_is_no_prefix = False
        else:
            cfg_is_no_text = True
            cfg_is_no_prefix = True

        self._cfg_is_no_text = cfg_is_no_text
        self._cfg_is_no_prefix = cfg_is_no_prefix
        self._condition_attributes = condition_attributes
        self._accumulated_text = ""
        self._generation_active = False
        self._mimi_streaming = None

    def append_text(self, text: str):
        """Buffer text from LLM tokens for synthesis."""
        self._accumulated_text += text

    def process(self) -> Iterator[np.ndarray]:
        """Generate audio for any buffered text that forms complete segments.

        Yields audio chunks as float32 numpy arrays at the backend's sample rate.
        Flushes text on sentence boundaries (. ! ? newline) to produce natural
        speech segments without waiting for all text.
        """
        import re

        sentence_end = re.search(r"[.!?]\s|[.!?]$|\n", self._accumulated_text)
        if not sentence_end:
            return

        end_pos = sentence_end.end()
        segment = self._accumulated_text[:end_pos].strip()
        self._accumulated_text = self._accumulated_text[end_pos:]

        if not segment:
            return

        yield from self._synthesize_segment(segment)

    def process_last(self) -> Iterator[np.ndarray]:
        """Synthesize any remaining buffered text (call at end of response)."""
        remaining = self._accumulated_text.strip()
        self._accumulated_text = ""
        if remaining:
            yield from self._synthesize_segment(remaining)

    def _synthesize_segment(self, text: str) -> Iterator[np.ndarray]:
        """Run TTS on a text segment and yield audio chunks."""
        yield from self._backend.generate_audio_stream(
            text=text,
            condition_attributes=self._condition_attributes,
        )

    def reset(self):
        """Reset for a new conversation turn."""
        self._accumulated_text = ""
        self._generation_active = False
