"""Configuration for the voice pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    """Configuration for the real-time voice pipeline."""

    sample_rate: int = Field(default=48000, description="CoreAudio sample rate (Hz)")
    buffer_frames: int = Field(
        default=256, description="CoreAudio buffer size in frames (lower = less latency, more CPU)"
    )
    tts_sample_rate: int = Field(default=24000, description="TTS native sample rate")

    voice: str = Field(default="tara/narration", description="TTS voice name")
    stt_model: str = Field(default="kyutai/stt-1b-en_fr", description="Kyutai STT model name")
    stt_backend: str = Field(
        default="auto",
        description=(
            "STT backend: 'dsm' (MLX + semantic VAD), 'rust' (candle Metal), "
            "'pytorch', or 'auto' (prefer dsm > rust > pytorch)"
        ),
    )
    stt_model_path: str = Field(
        default="model.safetensors",
        description="Model filename within HF repo (for Rust backend)",
    )

    tts_backend: str = Field(
        default="dsm",
        description="TTS backend: 'dsm' (1.6B, high quality) or 'pocket' (100M, fast)",
    )
    tts_quantize: int | None = Field(
        default=8,
        description="DSM TTS quantization bits (4 or 8). None for full precision. Ignored for pocket.",
    )
    tts_streaming: bool = Field(
        default=True,
        description=(
            "Enable streaming text input for DSM TTS (eliminates sentence buffer latency). "
            "Only effective with tts_backend='dsm'."
        ),
    )

    vad_energy_threshold: float = Field(
        default=0.01, description="VAD RMS energy threshold for speech onset (C engine)"
    )
    vad_silence_threshold: float = Field(
        default=0.005, description="VAD RMS threshold for speech offset (C engine)"
    )
    vad_enabled: bool = Field(
        default=False,
        description=(
            "Hands-free mode. With DSM STT, uses semantic VAD (model-based end-of-turn). "
            "With other STT backends, uses energy VAD from C engine. False = push-to-talk."
        ),
    )

    sentence_buffer_mode: str = Field(
        default="speculative", description="Sentence flush mode: 'sentence' or 'speculative'"
    )
    min_words_speculative: int = Field(
        default=5, description="Minimum words before clause-boundary flush in speculative mode"
    )

    workspace: str | None = Field(
        default=None, description="Workspace path for Claude Code (default: cwd)"
    )
    claude_model: str | None = Field(
        default=None, description="Claude model override (e.g. claude-sonnet-4-5)"
    )
    system_prompt: str | None = Field(
        default=None, description="Override the default voice system prompt"
    )


VOICE_SYSTEM_PROMPT = """\
You are a voice coding assistant powered by pocket-tts. The user is speaking \
to you through a microphone and hearing your responses through text-to-speech. \
Keep your responses concise and conversational — the user cannot see long text \
output. When running commands or editing files, briefly describe what you're \
doing. Avoid code blocks in your spoken responses unless the user asks to see \
code. If you need to show code, use the voice_speak tool to narrate key parts. \
When you execute tools (Read, Write, Bash, etc.), the terminal will show the \
details — just explain the intent and result verbally.\
"""
