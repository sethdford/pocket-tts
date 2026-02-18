import io
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from queue import Queue

import numpy as np
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.tts_model import TTSModel, export_model_state
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None
global_model_state = None

web_app = FastAPI(
    title="Kyutai Pocket TTS API",
    description=(
        "Text-to-Speech generation API powered by MLX on Apple Silicon. "
        "Supports plain text and SSML input, multiple voices, streaming audio."
    ),
    version="1.1.0",
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": tts_model is not None}


@web_app.get("/voices")
async def list_voices():
    """List available predefined voices.

    Returns a list of voice objects with name and URL.
    """
    voices = []
    for name, url in PREDEFINED_VOICES.items():
        voices.append({"name": name, "url": url})
    return {"voices": voices}


def write_to_queue(queue, text_to_generate, model_state):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def generate_data_with_state(text_to_generate: str, model_state: dict):
    queue = Queue()

    thread = threading.Thread(target=write_to_queue, args=(queue, text_to_generate, model_state))
    thread.start()

    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


def _resolve_voice_state(voice_url: str | None = None, voice_wav: UploadFile | None = None) -> dict:
    """Resolve voice state from URL, uploaded file, or default."""
    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "voice_url must be a predefined voice name "
                    f"({list(PREDEFINED_VOICES.keys())}), or start with http://, https://, hf://"
                ),
            )
        model_state = tts_model._cached_get_state_for_audio_prompt(voice_url)
        logger.info("Using voice from URL: %s", voice_url)
        return model_state

    elif voice_wav is not None:
        suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name

        try:
            model_state = tts_model.get_state_for_audio_prompt(Path(temp_file_path), truncate=True)
        finally:
            os.unlink(temp_file_path)
        return model_state

    return global_model_state


@web_app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "pocket-tts", "object": "model", "created": 1700000000, "owned_by": "kyutai"}
        ],
    }


@web_app.post("/v1/audio/speech")
def openai_speech(request: dict):
    """OpenAI-compatible text-to-speech endpoint.

    Drop-in replacement for OpenAI's ``/v1/audio/speech`` API. Any client
    using the ``openai`` Python package can point to this server with zero
    code changes::

        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        response = client.audio.speech.create(
            model="pocket-tts",
            voice="alba",
            input="Hello world!",
        )
        response.stream_to_file("output.wav")

    Request body (JSON):
        model (str): Model name (ignored; always uses pocket-tts).
        input (str): The text to generate audio for. Supports plain text and SSML.
        voice (str): Voice to use. One of the predefined voice names or a URL.
        response_format (str): Audio format. Currently only "wav" is supported.
        speed (float): Speech speed multiplier (reserved for future use).

    Returns:
        Streaming WAV audio with chunked transfer encoding.
    """
    text = request.get("input", "")
    voice = request.get("voice", None)
    response_format = request.get("response_format", "wav")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="input text cannot be empty")

    if response_format not in ("wav", "pcm"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Use 'wav' or 'pcm'.",
        )

    # Resolve voice state
    if voice and voice in PREDEFINED_VOICES:
        model_state = tts_model._cached_get_state_for_audio_prompt(voice)
    elif voice:
        try:
            model_state = tts_model._cached_get_state_for_audio_prompt(voice)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid voice: {e}")
    else:
        model_state = global_model_state

    media = "audio/wav" if response_format == "wav" else "audio/pcm"
    return StreamingResponse(
        generate_data_with_state(text, model_state),
        media_type=media,
        headers={
            "Content-Disposition": "attachment; filename=speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(..., description="Text to convert to speech. Supports plain text and SSML."),
    voice_url: str | None = Form(
        None, description="Voice URL (http://, https://, hf://) or predefined voice name"
    ),
    voice_wav: UploadFile | None = File(
        None, description="Uploaded voice WAV file (mutually exclusive with voice_url)"
    ),
):
    """Generate speech from text.

    Accepts plain text or SSML (starting with `<speak>`). Returns streaming
    WAV audio with chunked transfer encoding.

    Supports voice selection via predefined names, HuggingFace URLs, or uploaded WAV files.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    model_state = _resolve_voice_state(voice_url, voice_wav)

    return StreamingResponse(
        generate_data_with_state(text, model_state),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@web_app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS streaming.

    Protocol:
    1. Client sends JSON: {"text": "...", "voice": "alba"} (voice is optional)
    2. Server streams back binary PCM audio chunks (int16, mono, at model sample rate)
    3. Server sends JSON: {"done": true, "sample_rate": 24000} when generation is complete

    Client can send multiple requests on the same connection.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            text = request.get("text", "")
            voice = request.get("voice")

            if not text.strip():
                await websocket.send_json({"error": "Text cannot be empty"})
                continue

            # Resolve voice
            if voice and voice in PREDEFINED_VOICES:
                model_state = tts_model._cached_get_state_for_audio_prompt(voice)
            elif voice:
                try:
                    model_state = tts_model._cached_get_state_for_audio_prompt(voice)
                except Exception as e:
                    await websocket.send_json({"error": f"Invalid voice: {e}"})
                    continue
            else:
                model_state = global_model_state

            sr = tts_model.config.mimi.sample_rate
            total_samples = 0

            # Stream audio chunks as binary PCM
            from pocket_tts.native import pcm_convert

            for chunk in tts_model.generate_audio_stream(
                model_state=model_state, text_to_generate=text
            ):
                # NEON SIMD PCM conversion (single-pass, zero intermediate allocs)
                await websocket.send_bytes(pcm_convert(chunk))
                total_samples += len(chunk)

            # Signal completion
            duration_ms = int(total_samples * 1000 / sr)
            await websocket.send_json(
                {
                    "done": True,
                    "sample_rate": sr,
                    "total_samples": total_samples,
                    "duration_ms": duration_ms,
                }
            )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    config: Annotated[
        str,
        typer.Option(
            help="Path to locally-saved model config .yaml file or model variant signature"
        ),
    ] = DEFAULT_VARIANT,
    quantize: Annotated[
        int | None, typer.Option(help="Quantize model to 4 or 8 bits to reduce memory usage")
    ] = None,
    mimi_dtype: Annotated[
        str | None,
        typer.Option(help="Mimi decoder dtype: 'bfloat16' or 'float16' for lower latency"),
    ] = None,
    preload_voices: Annotated[
        bool, typer.Option(help="Pre-load all predefined voices into memory for instant switching")
    ] = True,
):
    """Start the FastAPI server."""

    global tts_model, global_model_state
    tts_model = TTSModel.load_model(config, quantize=quantize, mimi_dtype=mimi_dtype)

    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    logger.info(f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB")

    # Pre-load all predefined voices into the LRU cache for instant switching.
    # This eliminates 500ms+ latency from HuggingFace cache lookups and
    # safetensors deserialization on first use of each voice during serving.
    if preload_voices:
        logger.info("Pre-loading %d predefined voices...", len(PREDEFINED_VOICES))
        for voice_name in PREDEFINED_VOICES:
            try:
                tts_model._cached_get_state_for_audio_prompt(voice_name)
                logger.info("  Loaded voice: %s", voice_name)
            except Exception as e:
                logger.warning("  Failed to load voice '%s': %s", voice_name, e)
        logger.info("Voice pre-loading complete.")

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    config: Annotated[
        str, typer.Option(help="Model signature or path to config .yaml file")
    ] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
    quantize: Annotated[
        int | None, typer.Option(help="Quantize model to 4 or 8 bits to reduce memory usage")
    ] = None,
    mimi_dtype: Annotated[
        str | None,
        typer.Option(help="Mimi decoder dtype: 'bfloat16' or 'float16' for lower latency"),
    ] = None,
    speculative_tokens: Annotated[
        int | None,
        typer.Option(help="Enable speculative decoding with N draft tokens per round (e.g., 4)"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(help="Random seed for reproducible generation"),
    ] = None,
):
    """Generate speech using Kyutai Pocket TTS."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text == "-":
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            config,
            temperature,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
            quantize=quantize,
            mimi_dtype=mimi_dtype,
        )

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
            speculative_tokens=speculative_tokens,
            seed=seed,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    config: Annotated[str, typer.Option(help="Model config path or signature")] = DEFAULT_VARIANT,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


# ------------------------------------------------------
# The pocket-tts voice pipeline CLI implementation
# ------------------------------------------------------

voice_app = typer.Typer(help="Real-time voice AI pipeline")
cli_app.add_typer(voice_app, name="voice")


@voice_app.command()
def claude(
    workspace: Annotated[
        str | None, typer.Option(help="Workspace directory for Claude Code (default: cwd)")
    ] = None,
    voice: Annotated[str, typer.Option(help="TTS voice name")] = DEFAULT_AUDIO_PROMPT,
    stt_model: Annotated[
        str, typer.Option(help="Kyutai STT model: kyutai/stt-1b-en_fr or kyutai/stt-2.6b-en")
    ] = "kyutai/stt-1b-en_fr",
    stt_backend: Annotated[
        str,
        typer.Option(
            help=(
                "STT backend: 'auto' (prefer dsm MLX), 'dsm' (MLX + semantic VAD), "
                "'rust' (candle Metal), 'pytorch'"
            )
        ),
    ] = "auto",
    tts_backend: Annotated[
        str,
        typer.Option(
            help="TTS backend: 'dsm' (1.6B, high quality) or 'pocket' (100M, fast, low memory)"
        ),
    ] = "dsm",
    tts_quantize: Annotated[
        int | None,
        typer.Option(
            help="DSM TTS quantization bits (4 or 8). None for full precision. Ignored for pocket."
        ),
    ] = 8,
    no_tts_streaming: Annotated[
        bool,
        typer.Option(
            "--no-tts-streaming",
            help="Disable streaming text input for DSM TTS (use sentence buffer instead)",
        ),
    ] = False,
    vad: Annotated[
        bool,
        typer.Option(
            "--vad",
            help="Hands-free mode. Uses semantic VAD with DSM STT, energy VAD otherwise.",
        ),
    ] = False,
    sample_rate: Annotated[
        int, typer.Option(help="CoreAudio sample rate (48000 for native, 24000 for direct)")
    ] = 48000,
    buffer_frames: Annotated[
        int, typer.Option(help="CoreAudio buffer frames (lower = less latency, more CPU)")
    ] = 256,
    sentence_mode: Annotated[
        str, typer.Option(help="Sentence flush mode: 'speculative' (low latency) or 'sentence'")
    ] = "speculative",
    claude_model: Annotated[
        str | None, typer.Option(help="Claude model override (e.g. claude-sonnet-4-5)")
    ] = None,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Suppress non-essential output")
    ] = False,
):
    """Start a voice conversation with Claude Code.

    Speak through your microphone, hear Claude's responses through your speakers.
    Claude has full access to your workspace (Read, Write, Bash, etc.).
    Uses CoreAudio VoiceProcessingIO for ultra-low-latency audio with echo cancellation.

    Default mode is push-to-talk (hold spacebar). Use --vad for hands-free.

    TTS backends:
      dsm (default): Kyutai DSM TTS 1.6B -- high quality, streaming text input, ~1.8GB memory
      pocket: Original pocket-tts 100M -- fast, low memory (~400MB)

    STT backends:
      auto (default): Prefer DSM MLX > Rust candle > PyTorch
      dsm: MLX-native with built-in semantic VAD (recommended with --vad)
      rust: Rust/candle/Metal FFI
      pytorch: PyTorch moshi (fallback)
    """
    import asyncio

    from pocket_tts.voice.config import VoiceConfig

    log_level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config = VoiceConfig(
        workspace=workspace,
        voice=voice,
        stt_model=stt_model,
        stt_backend=stt_backend,
        tts_backend=tts_backend,
        tts_quantize=tts_quantize,
        tts_streaming=not no_tts_streaming,
        vad_enabled=vad,
        sample_rate=sample_rate,
        buffer_frames=buffer_frames,
        sentence_buffer_mode=sentence_mode,
        claude_model=claude_model,
    )

    from pocket_tts.voice.claude_voice import voice_loop

    asyncio.run(voice_loop(config))


if __name__ == "__main__":
    cli_app()
