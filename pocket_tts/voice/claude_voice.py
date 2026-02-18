"""Claude Code voice orchestrator using the Claude Agent SDK.

Coordinates: Mic -> C Engine -> STT -> ClaudeSDKClient -> TTS -> Speaker

Supports two TTS backends:
  - "pocket": Original pocket-tts 100M (fast, low memory)
  - "dsm": DSM TTS 1.6B via moshi_mlx (high quality, streaming text input)

And three STT backends:
  - "dsm": DSM STT 1B via moshi_mlx with semantic VAD (recommended)
  - "rust": Rust/candle/Metal FFI backend
  - "pytorch": PyTorch moshi backend (fallback)
  - "auto": Prefer dsm > rust > pytorch

The C engine handles the entire real-time audio path (CoreAudio VoiceProcessingIO
with AEC, lock-free ring buffers, energy VAD). Python handles only async
orchestration: STT (MLX/Rust/PyTorch), Claude Agent SDK, and TTS (MLX).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from pocket_tts.voice.config import VOICE_SYSTEM_PROMPT, VoiceConfig
from pocket_tts.voice.native_audio import (
    VAD_SILENCE,
    VAD_SPEECH,
    VAD_SPEECH_END,
    VAD_SPEECH_START,
    NativeVoiceEngine,
)
from pocket_tts.voice.sentence_buffer import SentenceBuffer

logger = logging.getLogger(__name__)

_CANDLE_REPO_MAP = {
    "kyutai/stt-1b-en_fr": "kyutai/stt-1b-en_fr-candle",
    "kyutai/stt-2.6b-en": "kyutai/stt-2.6b-en-candle",
}


def _create_stt_backend(config: VoiceConfig):
    """Create the appropriate STT backend based on config.

    Priority for 'auto': dsm (MLX) > rust (candle Metal) > pytorch.
    The DSM backend is preferred because it provides semantic VAD and
    shares the moshi_mlx package with DSM TTS.
    """
    backend = config.stt_backend

    if backend == "auto":
        try:
            import moshi_mlx  # noqa: F401

            from pocket_tts.voice.stt.dsm_stt import DsmSTT

            backend = "dsm"
            logger.info("Auto-selected DSM STT backend (MLX + semantic VAD)")
        except ImportError:
            try:
                from pocket_tts.voice.stt.rust_stt import RustSTT, is_available

                if is_available():
                    backend = "rust"
                    logger.info("DSM STT unavailable, using Rust STT backend")
                else:
                    backend = "pytorch"
                    logger.info("Rust STT unavailable, falling back to PyTorch backend")
            except ImportError:
                backend = "pytorch"
                logger.info("Using PyTorch STT backend (fallback)")

    if backend == "dsm":
        from pocket_tts.voice.stt.dsm_stt import DsmSTT

        return DsmSTT(
            hf_repo=config.stt_model,
            enable_vad=True,
        )

    if backend == "rust":
        from pocket_tts.voice.stt.rust_stt import RustSTT

        hf_repo = _CANDLE_REPO_MAP.get(config.stt_model, config.stt_model)
        return RustSTT(
            hf_repo=hf_repo,
            model_path=config.stt_model_path,
            enable_vad=True,
        )

    from pocket_tts.voice.stt.kyutai_stt import KyutaiSTT

    return KyutaiSTT(model_name=config.stt_model)


def _create_tts_backend(config: VoiceConfig):
    """Create the appropriate TTS backend based on config.

    Returns a (tts, voice_state_or_attrs, is_dsm) tuple. The caller uses
    is_dsm to decide which generation path to take.
    """
    if config.tts_backend == "dsm":
        try:
            from pocket_tts.voice.dsm_tts import DsmTTSBackend

            backend = DsmTTSBackend(
                quantize=config.tts_quantize,
            )
            backend.load()
            voice_path = backend.get_voice_path(config.voice)
            condition_attrs = backend.make_condition_attributes([voice_path])
            return backend, condition_attrs, True
        except ImportError:
            logger.warning(
                "DSM TTS backend unavailable (moshi_mlx not installed). "
                "Falling back to pocket-tts."
            )

    from pocket_tts.models.tts_model import TTSModel

    tts = TTSModel.load_model()
    model_state = tts.get_state_for_audio_prompt(config.voice)
    return tts, model_state, False


def _write_audio_to_engine(
    engine: NativeVoiceEngine, audio: np.ndarray, target_sr: int
):
    """Resample if needed and write audio to the playback ring buffer."""
    audio = np.asarray(audio).flatten().astype(np.float32)
    if target_sr == 48000:
        audio = engine.resample_24_to_48(audio)
    engine.write_playback(audio)


async def _audio_frame_generator(
    engine: NativeVoiceEngine, stt_frame_size: int, sample_rate: int
) -> AsyncIterator[np.ndarray]:
    """Yield audio frames from the C engine's capture ring buffer."""
    buf = np.array([], dtype=np.float32)
    while True:
        raw = engine.read_capture(max_frames=4096)
        if len(raw) > 0:
            if sample_rate == 48000:
                raw = engine.resample_48_to_24(raw)
            buf = np.concatenate([buf, raw])
            while len(buf) >= stt_frame_size:
                yield buf[:stt_frame_size]
                buf = buf[stt_frame_size:]
        else:
            await asyncio.sleep(0.005)


async def _capture_speech_ptt(engine: NativeVoiceEngine, terminal_print: bool = True) -> np.ndarray:
    """Push-to-talk: capture audio while spacebar is held."""
    if terminal_print:
        print("\n  [Hold SPACE to speak, release to send] ", end="", flush=True)

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (ImportError, termios.error):
        old_settings = None
        fd = None

    try:
        while True:
            if fd is not None:
                ch = sys.stdin.read(1)
                if ch == " ":
                    break
            else:
                await asyncio.sleep(0.1)
                break

        if terminal_print:
            print("\r  [Recording...] ", end="", flush=True)

        frames = []
        engine.read_capture(max_frames=65536)

        while True:
            audio = engine.read_capture(max_frames=4096)
            if len(audio) > 0:
                frames.append(audio)

            if fd is not None:
                import select

                readable, _, _ = select.select([sys.stdin], [], [], 0)
                if readable:
                    ch = sys.stdin.read(1)
                    if ch != " ":
                        break
            else:
                await asyncio.sleep(0.01)
                if not frames or len(np.concatenate(frames)) > 24000 * 10:
                    break

            await asyncio.sleep(0.005)

        if terminal_print:
            print("\r  [Processing...] ", end="", flush=True)

    finally:
        if fd is not None and old_settings is not None:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames)


async def _capture_speech_vad(engine: NativeVoiceEngine) -> np.ndarray:
    """VAD mode: capture audio between SPEECH_START and SPEECH_END."""
    print("\n  [Listening...] ", end="", flush=True)

    while engine.get_vad_state() not in (VAD_SPEECH_START, VAD_SPEECH):
        engine.read_capture(max_frames=4096)
        await asyncio.sleep(0.01)

    print("\r  [Recording...] ", end="", flush=True)

    frames = []
    while True:
        audio = engine.read_capture(max_frames=4096)
        if len(audio) > 0:
            frames.append(audio)

        state = engine.get_vad_state()
        if state == VAD_SPEECH_END or state == VAD_SILENCE:
            if len(frames) > 0 and sum(len(f) for f in frames) > 1920:
                break

        await asyncio.sleep(0.005)

    print("\r  [Processing...] ", end="", flush=True)
    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames)


async def _capture_speech_semantic_vad(
    engine: NativeVoiceEngine, stt, config: VoiceConfig
) -> tuple[np.ndarray, list[str]]:
    """Capture speech using the DSM STT's semantic VAD for end-of-turn detection.

    Unlike energy VAD, semantic VAD understands when the user has finished a
    thought, handling pauses, filler words, and breathing gracefully. Returns
    both the raw audio and any text already transcribed during capture.

    Returns:
        Tuple of (raw_audio_array, list_of_transcribed_text_chunks).
    """
    from pocket_tts.voice.stt.dsm_stt import DsmSTT

    assert isinstance(stt, DsmSTT), "Semantic VAD requires DSM STT backend"

    print("\n  [Listening (semantic VAD)...] ", end="", flush=True)

    stt.reset()
    stt.clear_end_of_turn()
    frames = []
    transcribed = []
    recording = False
    silence_after_speech = 0

    engine.read_capture(max_frames=65536)

    while True:
        audio = engine.read_capture(max_frames=4096)
        if len(audio) == 0:
            await asyncio.sleep(0.005)
            continue

        frames.append(audio)

        if config.sample_rate == 48000:
            audio_24k = engine.resample_48_to_24(audio)
        else:
            audio_24k = audio

        frame_size = stt.frame_size or 1920
        buf = audio_24k
        while len(buf) >= frame_size:
            chunk = buf[:frame_size]
            buf = buf[frame_size:]

            text = await asyncio.get_running_loop().run_in_executor(
                None, stt._process_frame, chunk
            )
            if text:
                if not recording:
                    recording = True
                    print("\r  [Recording...] ", end="", flush=True)
                transcribed.append(text)
                silence_after_speech = 0

        if recording:
            if stt.end_of_turn_detected:
                print("\r  [End of turn detected] ", end="", flush=True)
                break
            silence_after_speech += len(audio)
            max_silence = int(config.sample_rate * 3.0)
            if silence_after_speech > max_silence:
                break

        total_samples = sum(len(f) for f in frames)
        if total_samples > config.sample_rate * 30:
            break

        await asyncio.sleep(0.001)

    print("\r  [Processing...] ", end="", flush=True)

    if hasattr(stt, "flush_remaining"):
        tail = await stt.flush_remaining()
        if tail:
            transcribed.append(tail)

    if not frames:
        return np.array([], dtype=np.float32), transcribed
    return np.concatenate(frames), transcribed


async def voice_loop(config: VoiceConfig):
    """Main voice loop: Mic -> STT -> Claude Code -> TTS -> Speaker.

    Uses ClaudeSDKClient for real-time bidirectional communication with
    Claude Code. The C voice engine handles all audio I/O with built-in
    AEC and VAD. Supports barge-in (interrupt Claude when user speaks).
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            create_sdk_mcp_server,
            tool,
        )
    except ImportError:
        raise ImportError(
            "Claude voice integration requires claude-agent-sdk. "
            "Install with: pip install claude-agent-sdk"
        ) from None

    workspace = Path(config.workspace or ".").resolve()

    # --- Initialize components ---
    print("Initializing voice pipeline...")

    engine = NativeVoiceEngine(sample_rate=config.sample_rate, buffer_frames=config.buffer_frames)

    stt = _create_stt_backend(config)
    await stt.load()

    tts, voice_state, is_dsm_tts = _create_tts_backend(config)

    use_streaming_tts = is_dsm_tts and config.tts_streaming
    use_semantic_vad = config.vad_enabled and _has_semantic_vad(stt)

    if use_streaming_tts:
        from pocket_tts.voice.dsm_tts import StreamingTTSGen

    sentence_buffer = SentenceBuffer(
        mode=config.sentence_buffer_mode, min_words_speculative=config.min_words_speculative
    )

    # --- Custom voice tools for Claude ---
    @tool("voice_speak", "Speak text aloud through the speaker", {"text": str})
    async def voice_speak_tool(args):
        text = args["text"]
        if is_dsm_tts:
            for chunk in tts.generate_audio_stream(text, condition_attributes=voice_state):
                _write_audio_to_engine(engine, chunk, config.sample_rate)
        else:
            for chunk in tts.generate_audio_stream(voice_state, text):
                _write_audio_to_engine(engine, chunk, config.sample_rate)
        return {"content": [{"type": "text", "text": f"Spoke: {text[:100]}..."}]}

    @tool("voice_status", "Check if the user is currently speaking", {})
    async def voice_status_tool(args):
        if use_semantic_vad:
            from pocket_tts.voice.stt.dsm_stt import DsmSTT

            if isinstance(stt, DsmSTT):
                vad_p = stt.vad_probability
                state_name = "end_of_turn" if stt.end_of_turn_detected else (
                    "speaking" if vad_p < 0.3 else "pausing"
                )
                return {
                    "content": [{
                        "type": "text",
                        "text": f"User state: {state_name} (vad_prob={vad_p:.2f})",
                    }]
                }
        state = engine.get_vad_state()
        state_name = {
            VAD_SILENCE: "silence",
            VAD_SPEECH_START: "speech_start",
            VAD_SPEECH: "speaking",
            VAD_SPEECH_END: "speech_end",
        }.get(state, "unknown")
        return {"content": [{"type": "text", "text": f"User mic state: {state_name}"}]}

    voice_tools = create_sdk_mcp_server(
        name="voice", version="1.0.0", tools=[voice_speak_tool, voice_status_tool]
    )

    system_prompt = config.system_prompt or VOICE_SYSTEM_PROMPT

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=[
            "Read",
            "Write",
            "Bash",
            "Grep",
            "Glob",
            "mcp__voice__voice_speak",
            "mcp__voice__voice_status",
        ],
        cwd=str(workspace),
        mcp_servers={"voice": voice_tools},
        permission_mode="acceptEdits",
    )

    if config.claude_model:
        options = ClaudeAgentOptions(
            **{**options.__dict__, "env": {"ANTHROPIC_MODEL": config.claude_model}}
        )

    # --- Start audio engine ---
    engine.start()
    if config.vad_enabled and not use_semantic_vad:
        engine.set_vad_thresholds(config.vad_energy_threshold, config.vad_silence_threshold)

    stt_backend_name = type(stt).__name__
    stt_label = f"{stt_backend_name} ({config.stt_model})"
    tts_label = f"DSM TTS 1.6B (q{config.tts_quantize})" if is_dsm_tts else "pocket-tts 100M"
    vad_label = "semantic (DSM)" if use_semantic_vad else (
        "energy (C engine)" if config.vad_enabled else "push-to-talk"
    )

    print("\nVoice pipeline ready!")
    print(f"  Audio: {config.sample_rate}Hz, {config.buffer_frames}-frame buffer")
    print(f"  STT: {stt_label}")
    print(f"  TTS: {tts_label}")
    print(f"  TTS voice: {config.voice}")
    print(f"  Workspace: {workspace}")
    print(f"  Input mode: {vad_label}")
    if use_streaming_tts:
        print("  TTS mode: streaming (no sentence buffer)")
    else:
        print(f"  Sentence buffer: {config.sentence_buffer_mode}")
    print()

    try:
        async with ClaudeSDKClient(options=options) as client:
            turn_count = 0
            while True:
                turn_count += 1

                # --- 1. Capture speech ---
                pre_transcribed = []
                if use_semantic_vad:
                    raw_audio, pre_transcribed = await _capture_speech_semantic_vad(
                        engine, stt, config
                    )
                elif config.vad_enabled:
                    raw_audio = await _capture_speech_vad(engine)
                else:
                    raw_audio = await _capture_speech_ptt(engine)

                if len(raw_audio) < 1920 and not pre_transcribed:
                    continue

                # --- 2. Resample + STT ---
                t0 = time.perf_counter()

                if pre_transcribed:
                    text = "".join(pre_transcribed).strip()
                else:
                    if config.sample_rate == 48000:
                        audio_24k = engine.resample_48_to_24(raw_audio)
                    else:
                        audio_24k = raw_audio

                    transcribed = []
                    stt.reset()

                    async def _single_frame_gen():
                        frame_size = stt.frame_size or 1920
                        i = 0
                        while i < len(audio_24k):
                            yield audio_24k[i : i + frame_size]
                            i += frame_size

                    async for text_chunk in stt.transcribe_stream(_single_frame_gen()):
                        transcribed.append(text_chunk)

                    if hasattr(stt, "flush_remaining"):
                        tail = await stt.flush_remaining()
                        if tail:
                            transcribed.append(tail)

                    text = "".join(transcribed).strip()

                stt_time = time.perf_counter() - t0

                if not text:
                    print("\r  [No speech detected]", end="", flush=True)
                    continue

                print(f"\r  You: {text}")
                print(f"  (STT: {stt_time * 1000:.0f}ms)", flush=True)

                # --- 3. Send to Claude Code ---
                t1 = time.perf_counter()
                await client.query(text)
                engine.clear_barge_in()
                sentence_buffer.reset()

                first_token_time = None
                first_audio_time = None
                response_text = []

                # --- 4. Stream response -> TTS -> Speaker ---
                if use_streaming_tts:
                    streaming_gen = StreamingTTSGen(tts, voice_state)

                async for msg in client.receive_response():
                    if engine.get_barge_in():
                        print("\n  [Barge-in detected — interrupting]")
                        await client.interrupt()
                        engine.clear_barge_in()
                        engine.flush_playback()
                        break

                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                if first_token_time is None:
                                    first_token_time = time.perf_counter() - t1

                                response_text.append(block.text)

                                if use_streaming_tts:
                                    streaming_gen.append_text(block.text)
                                    for chunk in streaming_gen.process():
                                        if first_audio_time is None:
                                            first_audio_time = time.perf_counter() - t1
                                        _write_audio_to_engine(
                                            engine, chunk, config.sample_rate
                                        )
                                        if engine.get_barge_in():
                                            break
                                else:
                                    sentence_buffer.add(block.text)
                                    while sentence_buffer.has_segment():
                                        segment = sentence_buffer.flush()
                                        if not segment:
                                            continue
                                        for chunk in tts.generate_audio_stream(
                                            voice_state, segment
                                        ):
                                            if first_audio_time is None:
                                                first_audio_time = time.perf_counter() - t1
                                            _write_audio_to_engine(
                                                engine, chunk, config.sample_rate
                                            )
                                            if engine.get_barge_in():
                                                break
                                        if engine.get_barge_in():
                                            break

                                if engine.get_barge_in():
                                    break

                            elif isinstance(block, ToolUseBlock):
                                tool_name = getattr(block, "name", "tool")
                                print(f"  [Tool: {tool_name}]", end="", flush=True)

                    elif isinstance(msg, ResultMessage):
                        break

                # Flush remaining text
                if not engine.get_barge_in():
                    if use_streaming_tts:
                        for chunk in streaming_gen.process_last():
                            if first_audio_time is None:
                                first_audio_time = time.perf_counter() - t1
                            _write_audio_to_engine(engine, chunk, config.sample_rate)
                            if engine.get_barge_in():
                                break
                    else:
                        remaining = sentence_buffer.flush_all()
                        if remaining:
                            for chunk in tts.generate_audio_stream(voice_state, remaining):
                                if first_audio_time is None:
                                    first_audio_time = time.perf_counter() - t1
                                _write_audio_to_engine(engine, chunk, config.sample_rate)
                                if engine.get_barge_in():
                                    break

                full_text = "".join(response_text)
                print(f"\n  Claude: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")

                if first_token_time is not None:
                    print(f"  (TTFT: {first_token_time * 1000:.0f}ms", end="")
                    if first_audio_time is not None:
                        print(f", TTFV: {first_audio_time * 1000:.0f}ms", end="")
                    print(")")

                while engine.is_playing() and not engine.get_barge_in():
                    await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nVoice session ended.")
    finally:
        engine.destroy()


def _has_semantic_vad(stt) -> bool:
    """Check if the STT backend supports semantic VAD."""
    try:
        from pocket_tts.voice.stt.dsm_stt import DsmSTT

        return isinstance(stt, DsmSTT)
    except ImportError:
        return False
