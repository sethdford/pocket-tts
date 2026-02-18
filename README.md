# Pocket Voice

Real-time voice AI on Apple Silicon. Talk to Claude (or any LLM) with your voice — zero Python in the hot path.

## What It Does

Pocket Voice is a full-duplex voice pipeline that runs entirely in native code:

```
Mic → STT → Claude → TTS → Speaker
```

- **Speech-to-Text**: Kyutai STT 1B (Rust + candle + Metal)
- **LLM**: Claude API via streaming SSE (C + libcurl)
- **Text-to-Speech**: Kyutai DSM TTS 1.6B (Rust + candle + Metal)
- **Audio I/O**: CoreAudio VoiceProcessingIO with built-in AEC (C)

All inference runs on Metal GPU. The orchestrator is a single C binary (`pocket-voice`) with no Python, no garbage collector, and no interpreter overhead.

## Performance

| Metric                                | Target                           |
| ------------------------------------- | -------------------------------- |
| Audio I/O latency                     | ~5ms (256-frame buffer at 48kHz) |
| End-to-end (speech end → first audio) | Network-bound (Claude TTFT)      |
| Barge-in response                     | <10ms                            |

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1+)
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Anthropic API key

### Build

```bash
cd native
make all    # Builds libpocket_voice.dylib, libpocket_stt.dylib,
            # libpocket_tts_rs.dylib, and the pocket-voice binary
```

### Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd native
./pocket-voice
```

Options:

```
./pocket-voice --help
./pocket-voice --claude-model claude-sonnet-4-20250514
./pocket-voice --system "You are a pirate. Speak like one."
./pocket-voice --no-vad              # Use energy VAD only
./pocket-voice --vad-threshold 0.8   # Stricter end-of-turn detection
```

Models are downloaded from HuggingFace automatically on first run.

## Architecture

```
native/
├── pocket_voice.c              CoreAudio engine, ring buffers, VAD
├── pocket_voice_pipeline.c     Main orchestrator (state machine)
├── cJSON.c/h                   JSON parsing for Claude API
├── Makefile                    Build system
├── pocket_stt/                 Rust cdylib: Kyutai STT 1B
│   └── src/lib.rs
└── pocket_tts_rs/              Rust cdylib: Kyutai DSM TTS 1.6B
    └── src/lib.rs
```

**State Machine**: `LISTENING → RECORDING → PROCESSING → STREAMING → SPEAKING → LISTENING`

The pipeline supports:

- **Barge-in**: Interrupt Claude mid-sentence by speaking
- **Semantic VAD**: Uses the STT model's built-in VAD for accurate end-of-turn detection
- **Multi-turn conversation**: 20-turn sliding window history with Claude
- **Streaming TTS**: Audio generation begins as soon as first Claude tokens arrive
- **Latency instrumentation**: TTFT, E2E, and total turn timing logged to stderr

## Python TTS Library

The original Pocket TTS Python library is still available for standalone TTS:

```bash
pip install pocket-voice
```

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

model = TTSModel.load_model()
voice = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(voice, "Hello world!")
scipy.io.wavfile.write("output.wav", model.sample_rate, audio)
```

CLI commands (`generate`, `serve`, `export-voice`) still work via `pocket-voice generate` or `pocket-tts generate`.

## Development

```bash
# Install pre-commit hooks
uvx pre-commit install

# Run Python tests
uv run pytest -n 3 -v

# Build native pipeline
cd native && make all

# Clean everything
cd native && make clean
```

## Project Structure

```
pocket-voice/
├── native/                 C/Rust voice pipeline (primary)
│   ├── pocket_voice.c      CoreAudio engine
│   ├── pocket_voice_pipeline.c  Orchestrator
│   ├── pocket_stt/         Rust STT cdylib
│   └── pocket_tts_rs/      Rust TTS cdylib
├── pocket_tts/             Python TTS library (original)
│   ├── models/             FlowLM, Mimi, TTSModel
│   ├── modules/            Transformer, attention, RoPE
│   ├── ssml/               SSML parsing + prosody
│   └── voice/              Python voice pipeline (deprecated)
├── tests/                  Test suite
├── docs/                   Documentation
└── pyproject.toml          Python package config
```

## Credits

Pocket TTS is small enough to run directly in your browser in WebAssembly/JavaScript.
We don't have official support for this yet, but you can try out one of these community implementations:

- [wasm-pocket-tts](https://github.com/LaurentMazare/xn/tree/main/wasm-pocket-tts) by @LaurentMazare: Rust port of pocket TTS with XN. Demo [here](https://laurentmazare.github.io/pocket-tts/)
- [pocket-tts-onnx-export](https://github.com/KevinAHM/pocket-tts-onnx-export) by @KevinAHM: Model exported to .onnx and run using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). Demo [here](https://huggingface.co/spaces/KevinAHM/pocket-tts-web)
- [pocket-tts](https://github.com/babybirdprd/pocket-tts) by @babybirdprd: Candle version (Rust) with WebAssembly and PyO3 bindings, meaning it can run on the web too.
- [jax-js](https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts) by @ekzhang: Using jax-js, a ML library for the web. Demo [here](https://jax-js.com/tts)

## License

## Alterative implementations

- [pocket-tts-mlx](https://github.com/jishnuvenugopal/pocket-tts-mlx) by @jishnuvenugopal - MLX backend optimized for Apple Silicon
- [pocket-tts-xn](https://github.com/LaurentMazare/xn/tree/main/pocket-tts) by @LaurentMazare - A Rust port of Pocket TTS implemented with XN.
- [pocket-tts-candle](https://github.com/babybirdprd/pocket-tts) by @babybirdprd - Candle version (Rust) with WebAssembly and PyO3 bindings.

## Projects using Pocket TTS

- [pocket-reader](https://github.com/lukasmwerner/pocket-reader) by @lukasmwerner- Browser screen reader
- [pocket-tts-wyoming](https://github.com/ikidd/pocket-tts-wyoming) by @ikidd - Docker container for pocket-tts using Wyoming protocol, ready for Home Assistant Voice use.
- [Sonorus](https://www.nexusmods.com/hogwartslegacy/mods/2409) by @KevinAHM - Talk to any named character in Hogwarts Legacy with their original voice.
- [Mac pocket-tts](https://github.com/slaughters85j/pocket-tts) by @slaughters85j - Mac Desktop App + macOS Quick Action
- [pocket-tts-openai_streaming_server](https://github.com/teddybear082/pocket-tts-openai_streaming_server) by @teddybear082 - OpenAI-compatible streaming server, dockerized and with an `.exe` release
- [pocket-tts-unity](https://github.com/lookbe/pocket-tts-unity) by @lookbe - A Unity 6 integration for Pocket-TTS.

## Prohibited use

Use of our model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.

## Authors

Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse\*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez

\*equal contribution
