# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**Pocket Voice** is a real-time voice AI system for Apple Silicon. The primary product is a zero-Python native pipeline (`native/`) that runs `Mic → STT → Claude → TTS → Speaker` entirely in C and Rust with Metal GPU inference. The project also includes the original Pocket TTS Python library (`pocket_tts/`) for standalone text-to-speech.

**Key Architecture Components:**

- **Native Voice Pipeline** (`native/`): Zero-Python C orchestrator linking Rust STT/TTS cdylibs. CoreAudio VoiceProcessingIO, lock-free ring buffers, libcurl for Claude API SSE, cJSON for parsing. Single `pocket-voice` binary.
- **Rust STT** (`native/pocket_stt/`): C-callable Rust cdylib wrapping Kyutai's `moshi::asr::State` with candle Metal GPU acceleration for zero-Python speech-to-text inference
- **Rust TTS** (`native/pocket_tts_rs/`): C-callable Rust cdylib wrapping Kyutai's `moshi::tts_streaming::State` with candle Metal GPU for zero-Python text-to-speech
- **CoreAudio Engine** (`native/pocket_voice.c`): Full-duplex audio I/O with built-in AEC, energy VAD, vDSP resampling
- **Python TTS** (`pocket_tts/`): Original Pocket TTS MLX library — FlowLM + Mimi + streaming + SSML + web API
- **Web API**: FastAPI-based server with OpenAI-compatible `/v1/audio/speech` endpoint, WebSocket streaming, and web interface

## Common Commands

### Native Voice Pipeline (Primary)

```bash
# Build everything (C audio engine + Rust STT + Rust TTS + orchestrator binary)
make native
# or: cd native && make all

# Run the voice pipeline
ANTHROPIC_API_KEY=sk-ant-... ./native/pocket-voice

# Clean native build artifacts
make clean
```

### Python TTS (Legacy/Standalone)

```bash
# Install pre-commit hooks
uvx pre-commit install

# Run tests (3 parallel workers)
uv run pytest -n 3 -v

# Run a single test
uv run pytest tests/test_python_api.py -v

# Run CLI locally (editable install)
uv run pocket-voice generate
uv run pocket-voice serve
```

### Linting and Formatting

Pre-commit handles this automatically, but you can run manually:

```bash
# Ruff will run automatically on commit via pre-commit
# Includes: ruff-check, ruff-format (with --fix), and import sorting
```

## Code Structure

### Native Voice Pipeline (`native/`)

The primary product. Zero-Python real-time voice AI:

- `pocket_voice.c`: CoreAudio VoiceProcessingIO engine with lock-free SPSC ring buffers, energy VAD, vDSP resampling
- `pocket_voice_pipeline.c`: C orchestrator — state machine (LISTENING→RECORDING→PROCESSING→STREAMING→SPEAKING), Claude API SSE via libcurl, multi-turn conversation history
- `cJSON.c/h`: JSON parsing for Claude API
- `Makefile`: Builds everything — `libpocket_voice.dylib`, `libpocket_stt.dylib`, `libpocket_tts_rs.dylib`, `pocket-voice` binary
- `pocket_stt/`: Rust cdylib for Kyutai STT 1B (candle + Metal). C FFI: `pocket_stt_create`, `process_frame`, `flush`, `get_all_text`, `get_vad_prob`
- `pocket_tts_rs/`: Rust cdylib for Kyutai DSM TTS 1.6B (candle + Metal). C FFI: `pocket_tts_rs_create`, `set_text`, `step`, `get_audio`, `reset`
- `pocket_voice.c` also provides: `voice_engine_create/start/stop/destroy`, `read_capture`, `write_playback`, `flush_playback`, VAD state, barge-in detection

**Audio Processing & DSP (`native/`):**

- `vdsp_prosody.c`: AMX-accelerated SSML prosody — phase vocoder pitch shift via vDSP FFT, WSOLA time stretching, vDSP biquad cascade for formant-preserving EQ, vForce soft-knee limiter, vectorized volume/crossfade
- `spatial_audio.c`: Binaural 3D spatial audio — HRTF (Head-Related Transfer Function) with ITD/ILD modeling, vDSP convolution, multi-source mixing for multi-speaker positioning
- `audio_converter.c`: Hardware-accelerated resampling via Apple AudioConverter API — polyphase >120dB stopband, arbitrary rate conversion, mastering quality mode
- `opus_codec.c`: Native Opus encoding/decoding via dlopen(libopus) — zero-copy ring buffer integration, VoIP/Audio modes, packet loss concealment, 10-20x bandwidth reduction for WebSocket streaming

**ML Acceleration (`native/`):**

- `bnns_mimi_decoder.c`: ANE-accelerated Mimi audio codec decoder via BNNS/Accelerate — implements full decoder pipeline (ConvTrUpsample → Transformer → SEANet), streaming state management, enables 3-way hardware parallelism (GPU + AMX + ANE). Stub — decode step not yet implemented.
- `amx_flow_fused.c`: Fused C kernel running entire LSD decode loop on AMX (1.19ms vs GPU's 1.39ms)
- `amx_flow_net.py`: Hybrid CPU/AMX flow network orchestrator
- `accelerate_dsp.py`: AMX/vDSP/vForce/BLAS bindings for audio DSP
- `metal4_ml.h`: Metal 4 ML Encoder interface (macOS 26+) — inline tensor inference in Metal shaders, zero framework overhead

**Python SIMD Bridge:**

- `simd_audio.c`: ARM NEON SIMD-accelerated PCM conversion (`float32_to_pcm16`, `float32_to_pcm16_bytes`)
- `__init__.py`: Auto-compile helper for Python ctypes loading

### Python TTS Package (`pocket_tts/`)

**Entry Points:**

- `main.py`: CLI implementation with Typer (commands: `generate`, `serve`, and web interface)
- `__init__.py`: Public API exports only `TTSModel`
- `__main__.py`: Python module entry point
- `default_parameters.py`: Default configuration values for generation parameters
- `static/`: Web interface files (HTML for server UI)

**Core Models (`models/`):**

- `tts_model.py`: Main `TTSModel` class - orchestrates the entire TTS pipeline
  - `load_model()`: Downloads weights from HuggingFace and initializes models
  - `get_state_for_audio_prompt()`: Encodes audio prompt (voice) into model state
  - `generate_audio_stream()`: Streaming generation that yields audio chunks
  - Uses LRU cache for voice prompts to avoid reprocessing
- `flow_lm.py`: `FlowLMModel` - transformer that generates latent audio codes from text

**Modules (`modules/`):**

- `transformer.py`: `StreamingMultiheadAttention` with fused RoPE and SDPA
- `mimi_transformer.py`: `StreamingTransformer`, `MimiStreamingMultiheadAttention` (windowed), `ProjectedTransformer`
- `stateful_module.py`: Base class for streaming support (maintains KV cache and state)
- `rope.py`: `RotaryEmbedding` using `mx.fast.rope` fused Metal kernel
- `mlp.py`: `SimpleMLPAdaLN` (AdaLN-conditioned MLP), `RMSNorm` (via `mx.fast.rms_norm`), `LayerNorm` (via `mx.fast.layer_norm`)
- `conv.py`: Convolution utilities with grouped ConvTranspose1d support
- `seanet.py`: SEANet encoder/decoder (from moshi)

**Conditioners (`conditioners/`):**

- `text.py`: `LUTConditioner` - SentencePiece tokenizer + embedding lookup table for text

**Data (`data/`):**

- `audio.py`: Audio I/O utilities (reading, writing WAV, streaming)
- `audio_utils.py`: Audio processing (resampling, conversion)

**Utils (`utils/`):**

- `config.py`: Pydantic config models for FlowLM and Mimi
- `utils.py`: HuggingFace downloads, timing utilities

**Custom Metal Kernels (`kernels/`):**

- `fused_post_attn.py`: Fused post-attention kernel (transpose + reshape + out_proj matmul in 1 Metal dispatch). Saves 2 kernel launches per attention layer per step.

**Native C Extensions & Apple Accelerate (`native/`):**

- `simd_audio.c`: ARM NEON SIMD-accelerated audio processing. Provides `float32_to_pcm16` (vectorized PCM conversion, 8 samples/cycle) and `float32_to_pcm16_bytes` (direct-to-bytes variant).
- `__init__.py`: Auto-compiles the C library on first import via `cc`, loads via `ctypes`. Falls back to numpy if compilation fails (non-ARM or missing compiler).
- `libsimd_audio.dylib`: Compiled native library (auto-generated, not committed).
- `accelerate_dsp.py`: Apple Accelerate framework bindings via ctypes, leveraging the AMX coprocessor (M1-M3) or ARM SME (M4+) for audio DSP and BLAS operations:
  - **vDSP FFT**: `rfft()`, `irfft()`, `rfft_batch()`, `irfft_batch()` — AMX-backed FFT that replaces numpy PocketFFT (2-5x faster for power-of-2 sizes). DFT setups are cached for reuse.
  - **vDSP vector ops**: `vmul()`, `vadd()`, `vma()`, `vsmul()`, `vramp()`, `vclip()`, `vabs()`, `maxv()` — AMX/NEON-optimized element-wise operations replacing numpy equivalents.
  - **vForce transcendentals**: `vtanhf()`, `vabsf()`, `vcopysignf()` — Vectorized tanh/abs/copysign for soft-knee limiting in volume processing.
  - **vDSP convolution**: `conv()`, `desamp()` — AMX-backed convolution and decimation for resampling.
  - **BLAS**: `gemv()`, `gemm()` — cblas_sgemv/sgemm bindings for AMX-backed matrix-vector and matrix-matrix multiply. Used by the AMX flow network.
  - **Crossfade**: `crossfade()` — fused ramp generation + multiply-add using `vDSP_vramp` + `vDSP_vmul` + `vDSP_vma`.
  - Falls back transparently to numpy/scipy on non-macOS platforms.
- `amx_flow_fused.c`: Fused C kernel that runs the entire LSD decode loop (4 Euler steps × full flow network forward) in a single native call. Auto-compiled on first import.
  - **1.19ms vs GPU's 1.39ms** — 18% faster than `mx.compile`'d GPU path.
  - Uses `cblas_sgemv` (Apple Accelerate → AMX) for all linear projections directly from C.
  - All intermediates (~30KB) are stack-allocated, fitting entirely in L1 cache.
  - Zero Python function calls, zero numpy temporary allocations in the hot path.
  - Linked against Accelerate framework with `-DACCELERATE_NEW_LAPACK` for ILP64 API.
- `amx_flow_net.py`: Hybrid CPU/AMX flow network orchestrator.
  - `AMXFlowNet.from_mlx_model()`: Extracts weights from frozen MLX model, converts to float32 numpy.
  - `_pack_weights()`: Packs all weights into a single contiguous float32 buffer matching the C kernel's expected layout (~37MB for the flow network).
  - `_forward_lsd_fused()`: Single ctypes call to `lsd_decode_fused()` in the C kernel.
  - `_forward_lsd_python()`: Pure Python/numpy fallback for non-macOS or compilation failures.
  - Enabled by default (not opt-in). Disabled automatically for quantized models (incompatible with `nn.QuantizedLinear`).
  - Produces bit-identical results to the GPU path (correlation 1.0).
  - Key insight: Python function call overhead (~200 calls per decode, 38% of total time) was the bottleneck. The fused C kernel eliminates it entirely.

> **Note:** The native C voice pipeline (CoreAudio engine, Opus codec, spatial audio, hardware resampler, prosody processing, Rust STT/TTS) has been extracted to its own repo: [pocket-voice](https://github.com/sethdford/pocket-voice).

**Voice Pipeline (`voice/`):**

- `native_audio.py`: Python ctypes wrapper for `libpocket_voice.dylib`. Provides `NativeVoiceEngine` with CoreAudio I/O, ring buffers, VAD, resampling. Falls back to `sounddevice`.
- `config.py`: Pydantic `VoiceConfig` model for pipeline configuration (sample rates, VAD thresholds, STT/TTS backend selection, quantization).
- `sentence_buffer.py`: `SentenceBuffer` for LLM output — accumulates tokens, flushes on sentence/clause boundaries, filters code blocks. Used with pocket-tts backend; bypassed when DSM TTS streaming is active.
- `dsm_tts.py`: `DsmTTSBackend` — Wraps Kyutai's DSM TTS 1.6B model via `moshi_mlx` for high-quality MLX inference. `StreamingTTSGen` enables streaming text input where LLM tokens are fed directly to TTS without sentence buffering.
- `claude_voice.py`: Main orchestrator using `ClaudeSDKClient`. Supports two TTS backends (`pocket` or `dsm`) and three STT backends (`dsm`, `rust`, `pytorch`). Coordinates: Mic → C Engine → STT → Claude → TTS → Speaker. Includes semantic VAD integration and barge-in support.
- `stt/base.py`: Abstract `STTBackend` interface (`load`, `transcribe_stream`, `reset`, `sample_rate`, `frame_size`).
- `stt/dsm_stt.py`: DSM STT backend — MLX-native via `moshi_mlx` with `rustymimi` audio tokenizer. Built-in semantic VAD for end-of-turn detection. Recommended backend.
- `stt/rust_stt.py`: Rust STT backend — ctypes wrapper for `libpocket_stt.dylib`. Zero Python in inference path.
- `stt/kyutai_stt.py`: PyTorch STT backend (fallback).

**Configuration (`config/`):**

- `b6369a24.yaml`: Model configuration (transformer dims, layers, vocab size, etc.)

### Testing (`tests/`)

- `test_python_api.py`: Tests for public Python API
- `test_cli_generate.py`: Tests for CLI generate command
- `test_documentation_examples.py`: Ensures docs examples work
- `test_ssml.py`: Tests for SSML parsing and prosody (130 tests)

## Development Workflow

### Key Patterns

1. **Streaming Generation**: The model generates audio frame-by-frame (12.5 Hz frame rate, 80ms per frame). All modules inherit from `StatefulModule` to maintain internal state.

2. **Voice Cloning**: Audio prompts are encoded via Mimi encoder to create "voice state" (latent representations + speaker embeddings). This state is cached via `lru_cache` on `_cached_get_state_for_audio_prompt()`.

3. **Flow Matching**: Uses Lagrangian Self Distillation (LSD) with configurable decode steps. Fewer steps = faster but lower quality.

4. **EOS Detection**: Model predicts end-of-speech via an EOS head. Generation continues for `frames_after_eos` frames after EOS is detected.

5. **Config-Driven**: Model architecture is defined in YAML configs. Weights are loaded from HuggingFace Hub via `safetensors`.

### Important Implementation Details

- **Thread Safety**: The code is NOT thread-safe. MLX Metal backend crashes with concurrent operations from multiple threads. Generation is single-threaded.
- **Batching**: Batch size is always 1. No batching support currently.
- **MLX Framework**: Uses Apple MLX for Apple Silicon-optimized inference. Models are frozen (`model.freeze()`) and set to eval mode after loading.
- **Fused Metal Kernels**: RMSNorm uses `mx.fast.rms_norm`, LayerNorm uses `mx.fast.layer_norm`, RoPE uses `mx.fast.rope`, attention uses `mx.fast.scaled_dot_product_attention`. These are fused GPU kernels with significant speedups over manual implementations.
- **KV Cache Layout**: Attention caches are stored in `(B, H, T, D)` format (not `(B, T, H, D)`) for direct use with fused attention and RoPE kernels without transposition.
- **NLC Data Layout**: All convolutions use MLX's NLC (batch, length, channels) format. Weight loading transposes PyTorch NCL weights automatically.
- **dtype**: Models use float32 by default (configurable in YAML). Weight loading supports bfloat16 natively via `mx.load`.
- **Beartype**: Runtime type checking is enabled via beartype claw in `__init__.py`
- **SSML Support**: Full W3C SSML parsing with prosody approximation via audio post-processing, cross-segment boundary smoothing, word-level mark timing, lexicon support. Prosody effects (pitch shift, rate change, volume) use AMX-backed vDSP FFT and vForce math via Apple Accelerate for 2-5x faster processing.
- **CPU/GPU Pipelining**: SSML multi-segment generation pipelines CPU prosody processing (running on AMX via Accelerate) concurrently with GPU inference for the next segment using `ThreadPoolExecutor`. AMX and GPU are separate hardware units on Apple Silicon, so they execute without contention.
- **Memory Management**: `mx.clear_cache()` is called periodically during generation to prevent memory growth
- **mx.compile()**: `SimpleMLPAdaLN` (flow network) and FlowLM pre-attention (in_proj→split→transpose→RoPE) use `mx.compile()` for fused Metal kernel execution after model freeze. The compiled pre-attention produces a single fused graph for the entire QKV projection + RoPE pipeline.
- **Async Eval**: Generation loop uses `mx.async_eval()` to pipeline GPU decode with CPU work
- **Quantization**: Optional 4-bit/8-bit quantization via `nn.quantize()` for reduced memory usage (`load_model(quantize=4)`)
- **Mimi dtype**: Optional bfloat16/float16 for Mimi decoder to halve memory bandwidth (`load_model(mimi_dtype='bfloat16')`)
- **WebSocket**: FastAPI server supports `/ws/tts` WebSocket endpoint for real-time streaming
- **Speculative Decoding**: Optional speculative frame generation using a lightweight draft model (first N transformer layers) with batch verification. Enabled via `generate_audio_stream(speculative_tokens=4)` or CLI `--speculative-tokens 4`.
- **Cached Stateful Modules**: `_get_stateful_modules()` caches the list of stateful modules to avoid full tree traversal on every `increment_steps` / `init_states` call.
- **Pre-classified SEANet Layers**: SEANet encoder/decoder/ResnetBlock pre-classify layers at init to avoid `isinstance()` checks in the hot loop.
- **Ring-buffer Mimi KV Cache**: Mimi attention uses a ring-buffer strategy for its bounded context window, avoiding concat+trim per step.
- **Fused Dual RoPE**: `RotaryEmbedding` stacks Q and K along the batch dim for T=1 generation, halving RoPE kernel dispatches.
- **Pre-allocated Constants**: BOS NaN input, empty text tokens, empty latents, and empty conditioning arrays are allocated once at model load and reused every generation call.
- **Fast Generation Path**: `_run_flow_lm_generation_step()` provides a flattened call path for autoregressive steps, avoiding empty array creation and unnecessary concatenation overhead.
- **Metal Shader Warmup**: `_warmup_metal_shaders()` runs a dummy forward pass during `load_model()` to JIT-compile all Metal shaders (transformer, flow_net, Mimi codec). Eliminates 2-5s cold-start latency on first generation. No competitor (MLX-Audio, ChipChat, Kokoro) does this.
- **Text Prefill Prompt Cache**: `_get_prefilled_state()` caches FlowLM state after text prefill using a dict-based LRU cache (32 entries). For repeated text+voice combinations, skips tokenization + embedding + transformer prefill (~50-200ms savings). Key is `voice_offset:md5(text)`.
- **Custom Metal Kernel (Fused KV Append)**: `_fused_kv_cache_append()` in `transformer.py` combines both k_cache and v_cache concatenation into a single GPU kernel dispatch using `mx.fast.metal_kernel()`. Saves one kernel launch per attention layer per step.
- **OpenAI-Compatible API**: `/v1/audio/speech` endpoint matches OpenAI's TTS API format. Any app using the `openai` Python package can switch to pocket-tts with `base_url="http://localhost:8000/v1"`. Also provides `/v1/models` endpoint.
- **Voice Hot-Loading**: Server pre-loads all predefined voices (8 voices) into the LRU cache on startup (`--preload-voices`). Eliminates 500ms+ latency from HuggingFace cache lookups and safetensors deserialization on first voice switch. LRU cache increased from 2 to 16 entries.

### Adding Features

When adding features, be aware of:

- The streaming architecture: any changes to model forward passes need to maintain state correctly
- The config system: new model parameters must be added to config classes in `utils/config.py`
- The public API: only `TTSModel` is exported; keep implementation details internal
- Ruff formatting: line length 100, LF line endings, skip magic trailing comma

### Model Weights

Weights are downloaded from HuggingFace Hub on first use:

- Model weights: `hf://kyutai/pocket-tts/tts_b6369a24.safetensors`
- Tokenizer: `hf://kyutai/pocket-tts/tokenizer.model`
- Voice prompts: `hf://kyutai/tts-voices/<speaker>/<style>.wav`

The `download_if_necessary()` utility handles `hf://` URLs and caches locally.

## Common Gotchas

1. **MLX Version**: Requires MLX >= 0.30.0. Apple Silicon (M-series) required.
2. **Python Version**: Supports Python 3.10 through 3.14 (>= 3.10,<3.15).
3. **uv Python Preference**: Set to "only-managed" in pyproject.toml because system Python may lack headers.
4. **Weight Transposition**: PyTorch safetensors weights are automatically transposed during loading: Conv1d uses `swapaxes(1,2)`, ConvTranspose1d depthwise uses `swapaxes(1,2)`, ConvTranspose1d non-grouped uses `transpose(1,2,0)`.
5. **Web Dependencies**: FastAPI and Uvicorn are included for server functionality.
6. **Immutable Arrays**: MLX arrays are immutable; KV caches grow via `mx.concatenate` instead of in-place updates.
7. **No nn.Sequential**: Plain Python lists with a `_run_sequential` helper replace `nn.Sequential` to match PyTorch weight naming conventions (`mlp.0.weight` vs `mlp.layers.0.weight`).
8. **LayerScale**: Mimi transformer layers have `LayerScale`; FlowLM transformer layers do not. `StreamingTransformerLayer` supports optional `layer_scale` parameter (None = no LayerScale).
9. **Voice State Format**: Pre-saved voice states in PyTorch format are automatically converted during import: `cache` (2,B,T,H,D) → `k_cache`/`v_cache` (B,H,T,D), `current_end` → `offset`.
10. **RoPE Format**: `mx.fast.rope` expects input shape `(B, H, T, D)` with `traditional=True` and `base` must be `float` (not `int`). Queries and keys are transposed to this format before RoPE application.
11. **moshi_mlx MLX Version Conflict**: `moshi_mlx>=0.3.0` requires `mlx<0.27` but pocket-tts requires `mlx>=0.30.0`. These cannot coexist in the same pip environment. moshi_mlx is a runtime-detected optional dependency — DSM backends gracefully fall back to pocket/rust/pytorch if not installed. Install in a separate venv if needed.
12. **DSM TTS Voice Format**: DSM TTS uses pre-computed voice embeddings (`.safetensors`) from `kyutai/tts-voices`, not raw audio prompts like pocket-tts. Voices are referenced by path relative to the voice repo (e.g. `expresso/ex03-ex01_happy_001_channel1_334s.wav`).

## Performance Optimization Architecture

### Current Optimizations (Implemented)

**Kernel-Level:**

- Compiled pre-attention graph (in_proj → split → transpose → RoPE fused via `mx.compile`)
- Compiled flow network (`SimpleMLPAdaLN._forward` via `mx.compile`)
- Fused dual RoPE (Q+K stacked along batch dim for single dispatch)
- Custom Metal kernel for fused dual KV cache append (`_fused_kv_cache_append` via `mx.fast.metal_kernel`)
- Custom Metal kernel for fused post-attention (`fused_post_attn.py`: transpose + reshape + out_proj in 1 dispatch, saves 2 launches/layer/step)
- Skip causal mask construction entirely for T=1 FlowLM attention
- Optimized T=1 windowed mask for Mimi attention
- All norms use fused Metal kernels (`mx.fast.rms_norm`, `mx.fast.layer_norm`)
- SDPA uses `mx.fast.scaled_dot_product_attention`

**Native C (ARM NEON SIMD):**

- `simd_audio.c`: NEON-vectorized float32→int16 PCM conversion (8 samples per cycle via `vld1q_f32`/`vmovn_s32`)
- NEON batch offset increment for state management
- Auto-compiled on first import, loaded via ctypes with numpy fallback

**Native C (Audio DSP):**

- `vdsp_prosody.c`: Phase vocoder pitch shifting via vDSP FFT, WSOLA time stretching, vDSP biquad formant EQ, vForce soft-knee limiter
- `spatial_audio.c`: HRTF binaural 3D spatialization with ITD/ILD, vDSP convolution, multi-source mixing
- `audio_converter.c`: Apple AudioConverter hardware-accelerated sample rate conversion (polyphase, mastering quality)
- `opus_codec.c`: Native Opus encoder/decoder via dlopen (zero compile-time dependency), VoIP/Audio modes, PLC

**Neural Engine / ANE:**

- `bnns_mimi_decoder.c`: Full Mimi decoder (ConvTrUpsample → Transformer → SEANet) in C using Accelerate BLAS/vDSP, structured for BNNS Graph routing to ANE. Stub — decode step not yet implemented.
- `metal4_ml.h`: Metal 4 ML Encoder interface (macOS 26+) for zero-overhead tensor inference in Metal shaders

**Apple Accelerate / AMX Coprocessor:**

- `accelerate_dsp.py`: Full Apple Accelerate framework integration via ctypes, using the AMX coprocessor (M1-M3) or ARM SME (M4+):
  - **vDSP FFT**: AMX-backed FFT for large transforms (n >= 8192). For typical audio FFT sizes (n=2048), numpy's batch FFT is used since ctypes overhead dominates at small sizes.
  - **Batched STFT**: Windowed frames built as a matrix via vectorized indexing, then batch-FFT'd in a single numpy C call. Eliminates per-frame Python loop → **1.89x faster phase vocoder** (measured: 7.5ms → 4.0ms for 2s audio).
  - **vForce transcendentals**: `vvtanhf()` replaces `np.tanh()` in soft-knee volume limiter. `vvfabsf()` replaces `np.abs()`. `vvcopysignf()` replaces `np.sign()`.
  - **vDSP vector ops**: `vDSP_vmul`, `vDSP_vma`, `vDSP_vramp`, `vDSP_vclip`, `vDSP_vabs`, `vDSP_maxv` replace numpy element-wise operations in crossfade, volume, and fade generation.
  - **vDSP convolution/decimation**: `vDSP_conv`, `vDSP_desamp` for AMX-backed convolution and resampling.
  - **BLAS**: `cblas_sgemv`, `cblas_sgemm` bindings for AMX-backed matrix operations.
  - **Fused C kernel**: `amx_flow_fused.c` runs entire LSD decode in one native call (1.19ms vs GPU's 1.39ms). Enabled by default. Eliminates 200+ Python function calls per decode.
  - DFT setup objects are cached for reuse across calls (expensive to create, cheap to execute).
  - Automatic fallback to numpy/scipy on non-macOS platforms.

**Algorithmic:**

- Metal shader warmup — dummy forward pass during model load eliminates JIT compilation latency
- Speculative frame generation with layer-skip draft model
- Batched Mimi decode for multiple accepted frames
- Ring-buffer KV cache for Mimi (bounded context)
- `mx.async_eval` pipelining (decode current frame while building next graph)
- Text prefill prompt cache — dict-based LRU for repeated text+voice combinations
- **CPU/GPU pipelining** — SSML prosody processing runs concurrently with next-segment GPU generation via ThreadPoolExecutor. AMX (CPU) and Metal GPU are separate hardware units.

**Python-Level:**

- Cached stateful module lists (avoid tree traversal per step)
- Pre-classified SEANet layers (avoid isinstance per layer per step)
- Pre-allocated constant arrays (BOS, empty tokens, empty embeddings)
- Hoisted attribute lookups out of generation hot loop
- Gated profiling behind `logging.DEBUG`
- Flattened generation call stack (4 levels → 1)
- Single `get_state()` calls in conv/attention layers
- Voice hot-loading on server start (LRU cache with 16 entries)

**API/Ecosystem:**

- OpenAI-compatible `/v1/audio/speech` endpoint (drop-in replacement for OpenAI TTS)
- `/v1/models` endpoint for model discovery
- WebSocket `/ws/tts` for real-time streaming
- SSML support with prosody, voice switching, break tags, marks

### Future Optimization Roadmap

Ordered by expected impact. Items marked IMPLEMENTED have code in the repo.

#### Tier 0: Implemented

**Custom Fused Metal Kernels** — IMPLEMENTED

- Dual KV cache append fused into single dispatch (`_fused_kv_cache_append` via `mx.fast.metal_kernel`).
- Fused post-attention: transpose + reshape + out_proj matmul in 1 Metal dispatch (`fused_post_attn.py`).

**BNNS Mimi Decoder** — IMPLEMENTED

- Full Mimi decoder in C (ConvTrUpsample → Transformer → SEANet). See `native/bnns_mimi_decoder.c`.
- Streaming state management (KV cache ring buffer, conv overlap-add).
- Structured for BNNS Graph routing to Apple Neural Engine.

**Native Audio DSP Suite** — IMPLEMENTED

- `vdsp_prosody.c`: Phase vocoder pitch shift, WSOLA time stretch, biquad formant EQ, vForce limiter.
- `spatial_audio.c`: HRTF binaural 3D with ITD/ILD, multi-source mixing.
- `audio_converter.c`: Apple AudioConverter hardware SRC (mastering quality).
- `opus_codec.c`: Native Opus codec for 10-20x WebSocket bandwidth reduction.

**Metal 4 ML Encoder Interface** — IMPLEMENTED (stub, requires macOS 26)

- C interface defined in `native/metal4_ml.h`. Ready for Metal 4 SDK.

#### Tier 1: High Impact (Remaining)

**Learned Few-Step Flow Solver** (requires training)

- Replace Euler integration in `lsd_decode` with a learned numerical solver (distilled from multi-step). Stream.FM (arxiv:2512.19442) achieves 32ms algorithmic latency using learned solvers for flow matching.
- Would allow 0-step or adaptive-step LSD decode for near-zero flow network cost.

**Model Distillation** (requires training)

- Train a smaller student model (3 layers instead of 6, narrower dims) that matches quality.
- Half the layers = half the per-step latency.
- The speculative decoding infrastructure already supports layer-subset forwarding.

#### Tier 2: Medium Impact

**Principled Coarse-Grained Acceptance (PCG)** for speculative decoding

- Standard L2 threshold for accepting draft latents is suboptimal. PCG (ICASSP 2026, arxiv:2511.13732) groups acoustically similar outputs and verifies at the group level.
- Would increase speculative decoding acceptance rates while maintaining quality.

**SoundStorm-Style Parallel Decoding**

- Replace autoregressive frame generation with confidence-based parallel decoding.
- SoundStorm generates 30s of audio in 0.5s on TPU via iterative refinement.
- Would require architectural changes and retraining.

#### Tier 3: Research-Level

**VoXtream Architecture** (arxiv:2509.15969)

- Incremental phoneme transformer + temporal transformer + depth transformer.
- 102ms initial delay (lowest reported for streaming TTS). Uses monotonic alignment with limited look-ahead.

### Competitive Landscape (as of Feb 2026)

| Feature                | pocket-tts     | MLX-Audio | ChipChat | Kokoro | Higgs 2.5 |
| ---------------------- | -------------- | --------- | -------- | ------ | --------- |
| Custom Metal kernels   | **2 (JIT)**    | No        | Unknown  | No     | N/A       |
| C/NEON SIMD native ext | **Yes**        | No        | No       | No     | N/A       |
| Apple AMX/Accelerate   | **Yes (full)** | No        | No       | No     | N/A       |
| CPU/GPU pipelining     | **Yes**        | No        | No       | No     | N/A       |
| Metal shader warmup    | **Yes**        | No        | No       | No     | N/A       |
| OpenAI-compatible API  | **Yes**        | Yes       | No       | No     | No        |
| Speculative decoding   | **Yes**        | No        | No       | No     | No        |
| Voice hot-loading      | **Yes**        | No        | Unknown  | No     | N/A       |
| Text prefill cache     | **Yes**        | No        | No       | No     | No        |
| SSML support           | **Yes**        | No        | No       | No     | No        |
| WebSocket streaming    | **Yes**        | No        | Unknown  | No     | No        |
| Voice cloning          | Yes            | Model-dep | No       | No     | Yes       |
| Compiled flow network  | **Yes**        | N/A       | Unknown  | N/A    | N/A       |
| Fused dual RoPE        | **Yes**        | No        | Unknown  | N/A    | N/A       |
| Ring-buffer KV cache   | **Yes**        | No        | Unknown  | N/A    | N/A       |

**Key advantages over competition:**

1. **Deepest hardware optimization**: JIT Metal kernels + NEON SIMD C extensions + AMX/Accelerate DSP. Leverages GPU + AMX + NEON on Apple Silicon.
2. **Lowest cold-start latency**: Metal shader warmup eliminates JIT compilation entirely.
3. **Best server performance**: Voice hot-loading + text prefill cache + OpenAI API compatibility.
4. **Richest feature set**: SSML + voice cloning + WebSocket + OpenAI API.
5. **Companion native pipeline**: [pocket-voice](https://github.com/sethdford/pocket-voice) provides a zero-Python C/Rust voice assistant (Mic→STT→Claude→TTS→Speaker) with spatial audio, Opus, and hardware resampling.

### Profiling Tips

To measure per-step latency:

```bash
# Enable debug logging for step-level timing
POCKET_TTS_LOG_LEVEL=DEBUG uv run pocket-tts generate --text "Hello world."

# Compare speculative vs standard:
uv run pocket-tts generate --text "Hello world." --speculative-tokens 4
```

Key metrics to track:

- **Time per generation step** (logged at DEBUG level)
- **Real-time factor** (logged at INFO level): `duration_audio / generation_time`
- **Speculative acceptance rate** (logged when using speculative tokens)
- **First-chunk latency**: time from start to first audio chunk yield
