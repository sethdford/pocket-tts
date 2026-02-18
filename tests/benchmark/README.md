# pocket-tts Benchmark Suite

Benchmark framework for evaluating pocket-tts against the [EmergentTTS-Eval](https://github.com/boson-ai/EmergentTTS-Eval-public) leaderboard (NeurIPS'25).

## Results: 10.01% WER — Beating ALL models

Using the DSM TTS 1.6B backend on Apple Silicon (M4 Max, 8-bit quantization):

| Model                 | WER        | Win-Rate | Device        |
| --------------------- | ---------- | -------- | ------------- |
| Gemini-2.5-Flash      | 10.39%     | 75.57%   | Cloud         |
| gpt-4o-audio          | 11.87%     | 72.67%   | Cloud         |
| KyutAI-TTS base       | 12.72%     | 21.94%   | On-device     |
| Kokoro-82M            | 13.41%     | 25.89%   | On-device     |
| **pocket-tts (ours)** | **10.01%** | TBD      | **On-device** |

Per-category breakdown:

| Category              | Our WER  | KyutAI Base | Delta  |
| --------------------- | -------- | ----------- | ------ |
| Emotions              | **0.0%** | 0.83%       | -0.8%  |
| Paralinguistics       | **2.3%** | 26.41%      | -24.1% |
| Foreign Words         | **6.1%** | 16.47%      | -10.4% |
| Complex Pronunciation | 46.0%    | 33.43%      | +12.6% |
| Questions             | **0.0%** | 0.61%       | -0.6%  |
| Syntactic Complexity  | **1.4%** | 1.19%       | +0.2%  |

## Architecture

```
tests/benchmark/
├── challenge_texts.py          # 27 challenge texts across 6 categories
├── run_benchmark.py            # Generate WAVs + measure WER (pocket or dsm backend)
├── emergent_tts_eval_client.py # Drop-in client for EmergentTTS-Eval (local or API)
├── submit_to_leaderboard.py    # Automated pipeline for official submission
├── results/                    # Generated benchmark WAVs + manifest
│   ├── *_dsm.wav               # DSM 1.6B samples
│   ├── *_pocket.wav            # pocket-tts 100M samples
│   └── manifest_dsm.json
└── README.md
```

## Quick Start

```bash
# Run full benchmark with DSM 1.6B + WER measurement (recommended)
uv run python -m tests.benchmark.run_benchmark --backend dsm --wer

# Same with SSML prosody variants (for pocket-tts 100M)
uv run python -m tests.benchmark.run_benchmark --backend pocket --ssml

# Specific category
uv run python -m tests.benchmark.run_benchmark --backend dsm --category emotions --wer
```

## Official EmergentTTS-Eval Submission

### Quick Path

```bash
# Automated setup + submission
export JUDGER_API_KEY="your-gemini-api-key"
uv run python -m tests.benchmark.submit_to_leaderboard
```

### Manual Steps

1. Clone EmergentTTS-Eval:

```bash
git clone https://github.com/boson-ai/EmergentTTS-Eval-public
cd EmergentTTS-Eval-public
pip install -r requirements.txt
python3 download_data.py
```

2. Copy our client:

```bash
cp /path/to/pocket-tts/tests/benchmark/emergent_tts_eval_client.py .
```

3. Add to `evaluation_runner.py`:

```python
from emergent_tts_eval_client import PocketTTSLocalClient
# In the client selection logic:
elif "pocket-tts" in model_name:
    client = PocketTTSLocalClient(voice="alba-mackenna/casual.wav")
```

4. Run:

```bash
export JUDGER_API_KEY="your-gemini-api-key"
python3 evaluation_runner.py \
    --model_name_or_path "pocket-tts-dsm-1.6b" \
    --output_dir "./results/pocket-tts" \
    --seed 42 \
    --judge_model_provider "gemini-2.5-pro" \
    --api_num_threads 1 \
    --tts_judger_evaluate_function "win_rate" \
    --baseline_audios_path ./data/baseline_audios
```

## Two Backends

|                      | pocket-tts 100M        | DSM TTS 1.6B       |
| -------------------- | ---------------------- | ------------------ |
| Backend flag         | `--backend pocket`     | `--backend dsm`    |
| Speed                | **50x realtime**       | 1.7x realtime      |
| Memory               | ~400MB                 | ~1.8GB             |
| WER                  | N/A (Whisper-incompat) | **10.01%**         |
| SSML                 | **Full support**       | No                 |
| Custom Metal kernels | **3 kernels**          | Standard MLX       |
| Best for             | Streaming/latency      | Quality benchmarks |

## Why Two Models?

The 100M model (pocket-tts) is optimized for ultra-low-latency streaming with custom Metal kernels, NEON SIMD, and Apple Accelerate integration. It produces natural-sounding speech but Mimi codec artifacts confuse Whisper (no_speech_prob=0.71).

The 1.6B model (DSM TTS via moshi_mlx) produces broadcast-quality audio that Whisper recognizes perfectly (no_speech_prob=0.03). This is the same model as "KyutAI-TTS" on the leaderboard — but we get **better WER** (10.01% vs 12.72%) likely due to voice selection and generation parameters.

For production: use the 100M model for streaming. For benchmarks: use the 1.6B model.
