#!/usr/bin/env python3
"""Generate benchmark WAV files and measure WER for EmergentTTS-Eval categories.

Generates audio for all challenge texts (plain + SSML variants where available),
measures Word Error Rate via Whisper, and produces a results summary comparable
to the EmergentTTS-Eval leaderboard.

Supports two backends:
  - pocket (100M, fast, ~50x realtime, not Whisper-compatible)
  - dsm (1.6B, high quality, Whisper-compatible, same as leaderboard KyutAI-TTS)

Usage:
    # 1.6B DSM model with WER (the path to becoming the best)
    uv run python -m tests.benchmark.run_benchmark --backend dsm --wer

    # 1.6B DSM + SSML prosody (our competitive advantage)
    uv run python -m tests.benchmark.run_benchmark --backend dsm --ssml --wer

    # 100M pocket model (speed benchmark)
    uv run python -m tests.benchmark.run_benchmark --backend pocket --ssml

    # Specific category only
    uv run python -m tests.benchmark.run_benchmark --backend dsm --category emotions --wer
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from .challenge_texts import BENCHMARK_TEXTS, CATEGORIES

BENCHMARK_DIR = Path(__file__).parent / "results"
SEED = 42

KYUTAI_TTS_BASE_WER = 0.1272
KYUTAI_TTS_BASE_WINRATE = 0.2194


def _normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via minimum edit distance."""
    ref = _normalize_text(reference).split()
    hyp = _normalize_text(hypothesis).split()

    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref)][len(hyp)] / len(ref)


def _transcribe_with_whisper(wav_path: str) -> tuple[str, float]:
    """Transcribe a WAV file using mlx-whisper. Returns (text, no_speech_prob)."""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="en",
        condition_on_previous_text=False,
        no_speech_threshold=0.8,
    )
    no_speech = 0.0
    if result.get("segments"):
        no_speech = result["segments"][0].get("no_speech_prob", 0.0)
    return result["text"].strip(), no_speech


def _load_pocket_backend(voice: str):
    """Load pocket-tts 100M model. Returns (generate_fn, sample_rate, model_name)."""
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(voice)

    def generate(text: str, seed: int) -> np.ndarray:
        return model.generate_audio(voice_state, text, seed=seed)

    return generate, model.sample_rate, "pocket-tts-100M"


def _load_dsm_backend(voice: str):
    """Load DSM TTS 1.6B model. Returns (generate_fn, sample_rate, model_name)."""
    from pocket_tts.voice.dsm_tts import DsmTTSBackend

    backend = DsmTTSBackend(quantize=8)
    backend.load()

    condition_attrs = None
    if not voice.endswith(".wav") and not voice.endswith(".safetensors"):
        voice_key = f"{voice}.wav" if "/" in voice else f"alba-mackenna/{voice}.wav"
    else:
        voice_key = voice

    def generate(text: str, seed: int) -> np.ndarray:
        return backend.generate_audio(text, voice=voice_key)

    return generate, backend.sample_rate, "KyutAI-TTS-1.6B"


def generate_benchmark(
    backend: str = "dsm",
    use_ssml: bool = False,
    compute_wer_flag: bool = False,
    category_filter: str | None = None,
    voice: str = "alba-mackenna/casual",
):
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {backend} backend...")
    if backend == "dsm":
        generate_fn, sample_rate, model_name = _load_dsm_backend(voice)
    else:
        generate_fn, sample_rate, model_name = _load_pocket_backend(voice)

    print(f"Model: {model_name}, sample_rate={sample_rate}, voice={voice}")

    texts = BENCHMARK_TEXTS
    if category_filter:
        texts = [t for t in texts if t["category"] == category_filter]

    results = []
    category_stats = defaultdict(
        lambda: {"total": 0, "wer_sum": 0.0, "wer_count": 0, "durations": []}
    )

    total_t0 = time.monotonic()

    for i, spec in enumerate(texts):
        input_text = spec["text"]
        variant = "plain"

        if use_ssml and spec.get("ssml"):
            input_text = spec["ssml"]
            variant = "ssml"

        filename = f"{spec['id']}_{variant}_{backend}.wav"
        wav_path = BENCHMARK_DIR / filename

        t0 = time.monotonic()
        audio = generate_fn(input_text, seed=SEED + i)
        gen_ms = int((time.monotonic() - t0) * 1000)

        scipy.io.wavfile.write(str(wav_path), sample_rate, audio)

        n_samples = audio.shape[0] if hasattr(audio, "shape") else len(audio)
        duration = round(float(n_samples) / sample_rate, 3)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        sha = hashlib.sha256(np.asarray(audio).tobytes()).hexdigest()

        entry = {
            "id": spec["id"],
            "category": spec["category"],
            "difficulty": spec["difficulty"],
            "variant": variant,
            "backend": backend,
            "filename": filename,
            "text": spec["text"],
            "duration_seconds": duration,
            "rms_amplitude": round(rms, 6),
            "generation_ms": gen_ms,
            "checksum_sha256": sha,
        }

        wer_value = None
        transcript = None
        if compute_wer_flag:
            try:
                transcript, no_speech = _transcribe_with_whisper(str(wav_path))
                wer_value = _compute_wer(spec["text"], transcript)
                entry["transcript"] = transcript
                entry["wer"] = round(wer_value, 4)
                entry["no_speech_prob"] = round(no_speech, 4)
                category_stats[spec["category"]]["wer_sum"] += wer_value
                category_stats[spec["category"]]["wer_count"] += 1
            except ImportError:
                print(
                    "  [!] mlx-whisper not installed. Install: pip install mlx-whisper"
                )
                compute_wer_flag = False
            except Exception as e:
                print(f"  [!] Whisper error: {e}")
                entry["transcript_error"] = str(e)

        category_stats[spec["category"]]["total"] += 1
        category_stats[spec["category"]]["durations"].append(duration)
        results.append(entry)

        wer_str = f", WER={wer_value:.0%}" if wer_value is not None else ""
        print(
            f"  [{i + 1}/{len(texts)}] {spec['id']} ({spec['category']}, d={spec['difficulty']}) "
            f"— {duration}s, gen={gen_ms}ms{wer_str}"
        )
        if transcript:
            print(f"           ref: {spec['text'][:80]}")
            print(f"           hyp: {transcript[:80]}")

    total_s = time.monotonic() - total_t0
    total_audio = sum(r["duration_seconds"] for r in results)

    manifest = {
        "model": model_name,
        "backend": backend,
        "voice": voice,
        "variant": "ssml" if use_ssml else "plain",
        "seed_base": SEED,
        "sample_rate": sample_rate,
        "total_samples": len(results),
        "total_audio_seconds": round(total_audio, 2),
        "generation_seconds": round(total_s, 2),
        "samples": results,
    }

    manifest_path = BENCHMARK_DIR / f"manifest_{backend}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'=' * 74}")
    print(
        f"BENCHMARK RESULTS — {model_name} ({voice}, {'SSML' if use_ssml else 'plain'})"
    )
    print(f"{'=' * 74}")
    print(
        f"Total: {len(results)} samples, {total_audio:.1f}s audio, "
        f"generated in {total_s:.1f}s"
    )
    rtf = total_audio / total_s if total_s > 0 else 0
    print(f"Real-time factor: {rtf:.1f}x")
    print()

    header = f"{'Category':<25} {'N':>3} {'Avg Dur':>8}"
    if compute_wer_flag:
        header += f" {'Avg WER':>8} {'vs Base':>9}"
    print(header)
    print("-" * 74)

    overall_wer_sum = 0.0
    overall_wer_count = 0
    for cat_name in CATEGORIES:
        stats = category_stats.get(cat_name)
        if not stats or stats["total"] == 0:
            continue
        avg_dur = sum(stats["durations"]) / stats["total"]
        line = f"{cat_name:<25} {stats['total']:>3} {avg_dur:>7.2f}s"
        if compute_wer_flag and stats["wer_count"] > 0:
            avg_wer = stats["wer_sum"] / stats["wer_count"]
            delta = avg_wer - KYUTAI_TTS_BASE_WER
            sign = "+" if delta >= 0 else ""
            line += f" {avg_wer:>7.1%} {sign}{delta:>7.1%}"
            overall_wer_sum += stats["wer_sum"]
            overall_wer_count += stats["wer_count"]
        print(line)

    if compute_wer_flag and overall_wer_count > 0:
        total_wer = overall_wer_sum / overall_wer_count
        delta = total_wer - KYUTAI_TTS_BASE_WER
        sign = "+" if delta >= 0 else ""
        print("-" * 74)
        print(
            f"{'OVERALL':<25} {overall_wer_count:>3} {'':>8} "
            f"{total_wer:>7.1%} {sign}{delta:>7.1%}"
        )
        print()
        print("EmergentTTS-Eval Leaderboard Comparison:")
        print(f"  Gemini-2.5-Flash:     10.39% WER, 75.57% win-rate  (cloud)")
        print(f"  gpt-4o-audio:         11.87% WER, 72.67% win-rate  (cloud)")
        print(f"  KyutAI-TTS base:      12.72% WER, 21.94% win-rate  (on-device)")
        print(f"  Kokoro-82M:           13.41% WER, 25.89% win-rate  (on-device)")
        print(f"  pocket-tts (ours):    {total_wer:>5.2%} WER                    (on-device)")
        print()
        if total_wer < KYUTAI_TTS_BASE_WER:
            improvement = KYUTAI_TTS_BASE_WER - total_wer
            print(
                f"  >>> BEATING KyutAI-TTS base by {improvement:.2%} absolute WER <<<"
            )
            if total_wer < 0.1039:
                print(f"  >>> BEATING ALL models including cloud APIs <<<")
            elif total_wer < 0.1187:
                print(f"  >>> BEATING all on-device models AND gpt-4o-audio <<<")
            elif total_wer < 0.1341:
                print(f"  >>> BEATING all on-device models (KyutAI-TTS + Kokoro) <<<")
        else:
            gap = total_wer - KYUTAI_TTS_BASE_WER
            print(f"  Gap to KyutAI-TTS base: +{gap:.2%}")

    print(f"\nResults saved to: {manifest_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="pocket-tts benchmark suite")
    parser.add_argument(
        "--backend",
        choices=["pocket", "dsm"],
        default="dsm",
        help="TTS backend: pocket (100M, fast) or dsm (1.6B, high quality)",
    )
    parser.add_argument(
        "--ssml", action="store_true", help="Use SSML variants where available"
    )
    parser.add_argument(
        "--wer", action="store_true", help="Compute WER via mlx-whisper"
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Filter to one category"
    )
    parser.add_argument(
        "--voice", type=str, default=None, help="Voice to use (default: auto)"
    )
    args = parser.parse_args()

    if args.voice is None:
        args.voice = "alba-mackenna/casual" if args.backend == "dsm" else "alba"

    generate_benchmark(
        backend=args.backend,
        use_ssml=args.ssml,
        compute_wer_flag=args.wer,
        category_filter=args.category,
        voice=args.voice,
    )


if __name__ == "__main__":
    main()
