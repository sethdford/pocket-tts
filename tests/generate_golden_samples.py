#!/usr/bin/env python3
"""Regenerate golden reference samples for quality testing.

Run this after an intentional model or pipeline change:
    uv run python tests/generate_golden_samples.py

Produces:
  tests/fixtures/golden_samples/*.wav       — One WAV per voice+text combination
  tests/fixtures/golden_samples/manifest.json — Metadata, checksums, and parameters
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from pocket_tts import TTSModel

GOLDEN_SAMPLES_DIR = Path(__file__).parent / "fixtures" / "golden_samples"

SAMPLES = [
    # One sample per voice — standard pangram
    {"voice": "alba", "seed": 100, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "marius", "seed": 101, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "javert", "seed": 102, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "jean", "seed": 103, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "fantine", "seed": 104, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "cosette", "seed": 105, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "eponine", "seed": 106, "text": "The quick brown fox jumps over the lazy dog."},
    {"voice": "azelma", "seed": 107, "text": "The quick brown fox jumps over the lazy dog."},
    # Challenge texts
    {"voice": "alba", "seed": 200, "text": "She sells seashells by the seashore."},
    {
        "voice": "alba",
        "seed": 201,
        "text": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    },
    {"voice": "marius", "seed": 202, "text": "To be or not to be, that is the question."},
    {
        "voice": "jean",
        "seed": 203,
        "text": "I have a dream that one day this nation will rise up "
        "and live out the true meaning of its creed.",
    },
]


def main():
    GOLDEN_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model()
    manifest = {"sample_rate": model.sample_rate, "samples": []}
    total_t0 = time.monotonic()

    for i, spec in enumerate(SAMPLES):
        voice_state = model.get_state_for_audio_prompt(spec["voice"])
        t0 = time.monotonic()
        audio = model.generate_audio(voice_state, spec["text"], seed=spec["seed"])
        gen_ms = int((time.monotonic() - t0) * 1000)

        filename = f"{spec['voice']}_{spec['seed']}.wav"
        wav_path = GOLDEN_SAMPLES_DIR / filename
        scipy.io.wavfile.write(str(wav_path), model.sample_rate, audio)

        sha = hashlib.sha256(audio.tobytes()).hexdigest()
        duration = round(float(audio.shape[0]) / model.sample_rate, 3)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

        entry = {
            "filename": filename,
            "voice": spec["voice"],
            "seed": spec["seed"],
            "text": spec["text"],
            "num_samples": int(audio.shape[0]),
            "duration_seconds": duration,
            "rms_amplitude": round(rms, 6),
            "checksum_sha256": sha,
        }
        manifest["samples"].append(entry)
        print(f"  [{i + 1}/{len(SAMPLES)}] {filename}: {duration}s, rms={rms:.4f}, gen={gen_ms}ms")

    manifest_path = GOLDEN_SAMPLES_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_s = time.monotonic() - total_t0
    total_audio = sum(s["duration_seconds"] for s in manifest["samples"])
    print(f"\nGenerated {len(SAMPLES)} golden samples in {total_s:.1f}s")
    print(f"Total audio: {total_audio:.1f}s")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
