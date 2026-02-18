"""Golden reference tests for deterministic, reproducible audio generation.

These tests verify that pocket-tts produces bit-identical output when given the
same seed, voice, and text. A golden reference WAV (generated with seed=42) is
stored in tests/fixtures/ and compared against fresh generation output.

To regenerate the golden reference after an intentional model or pipeline change:
    uv run python -c "
    import scipy.io.wavfile, json, hashlib
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    voice = model.get_state_for_audio_prompt('alba')
    audio = model.generate_audio(voice, 'The quick brown fox jumps over the lazy dog.', seed=42)
    scipy.io.wavfile.write('tests/fixtures/golden_reference.wav', model.sample_rate, audio)
    meta = {'seed': 42, 'voice': 'alba', 'text': 'The quick brown fox jumps over the lazy dog.',
            'sample_rate': model.sample_rate, 'num_samples': int(audio.shape[0]),
            'duration_seconds': round(float(audio.shape[0]) / model.sample_rate, 4),
            'checksum_sha256': hashlib.sha256(audio.tobytes()).hexdigest()}
    json.dump(meta, open('tests/fixtures/golden_reference.json', 'w'), indent=2)
    print(f'Regenerated: {meta[\"num_samples\"]} samples, sha256={meta[\"checksum_sha256\"]}')
    "
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from pocket_tts import TTSModel

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_golden_reference() -> tuple[dict, np.ndarray]:
    meta_path = FIXTURES_DIR / "golden_reference.json"
    wav_path = FIXTURES_DIR / "golden_reference.wav"
    assert meta_path.exists(), f"Golden reference metadata not found: {meta_path}"
    assert wav_path.exists(), f"Golden reference WAV not found: {wav_path}"

    with open(meta_path) as f:
        metadata = json.load(f)
    sample_rate, audio = scipy.io.wavfile.read(str(wav_path))
    assert sample_rate == metadata["sample_rate"]
    return metadata, audio


def test_golden_reference_integrity():
    """Verify the golden reference WAV matches its stored checksum."""
    metadata, audio = _load_golden_reference()
    actual_sha = hashlib.sha256(audio.tobytes()).hexdigest()
    assert actual_sha == metadata["checksum_sha256"], (
        f"Golden reference WAV is corrupted or was modified.\n"
        f"Expected SHA-256: {metadata['checksum_sha256']}\n"
        f"Actual SHA-256:   {actual_sha}"
    )


def test_seeded_generation_matches_golden_reference():
    """Generate audio with the same seed and verify bit-identical output."""
    metadata, golden_audio = _load_golden_reference()

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(metadata["voice"])
    audio = model.generate_audio(voice_state, metadata["text"], seed=metadata["seed"])

    assert audio.shape == golden_audio.shape, (
        f"Shape mismatch: generated {audio.shape} vs golden {golden_audio.shape}"
    )
    assert np.array_equal(audio, golden_audio), (
        f"Generated audio differs from golden reference.\n"
        f"Max absolute diff: {np.max(np.abs(audio.astype(np.float64) - golden_audio.astype(np.float64)))}\n"
        f"This means the model output has changed. If intentional, regenerate the "
        f"golden reference (see module docstring)."
    )


def test_seeded_generation_is_reproducible():
    """Two generations with the same seed must produce bit-identical output."""
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    text = "Reproducibility is the hallmark of good science."
    seed = 12345

    audio_a = model.generate_audio(voice_state, text, seed=seed)
    audio_b = model.generate_audio(voice_state, text, seed=seed)

    assert audio_a.shape == audio_b.shape, (
        f"Shape mismatch between runs: {audio_a.shape} vs {audio_b.shape}"
    )
    assert np.array_equal(audio_a, audio_b), (
        f"Same seed produced different output.\n"
        f"Max absolute diff: {np.max(np.abs(audio_a.astype(np.float64) - audio_b.astype(np.float64)))}"
    )


def test_different_seeds_produce_different_output():
    """Different seeds must produce different audio (not stuck on one output)."""
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    text = "This sentence should sound different with different seeds."

    audio_a = model.generate_audio(voice_state, text, seed=100)
    audio_b = model.generate_audio(voice_state, text, seed=200)

    assert not np.array_equal(audio_a, audio_b), (
        "Different seeds produced identical output — seeding may be broken."
    )


def test_unseeded_generation_produces_audio():
    """Generation without a seed should still produce valid audio."""
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    audio = model.generate_audio(voice_state, "Hello, world.")

    assert audio.shape[0] > 0, "Generated audio is empty"
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all(), "Generated audio contains NaN or Inf"
