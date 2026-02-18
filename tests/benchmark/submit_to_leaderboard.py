#!/usr/bin/env python3
"""Automated EmergentTTS-Eval leaderboard submission for pocket-tts.

This script handles the full pipeline:
  1. Clones EmergentTTS-Eval (if not already present)
  2. Downloads benchmark data (1,645 test cases)
  3. Generates audio for all samples using DSM TTS 1.6B
  4. Computes WER via Whisper
  5. Optionally runs LALM judge for win-rate (requires JUDGER_API_KEY)

Usage:
    # Full run (generation + WER + win-rate with Gemini judge)
    export JUDGER_API_KEY="your-gemini-api-key"
    uv run python -m tests.benchmark.submit_to_leaderboard

    # Generation + WER only (no LALM judge)
    uv run python -m tests.benchmark.submit_to_leaderboard --no-judge

    # Resume from existing generated audio
    uv run python -m tests.benchmark.submit_to_leaderboard \\
        --fetch-audios-from ./results/pocket-tts

    # Specific categories/depths
    uv run python -m tests.benchmark.submit_to_leaderboard \\
        --categories emotions questions --depths 1 2

Estimated times (M4 Max, 8-bit quantization):
    Audio generation: ~80 min (1,645 samples × ~3s each)
    Whisper WER:      ~15 min
    LALM judging:     ~60 min (depends on Gemini API throughput)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

EMERGENT_TTS_REPO = "https://github.com/boson-ai/EmergentTTS-Eval-public"
BENCHMARK_ROOT = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_ROOT.parent.parent
EVAL_DIR = PROJECT_ROOT / "EmergentTTS-Eval-public"
RESULTS_DIR = BENCHMARK_ROOT / "results" / "leaderboard"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def setup_eval_repo():
    """Clone and set up EmergentTTS-Eval if not present."""
    if EVAL_DIR.exists():
        print(f"EmergentTTS-Eval already cloned at {EVAL_DIR}")
        return

    print("Cloning EmergentTTS-Eval...")
    _run(["git", "clone", EMERGENT_TTS_REPO, str(EVAL_DIR)])

    print("Installing EmergentTTS-Eval requirements...")
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=EVAL_DIR)

    print("Downloading benchmark data...")
    _run([sys.executable, "download_data.py"], cwd=EVAL_DIR)


def install_client():
    """Copy our client into the EmergentTTS-Eval directory."""
    src = BENCHMARK_ROOT / "emergent_tts_eval_client.py"
    dst = EVAL_DIR / "emergent_tts_eval_client.py"
    print(f"Installing pocket-tts client to {dst}")

    import shutil

    shutil.copy2(src, dst)

    runner = EVAL_DIR / "evaluation_runner.py"
    if runner.exists():
        content = runner.read_text()
        if "PocketTTSLocalClient" not in content:
            print("NOTE: You need to add PocketTTSLocalClient to evaluation_runner.py.")
            print("Add this to the client selection logic:")
            print()
            print('  from emergent_tts_eval_client import PocketTTSLocalClient')
            print('  elif "pocket-tts" in model_name:')
            print('      client = PocketTTSLocalClient(voice="alba-mackenna/casual.wav")')
            print()


def run_evaluation(
    no_judge: bool = False,
    categories: list[str] | None = None,
    depths: list[int] | None = None,
    fetch_from: str | None = None,
    voice: str = "alba-mackenna/casual.wav",
):
    """Run the full EmergentTTS-Eval pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "evaluation_runner.py",
        "--model_name_or_path",
        "pocket-tts-dsm-1.6b",
        "--output_dir",
        str(RESULTS_DIR),
        "--seed",
        "42",
        "--api_num_threads",
        "1",
        "--voice_to_use",
        voice,
    ]

    if not no_judge:
        judger_key = os.environ.get("JUDGER_API_KEY")
        if not judger_key:
            print("WARNING: JUDGER_API_KEY not set. Running without LALM judge.")
            print("Set JUDGER_API_KEY to your Gemini API key for win-rate evaluation.")
            no_judge = True
        else:
            cmd.extend([
                "--judge_model_provider",
                "gemini-2.5-pro",
                "--tts_judger_evaluate_function",
                "win_rate",
                "--baseline_audios_path",
                str(EVAL_DIR / "data" / "baseline_audios"),
            ])

    if categories:
        cmd.extend(["--categories_to_evaluate"] + categories)
    if depths:
        cmd.extend(["--depths_to_evaluate"] + [str(d) for d in depths])
    if fetch_from:
        cmd.extend(["--fetch_audios_from_path", fetch_from])

    print(f"\nRunning EmergentTTS-Eval pipeline...")
    print(f"Output: {RESULTS_DIR}")
    _run(cmd, cwd=EVAL_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Submit pocket-tts to the EmergentTTS-Eval leaderboard"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LALM judging (generation + WER only)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Evaluate specific categories only",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate specific depths only",
    )
    parser.add_argument(
        "--fetch-audios-from",
        type=str,
        default=None,
        help="Path to pre-generated audios (skip generation)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="alba-mackenna/casual.wav",
        help="DSM voice to use",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip cloning/setup of EmergentTTS-Eval",
    )
    args = parser.parse_args()

    if not args.skip_setup:
        setup_eval_repo()
        install_client()

    run_evaluation(
        no_judge=args.no_judge,
        categories=args.categories,
        depths=args.depths,
        fetch_from=args.fetch_audios_from,
        voice=args.voice,
    )


if __name__ == "__main__":
    main()
