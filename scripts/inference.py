"""CLI wrapper around naturalspeech2.inference.generate_audio.

Thin entry point — real work is in naturalspeech2/inference.py. Single-shot only
(one clip per call); batch generation lives in consumers that call generate_audio() directly.
"""

import argparse
from datetime import datetime
from pathlib import Path

import soundfile

from naturalspeech2.inference import MAX_SECONDS_PER_PHONEME, generate_audio, load_inference_model
from naturalspeech2.modules.encodec import SAMPLING_RATE
from naturalspeech2.paths import DATA_DIR
from naturalspeech2.utils.warning_filters import install_warning_filters


DEFAULT_OUTPUT_DIR = DATA_DIR / "inference_outputs"


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"inference_{timestamp}.wav"


def main():
    install_warning_filters()   # silence known-benign phonemizer / torch spam for clean CLI output
    parser = argparse.ArgumentParser(description="Generate audio from a NaturalSpeech2 checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Local .safetensors path (e.g. checkpoints/main_training/ema_best.safetensors) "
                             "or a Hugging Face repo id (downloaded on demand).")
    parser.add_argument("--prompt", type=Path, required=True,
                        help="Reference voice wav file (any sample rate; auto-resampled).")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize.")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output wav file path. Defaults to "
                             f"{DEFAULT_OUTPUT_DIR}/inference_<timestamp>.wav (gitignored).")
    parser.add_argument("--sampling-steps", type=int, default=150,
                        help="Number of Euler ODE steps (default: 150, matches training).")
    parser.add_argument("--prompt-seconds", type=float, default=None,
                        help="Slice the reference to this many leading seconds (default: full clip).")
    parser.add_argument("--max-seconds-per-phoneme", type=float, default=MAX_SECONDS_PER_PHONEME,
                        help=f"Per-phoneme duration cap guarding against runaway synthesis "
                             f"(default: {MAX_SECONDS_PER_PHONEME}s).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_path = args.output if args.output is not None else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_inference_model(args.checkpoint, device=args.device)
    audio_np, length = generate_audio(
        model, args.prompt, target_text=args.text,
        prompt_seconds=args.prompt_seconds,
        sampling_steps=args.sampling_steps,
        max_seconds_per_phoneme=args.max_seconds_per_phoneme,
    )
    soundfile.write(str(output_path), audio_np[:length], samplerate=SAMPLING_RATE)
    print(f"Wrote {output_path} ({length / SAMPLING_RATE:.2f} s)")


if __name__ == "__main__":
    main()
