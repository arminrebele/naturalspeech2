"""CLI wrapper around naturalspeech2.inference.generate_audio.

Thin entry point — all real work happens in `naturalspeech2/inference.py`.
Single-shot only (one clip per invocation); batch generation lives in
notebook / script consumers that call `generate_audio()` directly.
"""

import argparse
from datetime import datetime
from pathlib import Path

import soundfile

from naturalspeech2.inference import generate_audio, load_inference_model
from naturalspeech2.modules.encodec import SAMPLING_RATE
from naturalspeech2.paths import DATA_DIR


DEFAULT_OUTPUT_DIR = DATA_DIR / "inference_outputs"


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"inference_{timestamp}.wav"


def main():
    parser = argparse.ArgumentParser(description="Generate audio from a NaturalSpeech2 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a .safetensors checkpoint (e.g. checkpoints/ema_best.safetensors).")
    parser.add_argument("--prompt", type=Path, required=True,
                        help="Reference voice wav file (any sample rate; auto-resampled).")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize.")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output wav file path. Defaults to "
                             f"{DEFAULT_OUTPUT_DIR}/inference_<timestamp>.wav (gitignored).")
    parser.add_argument("--sampling-steps", type=int, default=150,
                        help="Number of Euler ODE steps (default: 150, matches training).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_path = args.output if args.output is not None else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_inference_model(args.checkpoint, device=args.device)
    audio_np, length = generate_audio(
        model, args.prompt, target_text=args.text,
        sampling_steps=args.sampling_steps,
    )
    soundfile.write(str(output_path), audio_np[:length], samplerate=SAMPLING_RATE)
    print(f"Wrote {output_path} ({length / SAMPLING_RATE:.2f} s)")


if __name__ == "__main__":
    main()
