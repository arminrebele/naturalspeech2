"""Compute per-channel log-mel statistics for aligner-input normalization.

Run once before training (analogous to compute_encodec_latent_stats.py). The aligner
consumes the dB-scale log-mel; raw, it carries a large per-channel spectral-envelope
offset (a frame-invariant common-mode that carries ~zero alignment information) plus
~3-4x per-channel std heterogeneity. The aligner's RMSNorm cannot remove the offset
(no mean subtraction), and it violates the squared-L2 attention's Xavier-init O(1)
zero-mean assumption. Standardizing per-channel (x - mu_c)/sigma_c fixes both — the
per-channel form of RAD-TTS's mel centering ((mel + 5.5)/2), matching the encodec/pitch
stats recipe.

Mel frames are cheap (STFT + mel + dB, no model/Encodec), so this runs fast on CPU/GPU.

Example:
    # in container
    python scripts/compute_mel_stats.py \
        --dataset data/mls_eng/train/chunks_otf/chunk_00000 \
        --num-clips 2048
"""
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk

from naturalspeech2.modules.log_mel_spectrogram import LogMelSpectrogramGenerator
from naturalspeech2.data.dataset import _decode_audio_to_target_sr
from naturalspeech2.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="data/mls_eng/train/chunks_otf/chunk_00000",
        help="Preprocessed HF dataset (load_from_disk). A train chunk is a shuffled, speaker-mixed sample.",
    )
    parser.add_argument(
        "--num-clips", type=int, default=2048,
        help="Clips to encode. 2048 clips ≈ millions of frames — plenty for a stable per-channel mean/std.",
    )
    parser.add_argument(
        "--output", default="models/mel_logmel_stats.pt",
        help="Output .pt path (relative to repo root or absolute).",
    )
    parser.add_argument("--sampling-rate", type=int, default=24000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument(
        "--min-std-ratio", type=float, default=0.0,
        help="Floor per-channel std to this fraction of the median std (0 = no floor). Use e.g. 0.1 to "
             "cap a floor-noise channel's amplification at ~10× the median (see the dead-channel warning).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print(f"Loading dataset: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    print(f"  rows available: {len(ds)}")

    # Bootstrap: identity normalization (stats_path=None) so the generator returns raw dB-mel.
    print("Loading LogMelSpectrogramGenerator (bootstrap — no normalization)...")
    gen = LogMelSpectrogramGenerator(
        sampling_rate=args.sampling_rate, n_fft=args.n_fft, n_mels=args.n_mels,
    ).to(args.device)
    gen.eval()

    # Incremental per-channel accumulation over VALID frames (float64 sums; pad frames excluded).
    n_mels = args.n_mels
    s = torch.zeros(n_mels, dtype=torch.float64)
    s2 = torch.zeros(n_mels, dtype=torch.float64)
    n_frames = 0
    n_done = 0
    idx = 0
    print(f"Encoding {args.num_clips} clips at sr={args.sampling_rate}...")
    while n_done < args.num_clips and idx < len(ds):
        row = ds[idx]
        idx += 1
        try:
            audio = _decode_audio_to_target_sr(row["audio"], args.sampling_rate)
        except Exception as e:
            print(f"  skip row {idx-1}: {e}")
            continue
        if audio.shape[0] < args.sampling_rate:  # drop <1s clips
            continue
        audio_t = torch.from_numpy(audio).float().to(args.device).unsqueeze(0)
        audio_lens = torch.tensor([audio.shape[0]], device=args.device)
        with torch.no_grad():
            enc, _mask, lens = gen(audio_t, audio_lens)   # enc [1, F, n_mels], lens [1]
        F_valid = int(lens.item())
        v = enc[0, :F_valid].double().cpu()               # [F_valid, n_mels]
        s += v.sum(dim=0)
        s2 += (v * v).sum(dim=0)
        n_frames += F_valid
        n_done += 1
        if n_done % 256 == 0:
            print(f"  done {n_done}/{args.num_clips}, {n_frames:,} frames")

    assert n_frames > 0, "no valid frames found"
    mean = (s / n_frames).float()                          # [n_mels]
    var = (s2 / n_frames - (s / n_frames) ** 2).clamp_min(0.0)
    std = var.sqrt().float()                               # [n_mels]
    assert std.min() > 0, f"per-channel std must be strictly positive; got min={std.min()}"

    # Optional std floor: (x - mu_c)/sigma_c amplifies a tiny-std channel by 1/sigma_c, injecting
    # floor noise into the aligner. Floor to a fraction of the median std to cap that amplification.
    if args.min_std_ratio > 0:
        floor = float(args.min_std_ratio * std.median())
        n_floored = int((std < floor).sum())
        std = std.clamp(min=floor)
        print(f"Floored {n_floored} channel(s) to std={floor:.3f} (--min-std-ratio {args.min_std_ratio})")

    print(f"\nEncoded {n_done} clips, {n_frames:,} total frames × {n_mels} channels")
    print(f"Per-channel summary (dB log-mel):")
    print(f"  mean range  : [{mean.min():+.2f}, {mean.max():+.2f}]  (envelope spread {mean.max()-mean.min():.1f} dB)")
    print(f"  std  range  : [{std.min():.2f}, {std.max():.2f}], mean-of-stds = {std.mean():.2f}")
    print(f"  std max/min : {(std.max() / std.min()):.2f}x  (>2x → per-channel beats single-scalar)")

    # Surface dead/near-floor channels and tell the operator exactly how to fix it.
    median_std = std.median()
    dead = torch.nonzero(std < 0.1 * median_std).flatten().tolist()
    if dead:
        amp = float((median_std / std[dead].clamp_min(1e-9)).max())
        print(f"  WARNING: {len(dead)} channel(s) have std < 0.1× median (idx {dead}); per-channel scaling "
              f"will amplify them ~{amp:.0f}× vs a median channel. If these are floor-noise (not fricative "
              f"signal), re-run with `--min-std-ratio 0.1` to floor them (caps amplification at ~10×); "
              f"otherwise leave as-is. The saved stats are usable either way.")
    else:
        print("  OK: no near-dead channels (all std ≥ 0.1× median) — no flooring needed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "std": std,
            "n_clips": n_done,
            "n_frames": n_frames,
            "sampling_rate": args.sampling_rate,
            "n_fft": args.n_fft,
            "n_mels": n_mels,
        },
        output_path,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
