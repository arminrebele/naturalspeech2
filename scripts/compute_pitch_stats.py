"""Compute log-F0 statistics (mean/std over voiced frames) for pitch-condition normalization.

Run once before training (analogous to compute_encodec_latent_stats.py). The diffusion
condition adds a projected pitch term to the phoneme encodings; feeding raw Hz makes that
term scale ~150x and dominate the content. Normalizing log-F0 to ~zero-mean/unit-std fixes
the scale, removes the speaker-F0-dependent magnitude, and (with an explicit voiced flag)
the voiced/unvoiced discontinuity.

Pitch is already precomputed in the cache (the `f0` column), so this reads it directly —
no model, no GPU, no audio decode. A few thousand clips give a very stable mean/std.

Example:
    # in container
    python scripts/compute_pitch_stats.py \
        --dataset data/mls_eng/train/chunks_otf/chunk_00000 \
        --num-clips 8192
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from naturalspeech2.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="data/mls_eng/train/chunks_otf/chunk_00000",
        help="Preprocessed HF dataset (load_from_disk). A train chunk is a shuffled, speaker-mixed sample.",
    )
    parser.add_argument(
        "--num-clips", type=int, default=8192,
        help="Clips to read f0 from. f0 is cached (cheap); 8192 clips ≈ millions of voiced frames — plenty.",
    )
    parser.add_argument(
        "--output", default="models/pitch_logf0_stats.pt",
        help="Output .pt path (relative to repo root or absolute).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print(f"Loading dataset: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    n = min(args.num_clips, len(ds))
    print(f"  rows available: {len(ds)}; reading f0 from {n} clips")

    # Incremental accumulation over voiced (f0>0) frames; raw-Hz stats too (for context).
    s_log = s2_log = 0.0
    s_hz = s2_hz = 0.0
    n_voiced = 0
    n_frames_total = 0
    for i in range(n):
        f0 = np.asarray(ds[i]["f0"], dtype=np.float64)
        n_frames_total += f0.size
        v = f0[f0 > 0]
        if v.size == 0:
            continue
        lf = np.log(v)
        s_log += lf.sum(); s2_log += (lf * lf).sum()
        s_hz += v.sum(); s2_hz += (v * v).sum()
        n_voiced += v.size
        if (i + 1) % 1024 == 0:
            print(f"  {i+1}/{n} clips, {n_voiced:,} voiced frames")

    assert n_voiced > 0, "no voiced frames found"
    mean = s_log / n_voiced
    var = s2_log / n_voiced - mean * mean
    std = math.sqrt(max(var, 0.0))
    hz_mean = s_hz / n_voiced
    hz_std = math.sqrt(max(s2_hz / n_voiced - hz_mean * hz_mean, 0.0))

    print(f"\n==== log-F0 stats over {n_voiced:,} voiced frames ({n} clips) ====")
    print(f"voiced fraction       : {100*n_voiced/n_frames_total:.1f}%")
    print(f"raw F0 (Hz)           : mean {hz_mean:.2f}  std {hz_std:.2f}")
    print(f"log-F0 mean (μ)       : {mean:.6f}   (≈ {math.exp(mean):.2f} Hz)")
    print(f"log-F0 std  (σ)       : {std:.6f}")
    # The model loads these as buffers from the saved .pt via model.pitch_stats_path (below) —
    # no config scalars to paste; point pitch_stats_path at this file (default already does).

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logf0_mean": float(mean),
            "logf0_std": float(std),
            "hz_mean": float(hz_mean),
            "hz_std": float(hz_std),
            "n_clips": n,
            "n_voiced_frames": n_voiced,
            "dataset": str(dataset_path),
        },
        output_path,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
