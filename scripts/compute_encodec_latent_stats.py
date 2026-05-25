"""Compute per-channel Encodec latent statistics for diffusion-target normalization.

Runs once before first training: walks a preprocessed HF dataset, encodes N clips
through `EncodecWrapper.get_latents` (with normalization disabled, i.e. raw output),
computes per-channel mean and std, saves to a `.pt` file that `EncodecWrapper`
loads at construction time and registers as persistent buffers.

Example:
    # in container
    python scripts/compute_encodec_latent_stats.py \
        --dataset data/mls_eng/dev/processed_otf \
        --output models/encodec_24khz/encodec_latent_stats.pt \
        --num-clips 1024
"""
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk

from naturalspeech2.modules.encodec import EncodecWrapper, LATENT_DIM
from naturalspeech2.data.dataset import _decode_audio_to_target_sr
from naturalspeech2.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="data/mls_eng/dev/processed_otf",
        help="Path (relative to repo root or absolute) to a preprocessed HF dataset loadable via `load_from_disk`.",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=1024,
        help="Number of clips to encode. 1024 gives ~36M per-channel samples — plenty for stable mean/std.",
    )
    parser.add_argument(
        "--output",
        default="models/encodec_24khz/encodec_latent_stats.pt",
        help="Output .pt path (relative to repo root or absolute).",
    )
    parser.add_argument("--bandwidth", type=int, default=24, help="Encodec bandwidth (kbps).")
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

    # Bootstrap mode: pass latent_stats_path=None so the encoder returns raw latents.
    print("Loading EncodecWrapper (bootstrap — no normalization)...")
    encoder = EncodecWrapper(bandwidth=args.bandwidth, latent_stats_path=None).to(args.device)
    encoder.eval()

    sr = 24000
    print(f"Encoding {args.num_clips} clips at sr={sr}...")
    chunks: list[torch.Tensor] = []
    n_done = 0
    idx = 0
    while n_done < args.num_clips and idx < len(ds):
        row = ds[idx]
        idx += 1
        try:
            audio = _decode_audio_to_target_sr(row["audio"], sr)
        except Exception as e:
            print(f"  skip row {idx-1}: {e}")
            continue
        if audio.shape[0] < sr:  # drop <1s clips
            continue
        audio_t = torch.from_numpy(audio).float().to(args.device).unsqueeze(0)
        audio_lens = torch.tensor([audio.shape[0]], device=args.device)
        with torch.no_grad():
            latents, lat_lens, _codes = encoder.get_latents(audio_t, audio_lens)
        F_valid = int(lat_lens.item())
        chunks.append(latents[0, :F_valid].cpu())
        n_done += 1
        if n_done % 128 == 0:
            print(f"  done {n_done}/{args.num_clips}")

    z = torch.cat(chunks, dim=0)  # [F_total, latent_dim]
    F_total, D = z.shape
    assert D == LATENT_DIM, f"expected D={LATENT_DIM}, got {D}"
    print(f"\nEncoded {n_done} clips, {F_total} total frames × {D} channels")

    mean = z.mean(dim=0).float()                # [latent_dim]
    std = z.std(dim=0, unbiased=False).float()  # [latent_dim]
    assert std.min() > 0, f"per-channel std must be strictly positive; got min={std.min()}"

    # Sanity: post-normalization per-scalar var should be ~1.0
    normalized = (z - mean) / std
    print(f"\nPost-normalization sanity check (should be ~0 / ~1):")
    print(f"  per-scalar mean = {normalized.mean().item():+.6f}")
    print(f"  per-scalar var  = {normalized.var(unbiased=False).item():.6f}")

    print(f"\nPer-channel summary:")
    print(f"  mean range  : [{mean.min():+.4f}, {mean.max():+.4f}], mean-of-means = {mean.mean():+.4f}")
    print(f"  std  range  : [{std.min():.4f}, {std.max():.4f}], mean-of-stds = {std.mean():.4f}")
    print(f"  std max/min : {(std.max() / std.min()):.2f}x  (>2x → per-channel beats single-scalar)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "std": std,
            "n_clips": n_done,
            "n_frames": F_total,
            "bandwidth_kbps": args.bandwidth,
        },
        output_path,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
