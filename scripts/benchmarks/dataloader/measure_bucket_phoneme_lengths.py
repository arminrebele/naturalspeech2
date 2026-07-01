"""Find out what to paste for the per-bucket `phoneme_length` of bucket_mapping (and verify it).

The phoneme-side analogue of find_optimal_buckets.py's phoneme column, but read off the split you're
about to train through the REAL sampler (so the bucket assignment is identical to training), instead
of re-deriving the audio boundaries. Use it after fixing the audio boundaries
(find_audio_buckets_from_parquet.py or find_optimal_buckets.py).

Two jobs at once:
  • DERIVE — the `max` column (echoed in the RECOMMENDED block at the end) is the value to paste: the
    per-bucket phoneme maximum = the minimum safe cap. phoneme_length is OPTIONAL when you run this, so
    after deriving audio-only boundaries you can run it straight away, read the numbers, and paste them.
  • VERIFY / GATE — if phoneme_length is already set, it flags any bucket the data would overflow and
    exits non-zero, so it can gate `preprocess && measure && train`.

phoneme_length is a HARD cap: a clip with more phonemes than its audio-bucket's value crashes
BucketedCollateFn. The train split is NOT phoneme-filtered (loaders.py clamps only dev/test) and its
per-bucket maxima only GROW as the subset grows — so re-run this whenever max_train_clips increases.
(Training self-guards too: the sampler fails loud at startup on a mismatch. This is the pre-flight
that hands you the numbers before you get there.)

CPU-only, seconds — reads the cached `phoneme_length` column, no audio decode, no GPU. Composes the
same config a run does, so point it at the exact run you're about to launch:

    python scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py +experiment=5M
    python scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py dataset.max_train_clips=200000
"""
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.data.dataset import DynamicBucketedBatchSampler
from naturalspeech2.data.loaders import create_dataset


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    split = cfg.dataset.train_split  # the only split not auto-clamped to the largest bucket cap
    dataset = create_dataset(cfg, split)

    # Real sampler → bucket assignment identical to training. phoneme_length may be unset (derive mode);
    # batch_size doesn't affect bucket_to_indices, so inject a placeholder when find_max hasn't run yet.
    raw_mapping = OmegaConf.to_container(cfg.dataloader.bucket_mapping, resolve=True)
    have_batch_size = all("batch_size" in b for b in raw_mapping)
    mapping = [dict(b) for b in raw_mapping]
    for b in mapping:
        b.setdefault("batch_size", 1)
    sampler = DynamicBucketedBatchSampler(dataset, bucket_mapping=mapping, shuffle=False,
                                          validate_phoneme_caps=False)
    phon = np.asarray(dataset.dataset["phoneme_length"], dtype=np.int64)

    header = f"{'bucket':>6} {'audio_length':>12} {'n':>8} {'p99':>6} {'max':>6} {'config':>7}  status"
    print(f"\nSplit '{split}': {len(phon)} clips  —  `max` is the value to set as phoneme_length\n"
          f"{header}\n{'-' * len(header)}")

    any_bad = any_unset = False
    recommended = []   # (audio_length, phoneme_length to paste, batch_size or None)
    for b_idx, indices in sorted(sampler.bucket_to_indices.items()):
        bucket = sampler.bucket_mapping[b_idx]
        cfg_p = bucket.get("phoneme_length")
        bs = bucket["batch_size"] if have_batch_size else None
        if not indices:
            # No clips here for this subset; keep current cap (or a placeholder) — re-run if it fills.
            print(f"{b_idx:>6} {bucket['audio_length']:>12} {0:>8} {'-':>6} {'-':>6} "
                  f"{(cfg_p if cfg_p is not None else '-'):>7}  (empty)")
            recommended.append((bucket['audio_length'], cfg_p if cfg_p is not None else 1, bs))
            continue
        bp = phon[indices]
        mx, p99 = int(bp.max()), int(np.percentile(bp, 99))
        recommended.append((bucket['audio_length'], mx, bs))
        if cfg_p is None:
            any_unset = True
            status, cfg_str = "unset → set to max", "-"
        elif mx > cfg_p:
            any_bad = True
            status, cfg_str = f"TOO LOW — set >= {mx}", str(cfg_p)
        else:
            status, cfg_str = "ok", str(cfg_p)
        print(f"{b_idx:>6} {bucket['audio_length']:>12} {len(indices):>8} {p99:>6} {mx:>6} {cfg_str:>7}  {status}")

    # Ready-to-paste recommendation: each bucket's max = its minimum safe phoneme_length.
    print("\nRECOMMENDED phoneme_length — paste into config/dataloader bucket_mapping:")
    print("bucket_mapping:")
    for audio_length, rec_p, bs in recommended:
        bs_str = f", batch_size: {bs}" if bs is not None else ""
        print(f"  - {{audio_length: {audio_length}, phoneme_length: {rec_p}{bs_str}}}")
    if not have_batch_size:
        print("  # + batch_size per bucket from find_max_batch_sizes.py")

    if any_bad:
        print("\n✗ Under-provisioned buckets above. Set each phoneme_length to >= its `max` "
              "(mind VRAM; never trim batch_size — re-run find_max if a cap jump forces it).")
        sys.exit(1)
    if any_unset:
        print("\nℹ phoneme_length unset for some buckets — paste the recommended values, then re-run to verify.")
        return
    print("\n✓ All buckets fit.")


if __name__ == "__main__":
    main()
