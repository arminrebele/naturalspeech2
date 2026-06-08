"""Pre-flight check: do the per-bucket `phoneme_length` pad widths cover the data?

Companion to find_max_batch_sizes.py — together they derive the two halves of bucket_mapping in
config/dataloader/base.yaml (find_max → batch_size; this → phoneme_length).

`phoneme_length` is a HARD cap: a clip whose phoneme count exceeds its audio-bucket's value crashes
BucketedCollateFn mid-run. The train split is NOT phoneme-filtered (loaders.py clamps only dev/test),
so its per-bucket maxes must be verified against the configured caps before every training run, and
re-checked whenever the train subset grows (more clips → higher maxes).

Reuses the real loader + sampler, so the bucket assignment is IDENTICAL to training and cannot drift.
CPU-only, seconds — reads the cached `phoneme_length` column, no audio decode, no GPU. Composes the same
config a run does, so point it at the exact run you're about to launch:

    python scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py +experiment=train_5M
    python scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py dataset.max_train_clips=200000

Exits non-zero if any bucket would crash, so it can gate a workflow: `preprocess && measure && train`.
"""
import logging
import sys

import hydra
import numpy as np
from omegaconf import DictConfig

from naturalspeech2.data.loaders import create_dataloader

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    split = cfg.dataset.train_split  # the only split not auto-clamped to the largest bucket cap
    loader, dataset = create_dataloader(cfg, split, num_workers=0)
    sampler = loader.batch_sampler  # bucket_to_indices already built from audio_length
    phon = np.asarray(dataset.dataset["phoneme_length"], dtype=np.int64)

    header = f"{'bucket':>6} {'audio_ceil':>11} {'n':>8} {'cfg':>6} {'max':>6} {'p99':>6} {'#over':>6}  status"
    print(f"\nSplit '{split}': {len(phon)} clips\n{header}\n{'-' * len(header)}")

    any_bad = False
    for b_idx, indices in sorted(sampler.bucket_to_indices.items()):
        bucket = sampler.bucket_mapping[b_idx]
        cfg_p = bucket["phoneme_length"]
        if not indices:
            print(f"{b_idx:>6} {bucket['audio_length']:>11} {0:>8} {cfg_p:>6} {'-':>6} {'-':>6} {'-':>6}  (empty)")
            continue
        bp = phon[indices]
        mx, p99, n_over = int(bp.max()), int(np.percentile(bp, 99)), int((bp > cfg_p).sum())
        bad = mx > cfg_p
        any_bad |= bad
        status = f"CRASH — needs ≥ {mx}" if bad else "ok"
        print(f"{b_idx:>6} {bucket['audio_length']:>11} {len(indices):>8} {cfg_p:>6} {mx:>6} {p99:>6} {n_over:>6}  {status}")

    if any_bad:
        print("\n✗ Under-provisioned buckets above. Raise their phoneme_length to ≥ the 'max' column "
              "(mind VRAM; never trim batch_size).")
        sys.exit(1)
    print("\n✓ All buckets fit.")


if __name__ == "__main__":
    main()
