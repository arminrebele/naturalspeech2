"""Derive the (audio_length, phoneme_length) bucket boundaries from the preprocessed train split.

All-in-one: DP-optimal audio boundaries AND the paired per-bucket max phoneme count, in one pass over
the cached `audio_length`/`phoneme_length` columns. Requires the split preprocessed — builds it through
create_dataset (the SAME path training uses: chunked store + dataset.max_train_clips), so set
dataset.max_train_clips=N to derive buckets on exactly the subset you'll train on (full split = the
final boundaries). Pair with find_max_batch_sizes.py (→ batch_size) afterwards.

If the full split isn't preprocessed yet and you only want the (final, subset-invariant) AUDIO
boundaries without paying the F0/phonemize cost, use find_audio_buckets_from_parquet.py instead, then
fill phoneme_length per bucket with measure_bucket_phoneme_lengths.py.
"""
import logging

import hydra
import numpy as np
from omegaconf import DictConfig

from naturalspeech2.data.loaders import create_dataset
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
from naturalspeech2.paths import PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger

from bucket_optimization import compute_optimal_buckets_dp, samples_to_frame_lengths

logger = logging.getLogger(__name__)

MIN_BUCKETS = 4
MAX_BUCKETS = 8


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def benchmark_buckets(cfg: DictConfig):
    # File logging to save optimal buckets
    log_file = PROJECT_ROOT / "logs" / "benchmarks" / "optimal_buckets_output.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    logger.info("Initializing Dataset Pipeline (This will prepopulate the cache for training)...")
    dataset = create_dataset(cfg, cfg.dataset.train_split)

    logger.info("Extracting audio lengths...")
    # Raw audio sample lengths → latent frame lengths (model pads by frames, rounded to a ×8 grid)
    raw_lengths = np.array(dataset.dataset["audio_length"])
    phoneme_lengths = np.array(dataset.dataset["phoneme_length"])
    frame_lengths = samples_to_frame_lengths(raw_lengths)

    total_original_frames = np.sum(frame_lengths)

    logger.info("\n" + "="*50)
    logger.info(f"Total sequences: {len(frame_lengths):,}")
    logger.info(f"Min frames: {np.min(frame_lengths):,} | Max frames: {np.max(frame_lengths):,}")
    logger.info("="*50 + "\n")

    # Cap max_buckets at unique-length count
    max_k = min(MAX_BUCKETS, len(np.unique(frame_lengths)))
    dp_results = compute_optimal_buckets_dp(frame_lengths, max_buckets=max_k)

    # Test for defined bucket ranges
    for K in range(MIN_BUCKETS, max_k + 1):
        buckets, min_padding = dp_results[K]

        padding_percentage = (min_padding / total_original_frames) * 100

        # Map sequences to their assigned bucket to calculate paired phoneme lengths
        bucket_indices = np.searchsorted(buckets, frame_lengths)

        logger.info(f"--- K = {K} Buckets ---")
        logger.info(f"Total Padding Waste:   {padding_percentage:.2f}%")

        for i, b_frame in enumerate(buckets):
            # Find all sequences that fall into this bucket
            mask = (bucket_indices == i)

            # Find the absolute maximum phoneme length among these specific sequences
            max_phonemes = int(np.max(phoneme_lengths[mask])) if np.any(mask) else 0
            b_sample = b_frame * ENCODER_HOP_LENGTH

            logger.info(f"  Bucket {i+1}: (audio_samples: {b_sample}, phoneme_tokens: {max_phonemes})")

        logger.info("-" * 25 + "\n")

if __name__ == "__main__":
    benchmark_buckets()
