"""Shared bucket-boundary math for the dataloader benchmark scripts.

Imported by find_optimal_buckets.py (audio + phoneme, off the preprocessed split) and
find_audio_buckets_from_parquet.py (audio only, off raw parquet headers) so the DP and the
samples→frame conversion can't drift between the two entry points.
"""
import logging

import numpy as np

from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH

logger = logging.getLogger(__name__)

# Latent-frame grid the runtime sampler/collate bucket on: round up to a multiple of 8 frames
# (Tensor-Core friendly). Boundaries emitted as (multiple-of-8 frames) × hop, so the sampler's
# raw-sample assignment and this frame assignment land every clip in the SAME bucket.
FRAME_RESOLUTION = 8


def samples_to_frame_lengths(raw_lengths: np.ndarray, resolution: int = FRAME_RESOLUTION) -> np.ndarray:
    """Raw audio samples → latent-frame lengths, rounded up to a multiple of `resolution`."""
    base_frame_lengths = np.ceil(raw_lengths / ENCODER_HOP_LENGTH).astype(int)
    return np.ceil(base_frame_lengths / resolution).astype(int) * resolution


def compute_optimal_buckets_dp(lengths: np.ndarray, max_buckets: int):
    """DP for optimal bucket sizes that minimize total padding waste."""
    # Group lengths to speed up DP
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    M = len(unique_lengths)

    dp = np.full((M + 1, max_buckets + 1), float('inf'))
    choice = np.zeros((M + 1, max_buckets + 1), dtype=int)

    dp[0][0] = 0

    # Precompute prefix sums for O(1) cost calculation
    count_prefix = np.zeros(M + 1)
    sum_prefix = np.zeros(M + 1)
    for i in range(M):
        count_prefix[i + 1] = count_prefix[i] + counts[i]
        sum_prefix[i + 1] = sum_prefix[i] + counts[i] * unique_lengths[i]

    def cost(j, i):
        """Cost of grouping unique lengths from index j to i-1 into a bucket of size unique_lengths[i-1]"""
        bucket_size = unique_lengths[i - 1]
        num_items = count_prefix[i] - count_prefix[j]
        items_sum = sum_prefix[i] - sum_prefix[j]
        return (num_items * bucket_size) - items_sum

    # DP Execution
    logger.info(f"Running Dynamic Programming optimization for up to {max_buckets} buckets over {M} unique lengths...")
    for k in range(1, max_buckets + 1):
        for i in range(1, M + 1):
            # Find the optimal starting point 'j' for this bucket
            for j in range(k - 1, i):
                c = cost(j, i)
                if dp[j][k - 1] + c < dp[i][k]:
                    dp[i][k] = dp[j][k - 1] + c
                    choice[i][k] = j

    # Backtrack: optimal boundaries for all bucket counts
    results = {}
    for k_target in range(1, max_buckets + 1):
        curr_idx = M
        buckets = []
        for k in range(k_target, 0, -1):
            buckets.append(unique_lengths[curr_idx - 1])
            curr_idx = choice[curr_idx][k]

        buckets.reverse()
        min_padding = dp[M][k_target]
        results[k_target] = (buckets, min_padding)

    return results
