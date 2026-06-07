import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import logging

from naturalspeech2.data.dataset import DatasetWrapper
from naturalspeech2.utils.utils import setup_file_logger
from naturalspeech2.paths import PROJECT_ROOT
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH

logger = logging.getLogger(__name__)

MIN_BUCKETS = 4
MAX_BUCKETS = 8

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

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def benchmark_buckets(cfg: DictConfig):
    # File logging to save optimal buckets
    log_file = PROJECT_ROOT / "logs" / "benchmarks" / "optimal_buckets_output.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    logger.info("Initializing Dataset Pipeline (This will prepopulate the cache for training)...")
    dataset = DatasetWrapper(
        dataset_source=cfg.dataset.source,
        dataset_name=cfg.dataset.name,
        split=cfg.dataset.train_split,
        text_column=cfg.dataset.text_column,
        audio_column=cfg.dataset.audio_column,
        filter_column=cfg.dataset.filter_column,
        filter_substring=cfg.dataset.filter_substring,
        token_vocabulary_path=cfg.dataset.token_vocabulary_path,
        min_audio_length=cfg.dataset.min_audio_length,
        max_audio_length=cfg.dataset.max_audio_length,
        min_phoneme_length=cfg.dataset.min_phoneme_length,
        max_phoneme_length=cfg.dataset.max_phoneme_length,
        sampling_rate=cfg.dataloader.sampling_rate,
        resample_on_the_fly=cfg.dataloader.resample_on_the_fly,
        num_proc_pitch=cfg.dataloader.num_proc_pitch,
        num_proc_phonemize=cfg.dataloader.num_proc_phonemize,
        num_proc_tokenize=cfg.dataloader.num_proc_tokenize,
    )
    
    logger.info("Extracting audio lengths...")
    # Raw audio sample lengths → latent frame lengths (model pads by frames)
    raw_lengths = np.array(dataset.dataset["audio_length"])
    phoneme_lengths = np.array(dataset.dataset["phoneme_length"])
    base_frame_lengths = np.ceil(raw_lengths / ENCODER_HOP_LENGTH).astype(int)
    
    # Round up to the nearest multiple of 8 (Tensor Core optimization).
    resolution = 8
    frame_lengths = np.ceil(base_frame_lengths / resolution).astype(int) * resolution
    
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
