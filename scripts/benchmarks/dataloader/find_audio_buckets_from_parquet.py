"""Derive the AUDIO bucket boundaries from raw parquet headers — no preprocessing (F0/phonemize) needed.

Audio bucket boundaries depend only on the audio-length distribution, which is knowable for the full
corpus straight from the audio file headers — so they can be fixed FINAL up front, before (or during) the
multi-day preprocessing, and stay correct for any train subset (a subset's lengths are a strict subrange).

Reads the same parquet the pipeline loads (HF cache; needs the parquet present, but skips decode + pyworld
F0 + phonemize), casts the audio column un-decoded, and computes each clip's resampled length from its
header alone: len = ceil(frames · target_sr / native_sr) — exactly what torchaudio.functional.resample
yields, so these lengths and the resulting boundaries match find_optimal_buckets on the same clip set.

Emits AUDIO boundaries only (DP-optimal, ×8-frame grid). Then:
  • per-bucket phoneme_length → measure_bucket_phoneme_lengths.py (reads the live subset; re-run as N grows)
  • per-bucket batch_size     → find_max_batch_sizes.py

    python scripts/benchmarks/dataloader/find_audio_buckets_from_parquet.py
"""
import io
import logging
import math
import re

import hydra
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from omegaconf import DictConfig

from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
from naturalspeech2.paths import DATA_DIR, PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger

from bucket_optimization import compute_optimal_buckets_dp, samples_to_frame_lengths

logger = logging.getLogger(__name__)

MIN_BUCKETS = 4
MAX_BUCKETS = 8


def _audio_lengths_batched(batch: dict, audio_column: str, target_sr: int) -> dict:
    """Resampled length per clip from its header (no decode). frames/samplerate via soundfile.info."""
    lengths = []
    for cell in batch[audio_column]:
        info = sf.info(io.BytesIO(cell["bytes"])) if cell.get("bytes") is not None else sf.info(cell["path"])
        # taF.resample output length == ceil(frames · target/orig) exactly; orig == target → frames.
        lengths.append(math.ceil(info.frames * target_sr / info.samplerate))
    return {"audio_length": lengths}


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    log_file = PROJECT_ROOT / "logs" / "benchmarks" / "audio_buckets_from_parquet.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    split = cfg.dataset.train_split   # buckets are train-derived; full split (subset cap ignored by design)
    target_sr = cfg.dataloader.sampling_rate
    audio_column = cfg.dataset.audio_column

    # Mirror DatasetWrapper's cache path + data_files restriction so this reuses the same HF cache (no re-download).
    clean_split = re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9]', '_', split.replace("%", "pct"))).strip('_')
    cache_dir = DATA_DIR / cfg.dataset.name / clean_split / "cache"

    logger.info(f"Loading parquet for '{cfg.dataset.name}' split '{split}' (headers only, no decode)...")
    ds = load_dataset(
        cfg.dataset.source,
        data_files={split: f"data/{split}-*.parquet"},
        split=split,
        cache_dir=str(cache_dir),
        verification_mode="no_checks",
    )
    ds = ds.cast_column(audio_column, Audio(decode=False))

    lengths_ds = ds.map(
        _audio_lengths_batched,
        batched=True,
        batch_size=1000,
        num_proc=cfg.dataloader.num_proc_pitch,
        remove_columns=ds.column_names,
        fn_kwargs={"audio_column": audio_column, "target_sr": target_sr},
        desc="Reading audio headers",
    )
    raw_lengths = np.asarray(lengths_ds["audio_length"], dtype=np.int64)

    # Mirror the pipeline's audio-length filter so the distribution matches the trained data.
    n_total = len(raw_lengths)
    keep = np.ones(n_total, dtype=bool)
    if cfg.dataset.min_audio_length is not None:
        keep &= raw_lengths >= cfg.dataset.min_audio_length
    if cfg.dataset.max_audio_length is not None:
        keep &= raw_lengths <= cfg.dataset.max_audio_length
    raw_lengths = raw_lengths[keep]

    frame_lengths = samples_to_frame_lengths(raw_lengths)
    total_original_frames = np.sum(frame_lengths)

    logger.info("\n" + "=" * 50)
    logger.info(f"Sequences: {len(frame_lengths):,} kept / {n_total:,} total (audio-length filter)")
    logger.info(f"Min frames: {np.min(frame_lengths):,} | Max frames: {np.max(frame_lengths):,} "
                f"({np.max(raw_lengths) / target_sr:.2f}s)")
    logger.info("=" * 50 + "\n")

    max_k = min(MAX_BUCKETS, len(np.unique(frame_lengths)))
    dp_results = compute_optimal_buckets_dp(frame_lengths, max_buckets=max_k)

    for K in range(MIN_BUCKETS, max_k + 1):
        buckets, min_padding = dp_results[K]
        padding_percentage = (min_padding / total_original_frames) * 100
        bucket_indices = np.searchsorted(buckets, frame_lengths)

        logger.info(f"--- K = {K} Buckets ---")
        logger.info(f"Total Padding Waste:   {padding_percentage:.2f}%")
        for i, b_frame in enumerate(buckets):
            n = int(np.sum(bucket_indices == i))
            b_sample = b_frame * ENCODER_HOP_LENGTH
            logger.info(f"  Bucket {i+1}: (audio_samples: {b_sample}, frames: {b_frame}, "
                        f"{b_sample / target_sr:.2f}s, n={n:,})")
        logger.info("-" * 25 + "\n")

    logger.info("Audio boundaries only. Next: measure_bucket_phoneme_lengths.py (phoneme_length) "
                "+ find_max_batch_sizes.py (batch_size).")


if __name__ == "__main__":
    main()
