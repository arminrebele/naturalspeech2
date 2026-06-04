"""Dataloader construction shared by the trainer and the decoupled eval daemon.

Bucketed batching: non-train splits widen their length caps to the largest train bucket (the
bucket mapping is derived from the train split). num_workers override lets the daemon run a
lighter loader than training."""
import logging

from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler

logger = logging.getLogger(__name__)


def create_dataloader(cfg, split: str, token_vocabulary_path: str = None, num_workers: int = None,
                      batch_size_divisor: int = 1):
    bucket_mapping = OmegaConf.to_container(cfg.dataloader.bucket_mapping, resolve=True)
    if batch_size_divisor > 1:
        # Daemon eval only: shrink the batch dimension (lower forward VRAM) WITHOUT touching the
        # per-bucket (audio_length, phoneme_length) pad targets — the collate keys on max length, not
        # batch_size, so padded/compiled shapes are unchanged. grad_accum compensates (run_decoupled_eval).
        bucket_mapping = [{**b, "batch_size": max(1, b["batch_size"] // batch_size_divisor)}
                          for b in bucket_mapping]

    max_audio_length = cfg.dataset.max_audio_length
    max_phoneme_length = cfg.dataset.max_phoneme_length

    if split != cfg.dataset.train_split:    # bucket mapping is derived from train split
        largest_bucket = max(bucket_mapping, key=lambda x: x['audio_length'])
        max_audio_length = largest_bucket['audio_length']
        max_phoneme_length = largest_bucket['phoneme_length']
        logger.info(f"Overriding upper boundaries for '{split}' split to match max bucket: "
                    f"audio={max_audio_length}, phonemes={max_phoneme_length}")

    dataset = DatasetWrapper(
        dataset_source=cfg.dataset.source,
        dataset_name=cfg.dataset.name,
        split=split,
        text_column=cfg.dataset.text_column,
        audio_column=cfg.dataset.audio_column,
        filter_column=cfg.dataset.filter_column,
        filter_substring=cfg.dataset.filter_substring,
        token_vocabulary_path=token_vocabulary_path,
        min_audio_length=cfg.dataset.min_audio_length,
        max_audio_length=max_audio_length,
        min_phoneme_length=cfg.dataset.min_phoneme_length,
        max_phoneme_length=max_phoneme_length,
        sampling_rate=cfg.dataloader.sampling_rate,
        resample_on_the_fly=cfg.dataloader.resample_on_the_fly,
        num_proc_pitch=cfg.dataloader.num_proc_pitch,
        num_proc_phonemize=cfg.dataloader.num_proc_phonemize,
        num_proc_tokenize=cfg.dataloader.num_proc_tokenize,
        # Cap ONLY the train split, applied pre-preprocessing in DatasetWrapper (only N clips pitch-extracted).
        max_train_clips=cfg.dataset.max_train_clips if split == cfg.dataset.train_split else None,
        subset_seed=cfg.seed,
        delete_raw_cache_after_preprocess=cfg.dataset.delete_raw_cache_after_preprocess,
    )

    sampler = DynamicBucketedBatchSampler(
        dataset,
        bucket_mapping=bucket_mapping,
        drop_last=cfg.dataloader.drop_last,
        shuffle=cfg.dataloader.shuffle
    )
    collate_fn = BucketedCollateFn(bucket_mapping=bucket_mapping)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.dataloader.num_workers if num_workers is None else num_workers,
        pin_memory=True
    )
    return loader, dataset
