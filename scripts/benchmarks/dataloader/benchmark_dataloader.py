import time
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.utils.utils import setup_file_logger
from naturalspeech2.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

NUM_BENCHMARK_STEPS = 150
WARMUP_STEPS = 20

@hydra.main(version_base=None, config_path="../../../config", config_name="config")

def benchmark(cfg: DictConfig):
    log_file = PROJECT_ROOT / "naturalspeech2" / "benchmarks" / "dataloader_benchmark.log"
    setup_file_logger(logger, log_file, mode="a", format_str="%(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required for benchmarking.")
    device = cfg.setup.device

    logger.info("--- Initializing Dataset ---")
    dataset = DatasetWrapper(
        dataset_source=cfg.dataset.source,
        dataset_name=cfg.dataset.name,
        split=cfg.dataset.train_split,
        text_column=cfg.dataset.text_column,
        audio_column=cfg.dataset.audio_column,
        filter_column=cfg.dataset.filter_column,
        filter_substring=cfg.dataset.filter_substring,
        token_vocabulary_path=cfg.dataset.token_vocabulary_path,
        sampling_rate=cfg.dataloader.sampling_rate,
        resample_on_the_fly=cfg.dataloader.resample_on_the_fly,
        num_proc_phonemize=cfg.dataloader.num_proc_phonemize,
        num_proc_tokenize=cfg.dataloader.num_proc_tokenize,
    )

    tokenizer = PhonemeTokenizer(token_vocabulary_path=dataset.token_vocabulary_path, with_backend=False)

    bucket_mapping = OmegaConf.to_container(cfg.dataloader.bucket_mapping, resolve=True)

    sampler = DynamicBucketedBatchSampler(
        dataset,
        bucket_mapping=bucket_mapping,
        drop_last=True,
        shuffle=cfg.dataloader.shuffle
    )
    collate_fn = BucketedCollateFn(bucket_mapping=bucket_mapping)

    loader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=collate_fn,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )

    logger.info("--- Initializing Model ---")
    model_cfg = model_cfg_from_omegaconf(cfg.model)
    model = NaturalSpeech2Model(
        model_cfg,
        token_vocabulary_size=tokenizer.token_vocabulary_size,
        sampling_rate=cfg.dataloader.sampling_rate,
    ).to(device)
    
    logger.info("Compiling model...")
    model = torch.compile(model)
    
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    data_times = []
    gpu_times = []
    starved_steps = 0

    logger.info(f"\nStarting benchmark: {NUM_BENCHMARK_STEPS} steps ({WARMUP_STEPS} warmup)")
    
    loader_iter = iter(loader)
    batch = next(loader_iter)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    for i in range(NUM_BENCHMARK_STEPS + WARMUP_STEPS):

        # Time Overlapped Iteration
        start_iter = time.perf_counter()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict, _ = model(**batch)
            loss, _, _ = loss_wrapper(loss_dict, step=i)
            
        # Time DataLoader Pre-fetch
        start_data = time.perf_counter()

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        end_data = time.perf_counter()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize() # Crucial for accurate GPU timing
        end_iter = time.perf_counter()

        if i >= WARMUP_STEPS:
            dt = end_data - start_data
            it = end_iter - start_iter
            # GPU compute time is effectively the total iteration minus CPU stall time
            gt = it - dt 
            data_times.append(dt)
            gpu_times.append(gt)
            
            if dt > gt:
                starved_steps += 1

    logger.info("\n--- Benchmark Results ---")
    logger.info(f"Data Loading Time : {np.mean(data_times):.4f}s avg | P95: {np.percentile(data_times, 95):.4f}s")
    logger.info(f"GPU Process Time  : {np.mean(gpu_times):.4f}s avg | P95: {np.percentile(gpu_times, 95):.4f}s")
    logger.info(f"Starvation Rate   : {(starved_steps / NUM_BENCHMARK_STEPS) * 100:.1f}% ({starved_steps}/{NUM_BENCHMARK_STEPS} steps)")
    logger.info(f"Overall Status    : {'STARVED ❌' if np.mean(data_times) > np.mean(gpu_times) else 'HEALTHY ✅'}")

if __name__ == "__main__":
    benchmark()
