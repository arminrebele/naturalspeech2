import time
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import torch._dynamo
import wandb

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer

logger = logging.getLogger(__name__)

NUM_BENCHMARK_STEPS = 500
WARMUP_STEPS = 50

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def benchmark(cfg: DictConfig):

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

    if cfg.wandb.log:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    iter_times = []
    cpu_times = []
    gpu_times = []
    gpu_fwd_times = []
    gpu_h2d_times = []
    gpu_bwd_times = []
    starved_steps = 0

    logger.info(f"\nStarting benchmark: {NUM_BENCHMARK_STEPS} steps ({WARMUP_STEPS} warmup)")
    
    loader_iter = iter(loader)
    batch = next(loader_iter)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_fwd_end = torch.cuda.Event(enable_timing=True)
    gpu_h2d_start = torch.cuda.Event(enable_timing=True)
    gpu_bwd_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_BENCHMARK_STEPS + WARMUP_STEPS):
        # Snapshot Dynamo's compilation counters before the forward pass
        compile_count_before = sum(torch._dynamo.utils.counters["frames"].values())

        start_iter = time.perf_counter()

        gpu_start.record()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict, _ = model(**batch)
            loss, _, _ = loss_wrapper(loss_dict, step=i)
        gpu_fwd_end.record()
            
        looped = False
        try:
            batch = next(loader_iter)
        except StopIteration:
            logger.warning(f"Step {i}: Dataloader ran out of unique data and looped! Page cache may now artificially inflate disk I/O speeds.")
            loader_iter = iter(loader)
            batch = next(loader_iter)
            looped = True
            
        gpu_h2d_start.record()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        gpu_bwd_start.record()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gpu_end.record()

        # Capture CPU time before blocking for the GPU
        cpu_end = time.perf_counter()

        torch.cuda.synchronize(device) # Crucial for accurate GPU timing
        end_iter = time.perf_counter()

        if i >= WARMUP_STEPS:
            # Total wall-clock time for the iteration (including sync)
            it = end_iter - start_iter
            # Pure CPU dispatch + Dataloading time
            ct = cpu_end - start_iter
            
            # True GPU compute time excluding CPU idle gaps, but including PCIe H2D transfers
            gt_fwd = gpu_start.elapsed_time(gpu_fwd_end)
            gt_h2d = gpu_h2d_start.elapsed_time(gpu_bwd_start)
            gt_bwd = gpu_bwd_start.elapsed_time(gpu_end)
            gt = (gt_fwd + gt_h2d + gt_bwd) / 1000.0
            
            # Skip metrics if a loop occurred to avoid worker spin-up poisoning the P95 metrics
            if looped:
                logger.warning(f"Step {i}: Skipping metrics due to dataloader worker spin-up latency.")
                continue

            # Check if Dynamo triggered a recompilation during this step
            compile_count_after = sum(torch._dynamo.utils.counters["frames"].values())
            if compile_count_after > compile_count_before:
                logger.warning(f"Step {i}: Detected graph recompile ({it:.2f}s). Skipping metrics.")
                continue

            iter_times.append(it)
            cpu_times.append(ct)
            gpu_times.append(gt)
            gpu_fwd_times.append(gt_fwd / 1000.0)
            gpu_h2d_times.append(gt_h2d / 1000.0)
            gpu_bwd_times.append(gt_bwd / 1000.0)
            
            # Strict check: CPU dispatch + Dataloading must be faster than GPU compute.
            # Dataloader must be fast enough to completely mask Python dispatch overhead.
            is_starved = ct > gt
            if is_starved:
                starved_steps += 1
                
            if cfg.wandb.log:
                wandb.log({
                    "benchmark/total_iter_time": it,
                    "benchmark/cpu_dispatch_time": ct,
                    "benchmark/gpu_compute_time": gt,
                    "benchmark/starved_flag": 1 if is_starved else 0,
                }, step=i - WARMUP_STEPS)

    valid_steps = len(iter_times)
    logger.info("\n--- Benchmark Results ---")
    if valid_steps > 0:
        logger.info(f"Total Iter Time   : {np.mean(iter_times):.4f}s avg | P95: {np.percentile(iter_times, 95):.4f}s")
        logger.info(f"CPU Dispatch Time : {np.mean(cpu_times):.4f}s avg | P95: {np.percentile(cpu_times, 95):.4f}s")
        logger.info(f"GPU Compute Time  : {np.mean(gpu_times):.4f}s avg | P95: {np.percentile(gpu_times, 95):.4f}s")
        logger.info(f"  ├─ Forward Pass : {np.mean(gpu_fwd_times):.4f}s avg | P95: {np.percentile(gpu_fwd_times, 95):.4f}s")
        logger.info(f"  ├─ H2D Transfer : {np.mean(gpu_h2d_times):.6f}s avg | P95: {np.percentile(gpu_h2d_times, 95):.6f}s")
        logger.info(f"  └─ Backward Pass: {np.mean(gpu_bwd_times):.4f}s avg | P95: {np.percentile(gpu_bwd_times, 95):.4f}s")
        logger.info(f"Starvation Rate   : {(starved_steps / valid_steps) * 100:.1f}% ({starved_steps}/{valid_steps} steps)")
        logger.info(f"Overall Status    : {'STARVED ❌' if starved_steps > 0 else 'HEALTHY ✅'}")
        
    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    benchmark()
