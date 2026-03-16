import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from naturalspeech2.data.dataset import DatasetWrapper, custom_collate_fn, BucketedBatchSampler
from naturalspeech2.model import NaturalSpeech2Model
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer

@hydra.main(version_base=None, config_path="../config", config_name="config")
def benchmark(cfg: DictConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required for benchmarking.")
    device = cfg.training.device 

    print("--- Initializing Dataset ---")
    dataset = DatasetWrapper(
        dataset_source=cfg.dataset.source,
        dataset_name=cfg.dataset.name,
        split=cfg.dataset.split,
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

    sampler = BucketedBatchSampler(
        dataset,
        batch_size=cfg.training.batch_size,
        drop_last=True,
        shuffle=cfg.dataloader.shuffle,
        block_size_multiplier=cfg.dataloader.block_size_multiplier
    )

    loader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=custom_collate_fn,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )

    print("--- Initializing Model ---")
    # We initialize a dummy model with minimal parameters just to simulate the forward/backward pass time
    model = NaturalSpeech2Model(
        device=device,
        token_vocabulary_size=tokenizer.token_vocabulary_size,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        sampling_rate=cfg.dataloader.sampling_rate
    ).to(device)
    
    print("Compiling model...")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    num_benchmark_steps = 150
    warmup_steps = 20
    data_times = []
    gpu_times = []
    starved_steps = 0

    print(f"\nStarting benchmark: {num_benchmark_steps} steps ({warmup_steps} warmup)")
    print(f"Workers: {cfg.dataloader.num_workers} | Resample On-The-Fly: {cfg.dataloader.resample_on_the_fly}")
    
    loader_iter = iter(loader)

    for i in range(num_benchmark_steps + warmup_steps):
        # 1. Time DataLoader
        start_data = time.perf_counter()
        batch = next(loader_iter)
        end_data = time.perf_counter()

        # 2. Move to GPU
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # 3. Time GPU Processing
        start_gpu = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize() # Crucial for accurate GPU timing
        end_gpu = time.perf_counter()

        if i >= warmup_steps:
            dt = end_data - start_data
            gt = end_gpu - start_gpu
            data_times.append(dt)
            gpu_times.append(gt)
            
            if dt > gt:
                starved_steps += 1

    print("\n--- Benchmark Results ---")
    print(f"Data Loading Time : {np.mean(data_times):.4f}s avg | P95: {np.percentile(data_times, 95):.4f}s")
    print(f"GPU Process Time  : {np.mean(gpu_times):.4f}s avg | P95: {np.percentile(gpu_times, 95):.4f}s")
    print(f"Starvation Rate   : {(starved_steps / num_benchmark_steps) * 100:.1f}% ({starved_steps}/{num_benchmark_steps} steps)")
    print(f"Overall Status    : {'STARVED ❌' if np.mean(data_times) > np.mean(gpu_times) else 'HEALTHY ✅'}")

if __name__ == "__main__":
    benchmark()
