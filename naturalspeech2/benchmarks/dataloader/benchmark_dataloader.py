import time
import logging
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.utils.utils import setup_file_logger

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def benchmark(cfg: DictConfig):
    log_file = Path(__file__).parent / "benchmark_dataloader.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required for benchmarking.")
    device = cfg.training.device 

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
    # Mirror train.py parameters exactly to guarantee realistic GPU timing for the bottleneck calculation
    model_args = {
        'device': device,
        'token_vocabulary_size': tokenizer.token_vocabulary_size,
        'hidden_dim': cfg.model.hidden_dim,
        'latent_dim': cfg.model.latent_dim,
        'sampling_rate': cfg.dataloader.sampling_rate,
        'rope_base': cfg.model.rope_base,
        'rope_max_seq_len': cfg.model.rope_max_seq_len,
        'min_prompt_pct': cfg.model.min_prompt_pct,
        'max_prompt_pct': cfg.model.max_prompt_pct,

        'n_fft': cfg.model.mel.n_fft,
        'hop_length': cfg.model.mel.hop_length,
        'n_mels': cfg.model.mel.n_mels,
        'f_min': cfg.model.mel.f_min,
        'f_max': cfg.model.mel.f_max,

        'phoneme_encoder_layers': cfg.model.phoneme_encoder.transformer_layers,
        'phoneme_encoder_heads': cfg.model.phoneme_encoder.attention_heads,
        'phoneme_encoder_filter_size': cfg.model.phoneme_encoder.conv1d_filter_size,
        'phoneme_encoder_kernel_size': cfg.model.phoneme_encoder.conv1d_kernel_size,
        'phoneme_encoder_dropout': cfg.model.phoneme_encoder.dropout,

        'aligner_attn_channels': cfg.model.aligner.attn_channels,
        'aligner_temperature': cfg.model.aligner.temperature,
        'prior_w': cfg.model.aligner.prior_w,

        'speech_prompt_encoder_layers': cfg.model.speech_prompt_encoder.transformer_layers,
        'speech_prompt_encoder_heads': cfg.model.speech_prompt_encoder.attention_heads,
        'speech_prompt_encoder_filter_size': cfg.model.speech_prompt_encoder.conv1d_filter_size,
        'speech_prompt_encoder_kernel_size': cfg.model.speech_prompt_encoder.conv1d_kernel_size,
        'speech_prompt_encoder_dropout': cfg.model.speech_prompt_encoder.dropout,

        'duration_predictor_conv1d_layers': cfg.model.duration_predictor.conv1d_layers,
        'duration_predictor_conv1d_kernel_size': cfg.model.duration_predictor.conv1d_kernel_size,
        'duration_predictor_attention_layers': cfg.model.duration_predictor.attention_layers,
        'duration_predictor_attention_heads': cfg.model.duration_predictor.attention_heads,
        'duration_predictor_dropout': cfg.model.duration_predictor.dropout,
    }
    model = NaturalSpeech2Model(**model_args).to(device)
    
    logger.info("Compiling model...")
    model = torch.compile(model)
    
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    num_benchmark_steps = 150
    warmup_steps = 20
    data_times = []
    gpu_times = []
    starved_steps = 0

    logger.info(f"\nStarting benchmark: {num_benchmark_steps} steps ({warmup_steps} warmup)")
    logger.info(f"Workers: {cfg.dataloader.num_workers} | Resample On-The-Fly: {cfg.dataloader.resample_on_the_fly}")
    
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
        loss, _ = loss_wrapper(outputs)
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

    logger.info("\n--- Benchmark Results ---")
    logger.info(f"Data Loading Time : {np.mean(data_times):.4f}s avg | P95: {np.percentile(data_times, 95):.4f}s")
    logger.info(f"GPU Process Time  : {np.mean(gpu_times):.4f}s avg | P95: {np.percentile(gpu_times, 95):.4f}s")
    logger.info(f"Starvation Rate   : {(starved_steps / num_benchmark_steps) * 100:.1f}% ({starved_steps}/{num_benchmark_steps} steps)")
    logger.info(f"Overall Status    : {'STARVED ❌' if np.mean(data_times) > np.mean(gpu_times) else 'HEALTHY ✅'}")

if __name__ == "__main__":
    benchmark()
