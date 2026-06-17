"""Allocator / memory-fragmentation stress test for the bucket batch sizes.

Cycles random per-bucket dummy batches through a real fwd+bwd+opt step (torch.compiled, like training)
so back-to-back shape jumps exercise the CUDA allocator — surfacing an OOM deep into training that the
per-bucket sizing (find_max_batch_sizes) missed. Catches the OOM, reports the failing bucket + peak
VRAM + memory summary, then re-raises; on success reports peak VRAM over the run. GPU-only.

For compile-mode A/B + shape-leak (graph) analysis, see benchmark_compile_modes.py.
"""
import os
import random
import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from naturalspeech2.paths import DATA_DIR, PROJECT_ROOT
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import LossWrapper, NaturalSpeech2Model
from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators, generate_dummy_batch

logger = logging.getLogger(__name__)

NUM_ITERATIONS = 500


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def stress_test(cfg: DictConfig):
    log_file = PROJECT_ROOT / "logs" / "benchmarks" / "stress_test_fragmentation.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required for the fragmentation stress test.")
    device = cfg.setup.device

    # Buckets sorted by audio_length, each tagged with the min audio samples that route to it.
    enhanced_buckets, min_audio_len = [], 1
    for b in sorted(cfg.dataloader.bucket_mapping, key=lambda x: x.audio_length):
        enhanced_buckets.append({"audio_length": b.audio_length, "phoneme_length": b.phoneme_length,
                                 "batch_size": b.batch_size, "min_audio_samples": min_audio_len})
        min_audio_len = b.audio_length + 1

    vocab_path = cfg.dataset.token_vocabulary_path or (DATA_DIR / cfg.dataset.name / "token_vocabulary.json")
    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=False)
    vocab_size = tokenizer.token_vocabulary_size

    logger.info("--- Initializing Model for Stress Test ---")
    model = NaturalSpeech2Model(
        model_cfg_from_omegaconf(cfg.model),
        token_vocabulary_size=vocab_size,
        sampling_rate=cfg.dataloader.sampling_rate,
    ).to(device)

    logger.info("Compiling model (This will cache multiple graphs during the loop)...")
    model = torch.compile(model)
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay,
        cfg.training.learning_rate,
        (cfg.training.beta1, cfg.training.beta2),
    )
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True),
        loss_warmup_hold_steps=OmegaConf.to_container(cfg.model.loss_warmup_hold_steps, resolve=True),
    ).to(device)

    logger.info(f"\nStarting {NUM_ITERATIONS} iterations of forced shape fragmentation...")
    for i in tqdm(range(NUM_ITERATIONS)):
        # Random shape → force max dynamic-allocation jumping.
        bucket = random.choice(enhanced_buckets)
        batch = generate_dummy_batch(
            batch_size=bucket["batch_size"],
            audio_samples=bucket["audio_length"],
            phoneme_samples=bucket["phoneme_length"],
            min_audio_samples=bucket["min_audio_samples"],
            vocab_size=vocab_size,
            device=device,
        )

        try:
            denominators = compute_denominators([batch], cfg)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_dict = model(**batch)
                loss, _, _ = loss_wrapper(loss_dict, denominators=denominators)
            loss.backward()
            if cfg.training.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.error(f"\n❌ OOM ERROR CAUGHT AT ITERATION {i + 1}/{NUM_ITERATIONS}!")
            logger.error(f"Failed on Bucket Shape: {bucket['audio_length']} audio samples, "
                         f"{bucket['phoneme_length']} phonemes, Batch Size: {bucket['batch_size']}")
            logger.error(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
            logger.error(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            logger.error("\n--- CUDA Memory Summary ---")
            logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
            raise

    logger.info("\n✅ STRESS TEST PASSED SUCCESSFULLY!")
    logger.info(f"Model survived {NUM_ITERATIONS} random shape jumps without memory fragmentation failure.")
    logger.info(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
    logger.info(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == "__main__":
    # Memory expansion → mitigate fragmentation across bucket-size jumps.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    stress_test()
