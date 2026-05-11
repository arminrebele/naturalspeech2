import os
import random
import time
import logging
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch._dynamo
from find_max_batch_sizes import generate_dummy_batch
from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import LossWrapper
from naturalspeech2.utils.utils import setup_file_logger

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def stress_test(cfg: DictConfig):
    log_file = Path(__file__).parent / "stress_test_fragmentation.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    device = cfg.setup.device
    
    bucket_mapping = cfg.dataloader.bucket_mapping

    vocab_path = cfg.dataset.token_vocabulary_path
    if vocab_path is None:
        vocab_path = DATA_DIR / f"{cfg.dataset.name}_token_vocabulary.json"
        
    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=False)
    vocab_size = tokenizer.token_vocabulary_size

    logger.info("--- Initializing Model for Stress Test ---")
    from naturalspeech2.model import NaturalSpeech2Model
    from naturalspeech2.config.schema import model_cfg_from_omegaconf

    model_cfg = model_cfg_from_omegaconf(cfg.model)
    model = NaturalSpeech2Model(
        model_cfg,
        token_vocabulary_size=vocab_size,
        sampling_rate=cfg.dataloader.sampling_rate,
    ).to(device)
    logger.info("Compiling model (This will cache multiple graphs during the loop)...")
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
    ).to(device)
    
    num_iterations = 500
    logger.info(f"\nStarting {num_iterations} iterations of forced shape fragmentation...")
    
    start_time = time.time()
    for i in tqdm(range(num_iterations)):
        # Randomly select a shape to force maximum dynamic memory allocation jumping
        bucket = random.choice(bucket_mapping)
        
        batch = generate_dummy_batch(
            batch_size=bucket.batch_size, 
            audio_samples=bucket.audio_length,
            phoneme_samples=bucket.phoneme_length,
            vocab_size=vocab_size,
            device=device
        )
        
        try:
            optimizer.zero_grad(set_to_none=True)
            loss_dict, _ = model(**batch)
            loss, _, _ = loss_wrapper(loss_dict)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"\n❌ OOM ERROR CAUGHT AT ITERATION {i + 1}/{num_iterations}!")
                logger.error(f"Failed on Bucket Shape: {bucket.audio_length} audio samples, {bucket.phoneme_length} phonemes, Batch Size: {bucket.batch_size}")
                logger.error(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                logger.error(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                logger.error("\n--- CUDA Memory Summary ---")
                logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
                raise e
            else:
                raise e
        
    total_time = time.time() - start_time
    logger.info(f"\n✅ STRESS TEST PASSED SUCCESSFULLY!")
    logger.info(f"Model survived {num_iterations} random shape jumps without memory fragmentation failure.")
    logger.info(f"Total Time: {total_time:.2f}s | Avg Step: {(total_time/num_iterations)*1000:.1f}ms")
    logger.info(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    logger.info(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    logger.info("\n--- Torch.Compile Summary ---")
    counters = torch._dynamo.utils.counters
    
    if "stats" in counters and "unique_graphs" in counters["stats"]:
        num_graphs = counters["stats"]["unique_graphs"]
        logger.info(f"Total Unique Graphs Compiled: {num_graphs}")
        
        if num_graphs > len(bucket_mapping):
            logger.warning(f"⚠️ WARNING: Number of compiled graphs ({num_graphs}) exceeds number of buckets ({len(bucket_mapping)}). Graph breaks or dynamic value recompiles are occurring!")
        else:
            logger.info("✅ Compilation efficiency is optimal (Graphs <= Buckets).")
    else:
        logger.info(f"Dynamo Counters: {dict(counters)}")

if __name__ == "__main__":
    # Enable PyTorch Memory Expansion to heavily mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    stress_test()
