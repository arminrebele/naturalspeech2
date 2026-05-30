import os
import random
import logging
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch._dynamo
from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import LossWrapper
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators, generate_dummy_batch

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def stress_test(cfg: DictConfig):
    log_file = Path("research/benchmarks/stress_test_fragmentation.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    device = cfg.setup.device
    
    sorted_buckets = sorted(cfg.dataloader.bucket_mapping, key=lambda x: x.audio_length)
    enhanced_buckets = []
    min_audio_len = 1
    for b in sorted_buckets:
        enhanced_buckets.append({
            "audio_length": b.audio_length,
            "phoneme_length": b.phoneme_length,
            "batch_size": b.batch_size,
            "min_audio_samples": min_audio_len
        })
        min_audio_len = b.audio_length + 1

    vocab_path = cfg.dataset.token_vocabulary_path
    if vocab_path is None:
        vocab_path = DATA_DIR / cfg.dataset.name / "token_vocabulary.json"
        
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
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay, 
        cfg.training.learning_rate, 
        (cfg.training.beta1, cfg.training.beta2)
    )
    
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
    ).to(device)
    
    num_iterations = 500
    logger.info(f"\nStarting {num_iterations} iterations of forced shape fragmentation...")
    
    for i in tqdm(range(num_iterations)):
        # Randomly select a shape to force maximum dynamic memory allocation jumping
        bucket = random.choice(enhanced_buckets)
        
        batch = generate_dummy_batch(
            batch_size=bucket["batch_size"], 
            audio_samples=bucket["audio_length"],
            phoneme_samples=bucket["phoneme_length"],
            min_audio_samples=bucket["min_audio_samples"],
            vocab_size=vocab_size,
            device=device
        )
        
        try:
            denominators = compute_denominators([batch], cfg)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_dict = model(**batch)
                loss, _, _ = loss_wrapper(loss_dict, denominators=denominators)
                
            loss.backward()
            
            if cfg.training.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"\n❌ OOM ERROR CAUGHT AT ITERATION {i + 1}/{num_iterations}!")
                logger.error(f"Failed on Bucket Shape: {bucket['audio_length']} audio samples, {bucket['phoneme_length']} phonemes, Batch Size: {bucket['batch_size']}")
                logger.error(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
                logger.error(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
                logger.error("\n--- CUDA Memory Summary ---")
                logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
                raise e
            else:
                raise e
        
    logger.info(f"\n✅ STRESS TEST PASSED SUCCESSFULLY!")
    logger.info(f"Model survived {num_iterations} random shape jumps without memory fragmentation failure.")
    logger.info(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
    logger.info(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

    logger.info("\n========== TORCH.COMPILE STATUS ==========")
    counters = torch._dynamo.utils.counters
    graph_breaks = counters.get("graph_break", {})
    total_breaks = sum(graph_breaks.values())
    
    num_buckets = len(enhanced_buckets)
    
    logger.info(f"Total Traced Graph Breaks: {total_breaks} (Expected: {num_buckets} buckets * 1 code break = {num_buckets})")
    if total_breaks == num_buckets:
        logger.info("✅ No leaking shapes and exactly 1 intended graph break detected.")
    elif total_breaks > num_buckets:
        logger.warning("⚠️ WARNING: Too many traces! Either new graph breaks were introduced, or batch shapes are leaking.")
    logger.info("==========================================\n")

if __name__ == "__main__":
    # Enable PyTorch Memory Expansion to heavily mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    stress_test()
