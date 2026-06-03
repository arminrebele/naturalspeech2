import os
import sys
import subprocess
from pathlib import Path
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import LossWrapper
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators, generate_dummy_batch

logger = logging.getLogger(__name__)

def worker_process(cfg: DictConfig, audio_samples: int, phoneme_samples: int, min_audio_samples: int, batch_size: int, vocab_size: int) -> None:
    
    """Isolated process that runs the model to test VRAM limits."""
    
    from naturalspeech2.model import NaturalSpeech2Model
    from naturalspeech2.config.schema import model_cfg_from_omegaconf
    device = cfg.setup.device

    try:
        model_cfg = model_cfg_from_omegaconf(cfg.model)
        model = NaturalSpeech2Model(
            model_cfg,
            token_vocabulary_size=vocab_size,
            sampling_rate=cfg.dataloader.sampling_rate,
        ).to(device)
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
        
        # 10 fwd/bwd steps → steady-state memory
        for _ in range(10):
            batch = generate_dummy_batch(batch_size, audio_samples, phoneme_samples, min_audio_samples, vocab_size, device)
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
            
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        
        # stdout → orchestrator parses
        print(f"VRAM_STATS:{peak_alloc:.3f},{peak_res:.3f}")
        sys.exit(0) # Success
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            sys.exit(1) # Expected OOM
        else:
            raise e     # Unexpected error

def test_batch_size(audio_samples: int, phoneme_samples: int, min_audio_samples: int, batch_size: int, vocab_size: int) -> tuple[bool, float, float]:
    
    """Spawns the worker script and returns (Success, Peak_Alloc_GB, Peak_Reserved_GB)."""

    logger.info(f"  Testing Batch Size {batch_size}...")
    env = os.environ.copy()
    env["IS_WORKER"] = "1"
    env["WORKER_AUDIO_SAMPLES"] = str(audio_samples)
    env["WORKER_PHONEME_SAMPLES"] = str(phoneme_samples)
    env["WORKER_MIN_AUDIO_SAMPLES"] = str(min_audio_samples)
    env["WORKER_BATCH"] = str(batch_size)
    env["WORKER_VOCAB_SIZE"] = str(vocab_size)
    
    # pass sys.argv to preserve Hydra configs
    result = subprocess.run([sys.executable, __file__] + sys.argv[1:], env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        alloc, res = 0.0, 0.0
        for line in result.stdout.splitlines():
            if line.startswith("VRAM_STATS:"):
                parts = line.replace("VRAM_STATS:", "").split(",")
                alloc, res = float(parts[0]), float(parts[1])
                break
                
        logger.info(f"  -> ✅ SUCCESS | Peak Allocated: {alloc:.2f} GB | Peak Reserved: {res:.2f} GB")
        return True, alloc, res
    elif result.returncode == 1:
        logger.info("  -> ❌ OOM")
        return False, 0.0, 0.0
    else:
        logger.error(f"\nCrash Output:\n{result.stderr}")
        raise RuntimeError("Worker crashed with unexpected error.")

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if os.environ.get("IS_WORKER") == "1":
        worker_process(
            cfg, 
            int(os.environ["WORKER_AUDIO_SAMPLES"]), 
            int(os.environ["WORKER_PHONEME_SAMPLES"]), 
            int(os.environ["WORKER_MIN_AUDIO_SAMPLES"]),
            int(os.environ["WORKER_BATCH"]),
            int(os.environ["WORKER_VOCAB_SIZE"])
        )
        return
        
    # Setup Orchestrator Logging to File
    log_file = Path("research/benchmarks/max_batch_sizes.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")
        
    vocab_path = cfg.dataset.token_vocabulary_path
    if vocab_path is None:
        vocab_path = DATA_DIR / cfg.dataset.name / "token_vocabulary.json"
        
    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=False)
    vocab_size = tokenizer.token_vocabulary_size
        
    results = []
    
    logger.info("Starting Isolated Max Batch Size Search...\n")
    
    # Sort buckets by audio length (matches dataset.py)
    sorted_buckets = sorted(cfg.dataloader.bucket_mapping, key=lambda x: x.audio_length)
    min_audio_len = 1

    for bucket in sorted_buckets:
        audio_len = bucket.audio_length
        phoneme_len = bucket.phoneme_length
        
        logger.info(f"--- Searching for Bucket: {audio_len} audio samples / {phoneme_len} phonemes ({(audio_len/cfg.dataloader.sampling_rate):.2f}s) ---")
        
        # Phase 1: Exponential search to find the upper bound
        bs = 2
        last_alloc, last_res = 0.0, 0.0
        while True:
            success, alloc, res = test_batch_size(audio_len, phoneme_len, min_audio_len, bs, vocab_size)
            if success:
                last_alloc, last_res = alloc, res
                bs *= 2
            else:
                break
            
        # Phase 2: Binary search between bs//2 and bs
        low = bs // 2
        high = bs - 1
        max_stable_bs = low if last_alloc > 0.0 else 0
        max_alloc, max_res = last_alloc, last_res # Default to last successful exp search result
        
        while low <= high:
            mid = (low + high) // 2
            success, alloc, res = test_batch_size(audio_len, phoneme_len, min_audio_len, mid, vocab_size)
            if success:
                max_stable_bs = mid
                max_alloc, max_res = alloc, res
                low = mid + 1
            else:
                high = mid - 1

        if max_stable_bs == 0:
            logger.error(f"\n❌ CRITICAL: Bucket {audio_len} is too large. OOM even at batch size 1!")
            raise RuntimeError(f"Bucket {audio_len} fails at batch_size 1. Reduce bucket sizes or model size.")
                
        # Safety margin: back off ~10% (≥1) for fragmentation/optimizer spikes
        safe_bs = max(1, int(max_stable_bs * 0.90))
        
        results.append({
            "audio_length": audio_len,
            "phoneme_length": phoneme_len,
            "batch_size": safe_bs
        })
        
        logger.info(f"\n✅ Absolute Max Batch Size: {max_stable_bs}")
        logger.info(f"   -> Peak Allocated VRAM: {max_alloc:.2f} GB")
        logger.info(f"   -> Peak Reserved VRAM:  {max_res:.2f} GB (Fragmentation Overhead: {max_res - max_alloc:.2f} GB)")
        logger.info(f"🟢 Recommended Safe Limit (10% Margin): {safe_bs}\n")
        
        # Next bucket's lower bound = this bucket's max + 1
        min_audio_len = audio_len + 1
        
    logger.info("="*50)
    logger.info("FINAL RECOMMENDED BUCKET MAPPING (Paste this into default.yaml):")
    logger.info("bucket_mapping:")
    for res in results:
        logger.info(f"  - {{audio_length: {res['audio_length']}, phoneme_length: {res['phoneme_length']}, batch_size: {res['batch_size']}}}")
    logger.info("="*50)

if __name__ == "__main__":
    # Memory expansion → mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
