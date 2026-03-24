import os
import sys
import subprocess
from pathlib import Path
import logging
import torch
import hydra
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.utils.utils import LossWrapper

logger = logging.getLogger(__name__)

def generate_dummy_batch(
    batch_size: int, 
    audio_samples: int, 
    phoneme_samples: int, 
    vocab_size: int,
    device: str
) -> dict[str, torch.Tensor]:
    
    """Generates dummy tensors representing perfectly bucketed sequences."""

    # Simulate real-world variation within the bucket to test torch.compile dynamics
    audio_lengths = torch.randint(max(1, audio_samples // 2), audio_samples + 1, (batch_size,), device=device)
    # Force at least one sequence to hit the max bucket boundary
    audio_lengths[0] = audio_samples
    idx_a = rearrange(torch.arange(audio_samples, device=device), 't -> 1 t')
    audio_mask_2d = idx_a < rearrange(audio_lengths, 'b -> b 1')
    
    # Generate audio and apply zero-padding outside valid lengths
    audio = torch.randn(batch_size, audio_samples, device=device)
    audio = audio.masked_fill(~audio_mask_2d, 0.0)
    
    audio_mask = rearrange(audio_mask_2d, 'b t -> b t 1')
    
    phoneme_tokens_lengths = torch.randint(max(1, phoneme_samples // 2), phoneme_samples + 1, (batch_size,), device=device)
    phoneme_tokens_lengths[0] = phoneme_samples
    idx_p = rearrange(torch.arange(phoneme_samples, device=device), 't -> 1 t')
    phoneme_tokens_mask_2d = idx_p < rearrange(phoneme_tokens_lengths, 'b -> b 1')
    
    # Generate tokens and apply zero-padding outside valid lengths (matching pad_token_id=0)
    phoneme_tokens = torch.randint(0, vocab_size, (batch_size, phoneme_samples), device=device)
    phoneme_tokens = phoneme_tokens.masked_fill(~phoneme_tokens_mask_2d, 0)
    
    phoneme_tokens_mask = rearrange(phoneme_tokens_mask_2d, 'b t -> b t 1')
    
    return {
        "audio": audio, 
        "audio_mask": audio_mask, 
        "audio_lengths": audio_lengths,
        "phoneme_tokens": phoneme_tokens, 
        "phoneme_tokens_mask": phoneme_tokens_mask, 
        "phoneme_tokens_lengths": phoneme_tokens_lengths,
    }

def worker_process(cfg: DictConfig, audio_samples: int, phoneme_samples: int, batch_size: int, vocab_size: int) -> None:
    
    """The isolated process that runs the actual model to test VRAM limits."""
    
    from naturalspeech2.model import NaturalSpeech2Model
    device = cfg.training.device
    
    # Mirror train.py parameters
    model_args = {
        'device': device, 
        'token_vocabulary_size': vocab_size,
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
    
    try:
        model = NaturalSpeech2Model(**model_args).to(device)
        model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        
        loss_wrapper = LossWrapper(
            loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
            loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
        ).to(device)
        
        # Perform 5 forward/backward steps to ensure steady-state memory allocation
        for _ in range(5):
            batch = generate_dummy_batch(batch_size, audio_samples, phoneme_samples, vocab_size, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss, _ = loss_wrapper(outputs)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        
        # Print stats to stdout so the orchestrator can parse them
        print(f"VRAM_STATS:{peak_alloc:.3f},{peak_res:.3f}")
        sys.exit(0) # Success
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            sys.exit(1) # Expected OOM
        else:
            raise e     # Unexpected error

def test_batch_size(audio_samples: int, phoneme_samples: int, batch_size: int, vocab_size: int) -> tuple[bool, float, float]:
    
    """Spawns the worker script and returns (Success, Peak_Alloc_GB, Peak_Reserved_GB)."""

    logger.info(f"  Testing Batch Size {batch_size}...")
    env = os.environ.copy()
    env["IS_WORKER"] = "1"
    env["WORKER_AUDIO_SAMPLES"] = str(audio_samples)
    env["WORKER_PHONEME_SAMPLES"] = str(phoneme_samples)
    env["WORKER_BATCH"] = str(batch_size)
    env["WORKER_VOCAB_SIZE"] = str(vocab_size)
    
    # We pass the original sys.argv to preserve Hydra configs
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

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if os.environ.get("IS_WORKER") == "1":
        worker_process(
            cfg, 
            int(os.environ["WORKER_AUDIO_SAMPLES"]), 
            int(os.environ["WORKER_PHONEME_SAMPLES"]), 
            int(os.environ["WORKER_BATCH"]),
            int(os.environ["WORKER_VOCAB_SIZE"])
        )
        return
        
    # Setup Orchestrator Logging to File
    log_file = Path(__file__).parent / "max_batch_sizes_output.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
        
    vocab_path = cfg.dataset.token_vocabulary_path
    if vocab_path is None:
        vocab_path = DATA_DIR / f"{cfg.dataset.name}_token_vocabulary.json"
        
    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=False)
    vocab_size = tokenizer.token_vocabulary_size
        
    results = []
    
    logger.info("Starting Isolated Max Batch Size Search...\n")
    for bucket in cfg.dataloader.bucket_mapping:
        audio_len = bucket.audio_length
        phoneme_len = bucket.phoneme_length
        
        logger.info(f"--- Searching for Bucket: {audio_len} audio samples / {phoneme_len} phonemes ({(audio_len/cfg.dataloader.sampling_rate):.2f}s) ---")
        
        # Phase 1: Exponential search to find the upper bound
        bs = 2
        last_alloc, last_res = 0.0, 0.0
        while True:
            success, alloc, res = test_batch_size(audio_len, phoneme_len, bs, vocab_size)
            if success:
                last_alloc, last_res = alloc, res
                bs *= 2
            else:
                break
            
        # Phase 2: Binary search to find the exact limit between (bs//2) and (bs)
        low = bs // 2
        high = bs - 1
        max_stable_bs = low if last_alloc > 0.0 else 0
        max_alloc, max_res = last_alloc, last_res # Default to last successful exp search result
        
        while low <= high:
            mid = (low + high) // 2
            success, alloc, res = test_batch_size(audio_len, phoneme_len, mid, vocab_size)
            if success:
                max_stable_bs = mid
                max_alloc, max_res = alloc, res
                low = mid + 1
            else:
                high = mid - 1

        if max_stable_bs == 0:
            logger.error(f"\n❌ CRITICAL: Bucket {audio_len} is too large. OOM even at batch size 1!")
            raise RuntimeError(f"Bucket {audio_len} fails at batch_size 1. Reduce bucket sizes or model size.")
                
        # Safety Margin: Back off by ~10% (at least 1) to leave room for fragmentation/optimizer spikes
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
        
    logger.info("="*50)
    logger.info("FINAL RECOMMENDED BUCKET MAPPING (Paste this into default.yaml):")
    logger.info("bucket_mapping:")
    for res in results:
        logger.info(f"  - {{audio_length: {res['audio_length']}, phoneme_length: {res['phoneme_length']}, batch_size: {res['batch_size']}}}")
    logger.info("="*50)

if __name__ == "__main__":
    # Enable PyTorch Memory Expansion to heavily mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
