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
from naturalspeech2.benchmarks.dataloader.find_max_batch_sizes import generate_dummy_batch
from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import LossWrapper
from naturalspeech2.utils.utils import setup_file_logger

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def stress_test(cfg: DictConfig):
    log_file = Path(__file__).parent / "stress_test_fragmentation.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    device = cfg.training.device
    
    bucket_mapping = cfg.dataloader.bucket_mapping

    vocab_path = cfg.dataset.token_vocabulary_path
    if vocab_path is None:
        vocab_path = DATA_DIR / f"{cfg.dataset.name}_token_vocabulary.json"
        
    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=False)
    vocab_size = tokenizer.token_vocabulary_size

    logger.info("--- Initializing Model for Stress Test ---")
    from naturalspeech2.model import NaturalSpeech2Model
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
        'n_mels': cfg.model.mel.n_mels,
        'f_min': cfg.model.mel.f_min,
        'f_max': cfg.model.mel.f_max,
        'phoneme_encoder_layers': cfg.model.phoneme_encoder.transformer_layers,
        'phoneme_encoder_heads': cfg.model.phoneme_encoder.attention_heads,
        'phoneme_encoder_filter_size': cfg.model.phoneme_encoder.conv1d_filter_size,
        'phoneme_encoder_kernel_size': cfg.model.phoneme_encoder.conv1d_kernel_size,
        'phoneme_encoder_conv_dropout': cfg.model.phoneme_encoder.conv_dropout,
        'phoneme_encoder_attn_weights_dropout': cfg.model.phoneme_encoder.attn_weights_dropout,
        'phoneme_encoder_attn_out_dropout': cfg.model.phoneme_encoder.attn_out_dropout,
        'aligner_attn_channels': cfg.model.aligner.attn_channels,
        'aligner_temperature': cfg.model.aligner.temperature,
        'prior_w': cfg.model.aligner.prior_w,
        'speech_prompt_encoder_layers': cfg.model.speech_prompt_encoder.transformer_layers,
        'speech_prompt_encoder_heads': cfg.model.speech_prompt_encoder.attention_heads,
        'speech_prompt_encoder_filter_size': cfg.model.speech_prompt_encoder.conv1d_filter_size,
        'speech_prompt_encoder_kernel_size': cfg.model.speech_prompt_encoder.conv1d_kernel_size,
        'speech_prompt_encoder_conv_dropout': cfg.model.speech_prompt_encoder.conv_dropout,
        'speech_prompt_encoder_attn_weights_dropout': cfg.model.speech_prompt_encoder.attn_weights_dropout,
        'speech_prompt_encoder_attn_out_dropout': cfg.model.speech_prompt_encoder.attn_out_dropout,
        'duration_predictor_conv1d_layers': cfg.model.duration_predictor.conv1d_layers,
        'duration_predictor_conv1d_kernel_size': cfg.model.duration_predictor.conv1d_kernel_size,
        'duration_predictor_attention_layers': cfg.model.duration_predictor.attention_layers,
        'duration_predictor_attention_heads': cfg.model.duration_predictor.attention_heads,
        'duration_predictor_conv_dropout': cfg.model.duration_predictor.conv_dropout,
        'duration_predictor_attn_weights_dropout': cfg.model.duration_predictor.attn_weights_dropout,
        'duration_predictor_attn_out_dropout': cfg.model.duration_predictor.attn_out_dropout,

        'pitch_predictor_conv1d_layers': cfg.model.pitch_predictor.conv1d_layers,
        'pitch_predictor_conv1d_kernel_size': cfg.model.pitch_predictor.conv1d_kernel_size,
        'pitch_predictor_attention_layers': cfg.model.pitch_predictor.attention_layers,
        'pitch_predictor_attention_heads': cfg.model.pitch_predictor.attention_heads,
        'pitch_predictor_conv_dropout': cfg.model.pitch_predictor.conv_dropout,
        'pitch_predictor_attn_weights_dropout': cfg.model.pitch_predictor.attn_weights_dropout,
        'pitch_predictor_attn_out_dropout': cfg.model.pitch_predictor.attn_out_dropout,
    }
    
    model = NaturalSpeech2Model(**model_args).to(device)
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
            outputs = model(**batch)
            loss, _ = loss_wrapper(outputs)
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
