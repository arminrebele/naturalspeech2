import os
import time
import math
import random
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

import torch
from torch.utils.data import DataLoader
import torch._dynamo
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_model
from einops import rearrange

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper, GradientAnalyzer
from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.paths import CHECKPOINTS_DIR, PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger

logger = logging.getLogger(__name__)

COMPILE_MILESTONES = [1, 250]

def get_lr(it, cfg):
    learning_rate = cfg.training.learning_rate
    warmup_iters = cfg.setup.warmup_iters
    schedule = cfg.training.lr_schedule

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
        
    elif schedule == "isr":
        assert warmup_iters > 0, "Inverse square root schedule requires warmup_iters > 0"
        decay_factor = math.sqrt(warmup_iters / it)
        return learning_rate * decay_factor
        
    elif schedule == "cosine":
        lr_decay_iters = cfg.setup.lr_decay_iters
        min_lr = cfg.training.min_lr
        
        if it > lr_decay_iters:
            return min_lr
        
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
        
    else:
        raise ValueError(f"Unknown lr_schedule: {schedule}")

def get_infinite_batches(loader, device, start_epoch=0, start_batch_idx=0, overfit_single_batch=False):
    """Continuously yields batches while tracking and setting dataloader state for instant resuming."""
    epoch = start_epoch
    sampler = loader.batch_sampler
    sampler.set_epoch(epoch)
    sampler.set_start_batch_idx(start_batch_idx)

    if overfit_single_batch:
        logger.info("OVERFIT TEST ACTIVE: Yielding the exact same batch endlessly.")
        batch = next(iter(loader))
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        
        while True:
            yield batch, epoch, 0

    while True:
        for batch_idx, batch in enumerate(loader, start=sampler.start_batch_idx):
            # Move immediately to device asynchronously
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)
            yield batch, epoch, batch_idx

        # Epoch finished
        epoch += 1
        sampler.set_epoch(epoch)
        sampler.set_start_batch_idx(0)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, loss_wrapper, eval_iters, device):
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        loader_iter = iter(loader)
        # Tensor-side accumulation across eval_iters; one .item() per split at the end
        # so the eval loop doesn't sync the GPU on every iteration.
        total_loss_sum = torch.zeros((), device=device)
        log_dict_sums = {}

        for k in range(eval_iters):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            for k_b, v in batch.items():
                batch[k_b] = v.to(device, non_blocking=True)

            with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                loss_dict = model(**batch)
                total_loss, logged_losses, _ = loss_wrapper(loss_dict)

            total_loss_sum = total_loss_sum + total_loss.detach()
            for key, val in logged_losses.items():
                log_dict_sums[key] = log_dict_sums.get(key, 0.0) + val

        out[split] = {
            'total_loss': (total_loss_sum / eval_iters).item(),
            'logged_losses': {key: (val / eval_iters).item() for key, val in log_dict_sums.items()},
        }
    model.train()
    return out

def create_dataloader(cfg, split: str, token_vocabulary_path: str = None):
    dataset = DatasetWrapper(
        dataset_source=cfg.dataset.source,
        dataset_name=cfg.dataset.name,
        split=split,
        text_column=cfg.dataset.text_column,
        audio_column=cfg.dataset.audio_column,
        filter_column=cfg.dataset.filter_column,
        filter_substring=cfg.dataset.filter_substring,
        token_vocabulary_path=token_vocabulary_path,
        sampling_rate=cfg.dataloader.sampling_rate,
        resample_on_the_fly=cfg.dataloader.resample_on_the_fly,
        num_proc_pitch=cfg.dataloader.num_proc_pitch,
        num_proc_phonemize=cfg.dataloader.num_proc_phonemize,
        num_proc_tokenize=cfg.dataloader.num_proc_tokenize,
    )
    bucket_mapping = OmegaConf.to_container(cfg.dataloader.bucket_mapping, resolve=True)
    sampler = DynamicBucketedBatchSampler(
        dataset,
        bucket_mapping=bucket_mapping,
        drop_last=cfg.dataloader.drop_last,
        shuffle=cfg.dataloader.shuffle if split == cfg.dataset.train_split else False
    )
    collate_fn = BucketedCollateFn(bucket_mapping=bucket_mapping)
    loader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=collate_fn,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )
    return loader, dataset

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires an NVIDIA GPU and CUDA installed, but none were detected.")
    
    device = cfg.setup.device
    device_type = 'cuda'
    
    # Create checkpoints directory and setup specific logs
    if cfg.setup.loss_analysis_run:
        log_dir = PROJECT_ROOT / "research"
        log_name = "loss_analysis.log"
    elif cfg.setup.gradient_analysis_run:
        log_dir = PROJECT_ROOT / "research"
        log_name = "gradient_analysis.log"
    elif cfg.setup.overfit_single_batch:
        log_dir = PROJECT_ROOT / "research"
        log_name = "overfit_test.log"
    else:
        log_dir = CHECKPOINTS_DIR
        log_name = "main_training.log"
        
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logger(logger, log_dir / log_name)
    
    logger.info("Initializing DataLoaders...")
    train_loader, train_dataset = create_dataloader(cfg, cfg.dataset.train_split, cfg.dataset.token_vocabulary_path)
    val_loader, val_dataset = create_dataloader(cfg, cfg.dataset.val_split, train_dataset.token_vocabulary_path)

    tokenizer = PhonemeTokenizer(token_vocabulary_path=train_dataset.token_vocabulary_path, with_backend=False)
    token_vocabulary_size = tokenizer.token_vocabulary_size

    # Setup static generation prompts for evaluation
    logger.info("Initializing custom text prompts for generation testing...")
    phonemizer = PhonemizerWrapper()
    
    custom_prompts = [
        "Hello, world! This is a test.", # Short (~3s)
        "The quick brown fox jumps over the lazy dog, while the sun sets.", # Medium (~6s)
        ("Natural speech synthesis has come a long way in recent years. "
         "Today, we can generate highly realistic human voices from just a "
         "few seconds of reference audio, opening up new possibilities for "
         "accessibility and content creation."), # Long (~15s)
        ("In the early days of artificial intelligence, text to speech systems "
         "sounded incredibly robotic and lacked emotional nuance. Researchers "
         "spent decades studying human phonetics, prosody, and intonation. "
         "Now, thanks to advanced deep learning techniques, diffusion models, "
         "and massive datasets, the boundaries between synthesized and natural "
         "voices are becoming indistinguishable. This marks a paradigm shift "
         "in how we interact with technology on a daily basis.") # Very long (~30s)
    ]
    
    custom_prompt_tokens = []
    for prompt in custom_prompts:
        phonemes = phonemizer(prompt)
        tokens = tokenizer(phonemes)
        custom_prompt_tokens.append(tokens)

    sampling_rate = cfg.dataloader.sampling_rate
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)

    # State initialization variables
    start_iter = 0
    best_val_loss = 1e9
    start_epoch = 0
    start_batch_idx = 0

    # Instantiate Model
    if cfg.setup.init_from == 'scratch':
        logger.info("Initializing a new model from scratch...")
        model_cfg = model_cfg_from_omegaconf(cfg.model)
        model = NaturalSpeech2Model(
            model_cfg,
            token_vocabulary_size=token_vocabulary_size,
            sampling_rate=sampling_rate,
        )
    elif cfg.setup.init_from == 'resume':
        logger.info(f"Resuming training from checkpoint in {CHECKPOINTS_DIR}...")
        ckpt_path = CHECKPOINTS_DIR / 'ckpt.pt'
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Rebuild from the cfg the checkpoint was trained with so the
        # architecture matches exactly even if base.yaml has since changed.
        ckpt_cfg = model_cfg_from_omegaconf(checkpoint['model_cfg'])
        model = NaturalSpeech2Model(
            ckpt_cfg,
            token_vocabulary_size=checkpoint['token_vocabulary_size'],
            sampling_rate=checkpoint['sampling_rate'],
        )

        state_dict = checkpoint['model']

        model.load_state_dict(state_dict)
        start_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint.get('epoch', 0)

        # Subsequent checkpoints must save the cfg the model was actually built with,
        # not the (possibly drifted) Hydra cfg captured above on resume.
        model_cfg_dict = checkpoint['model_cfg']
        start_batch_idx = checkpoint.get('batch_idx', 0) # Already points to the next batch due to pre-fetch

    model.to(device)
    
    loss_weights_dict = OmegaConf.to_container(cfg.model.loss_weights, resolve=True)
    loss_warmup_steps_dict = OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)

    if cfg.setup.loss_analysis_run:
        logger.info("LOSS ANALYSIS RUN: Forcing all dynamic loss weights to 1.0 and warmups to 0.")
        def override_dict(d, val):
            for k, v in d.items():
                if isinstance(v, dict):
                    override_dict(v, val)
                else:
                    d[k] = val
        override_dict(loss_weights_dict, 1.0)
        override_dict(loss_warmup_steps_dict, 0)
            
    loss_wrapper = LossWrapper(
        loss_weights=loss_weights_dict,
        loss_warmup_steps=loss_warmup_steps_dict
    ).to(device)
    
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay, 
        cfg.training.learning_rate, 
        (cfg.training.beta1, cfg.training.beta2)
    )
    
    if cfg.setup.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_wandb_id = checkpoint.get('wandb_id')
        logger.info("Resumed optimizer from checkpoint.")
    else:
        resume_wandb_id = None
        
    # Free memory
    checkpoint = None
    
    logger.info("Compiling the model... (this takes a minute)")
    unoptimized_model = model
    model = torch.compile(model)

    if cfg.wandb.log:
        wandb.init(
            project=cfg.wandb.project, 
            name=cfg.wandb.run_name, 
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True), 
            id=resume_wandb_id, 
            resume="allow" if resume_wandb_id else None
        )

    batch_generator = get_infinite_batches(train_loader, device, start_epoch, start_batch_idx, cfg.setup.overfit_single_batch)
    batch, current_epoch, current_batch_idx = next(batch_generator)
    
    loss_analysis_accumulators = {}
    t0 = time.perf_counter()
    logger.info("Starting training loop...")
    for iter_num in range(start_iter, cfg.setup.max_iters):
        
        # Apply LR scheduling
        lr = get_lr(iter_num, cfg) if cfg.training.decay_lr else cfg.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # -----------------------------
        # Evaluation & Checkpointing
        # -----------------------------
        if iter_num % cfg.setup.eval_interval == 0 and cfg.setup.save_checkpoint:
            losses = estimate_loss(model, train_loader, val_loader, loss_wrapper, cfg.setup.eval_iters, device)
            logger.info(f"Step {iter_num}: train loss {losses['train']['total_loss']:.4f}, val loss {losses['val']['total_loss']:.4f}")
            
            if cfg.wandb.log:
                eval_payload = {
                    "eval/iter": iter_num,
                    "eval/lr": lr,
                    "eval/train/total_loss": losses['train']['total_loss'],
                    "eval/val/total_loss": losses['val']['total_loss'],
                }
                
                for k, v in losses['train']['logged_losses'].items():
                    eval_payload[f"eval/train/losses/{k}"] = v
                for k, v in losses['val']['logged_losses'].items():
                    eval_payload[f"eval/val/losses/{k}"] = v

                # --- Generation Testing ---
                logger.info("Generating audio samples for evaluation...")

                # estimate_loss() flips back to train mode before returning; bracket the
                # sampling loop in eval mode so dropout (prompt encoder, predictors,
                # WaveNet blocks) doesn't perturb inference.
                unoptimized_model.eval()
                wandb_audios = []
                num_gen_samples = min(4, len(val_dataset))
                test_indices = random.sample(range(len(val_dataset)), num_gen_samples)

                for i, idx in enumerate(test_indices):
                    sample = val_dataset[idx]
                    tokens = custom_prompt_tokens[i % len(custom_prompt_tokens)]

                    # Add batch dimension
                    ref_audio = rearrange(sample["audio"], 't -> 1 t').to(device)
                    ref_audio_len = torch.tensor([sample["audio_length"]]).to(device)

                    ph_tokens = rearrange(torch.tensor(tokens), 'p -> 1 p').to(device)
                    ph_tokens_len = torch.tensor([len(tokens)]).to(device)
                    ph_tokens_mask = torch.ones((1, len(tokens), 1), dtype=torch.bool, device=device)

                    gen_kwargs = {
                        "reference_audio": ref_audio,
                        "reference_audio_lengths": ref_audio_len,
                        "phoneme_tokens": ph_tokens,
                        "phoneme_tokens_mask": ph_tokens_mask,
                        "phoneme_tokens_lengths": ph_tokens_len
                    }

                    # Unbatched generation (Batch size 1) — audio_lengths unused since B=1
                    generated_audio, _ = unoptimized_model.generate(**gen_kwargs)

                    audio_np = generated_audio[0].cpu().to(torch.float32).numpy()
                    wandb_audios.append(
                        wandb.Audio(audio_np, sample_rate=cfg.dataloader.sampling_rate, caption=f"Gen Sample {i}")
                    )
                unoptimized_model.train()

                eval_payload["eval/generated_samples"] = wandb_audios
                
                wandb.log(eval_payload)
                
            if iter_num > 0 and losses['val']['total_loss'] < best_val_loss:
                best_val_loss = losses['val']['total_loss']
                logger.info(f"Saving new best model to {CHECKPOINTS_DIR} (Atomic Save)")
                
                best_path = CHECKPOINTS_DIR / 'ckpt_best.safetensors'
                best_tmp_path = CHECKPOINTS_DIR / 'ckpt_best.tmp.safetensors'
                best_bak_path = CHECKPOINTS_DIR / 'ckpt_best_bak.safetensors'
                
                # 1. Save to a temporary file
                save_model(unoptimized_model, best_tmp_path)
                # 2. Backup the previous best file if it exists
                if best_path.exists():
                    best_path.replace(best_bak_path)
                # 3. Rename temp file to final destination (atomic)
                best_tmp_path.replace(best_path)

        # -----------------------------
        # Forward & Backward Pass
        # -----------------------------
        grad_accum_steps = cfg.setup.gradient_accumulation_steps
        
        accum_loss = torch.zeros((), device=device)
        accum_logged_losses = {}
        
        analyzer = GradientAnalyzer() if (cfg.setup.gradient_analysis_run and iter_num % cfg.setup.log_interval == 0) else None
        
        if analyzer:
            logger.info(f"Performing gradient analysis for step {iter_num} (this takes extra time)...")
            # 1. Buffer batches for identical sequential forward passes
            micro_batches = []
            for _ in range(grad_accum_steps):
                # Move batch to CPU for buffering so we don't spike VRAM
                cpu_batch = {k: v.cpu() for k, v in batch.items()}
                micro_batches.append(cpu_batch)
                batch, current_epoch, current_batch_idx = next(batch_generator)
                
            # Capture RNG states for mathematical fairness during analysis passes
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state(device)
                
            # 2. Independent passes for each loss term (keys discovered dynamically)
            optimizer.zero_grad(set_to_none=True)
            loss_keys = []
            loss_idx = 0
            
            while True:
                # Restore RNG state for perfect replication of the forward pass per loss term
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(gpu_rng_state, device=device)
                
                for b_cpu in micro_batches:
                    b_gpu = {k: v.to(device, non_blocking=True) for k, v in b_cpu.items()}
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss_dict = model(**b_gpu)
                        _loss, _logged_losses, weighted_tensors = loss_wrapper(loss_dict, step=iter_num)
                        
                        if not loss_keys:
                            loss_keys = list(weighted_tensors.keys())
                            
                        loss_term = weighted_tensors[loss_keys[loss_idx]] / grad_accum_steps
                    loss_term.backward()
                    
                    # Prevent High-Water Mark VRAM spikes: immediately free unused graphs
                    del b_gpu, loss_dict, _loss, _logged_losses, weighted_tensors, loss_term
                
                # Extract the standard .grad attributes to CPU
                analyzer.extract_gradients(unoptimized_model, loss_keys[loss_idx])
                optimizer.zero_grad(set_to_none=True)
                
                loss_idx += 1
                if loss_idx >= len(loss_keys):
                    break
                
            # 3. Standard update pass over the exact same data
            # Restore RNG state one last time so the actual training step aligns with the analysis
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state, device=device)
            
            for b_cpu in micro_batches:
                b_gpu = {k: v.to(device, non_blocking=True) for k, v in b_cpu.items()}
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss_dict = model(**b_gpu)
                    loss, logged_losses, weighted_tensors = loss_wrapper(loss_dict, step=iter_num)
                    scaled_loss = loss / grad_accum_steps
                    
                scaled_loss.backward()
                
                accum_loss = accum_loss + scaled_loss.detach()
                for k, v in logged_losses.items():
                    accum_logged_losses[k] = accum_logged_losses.get(k, 0.0) + v / grad_accum_steps
                    
                del b_gpu, loss_dict, loss, logged_losses, weighted_tensors, scaled_loss
                    
        else:
            for _ in range(grad_accum_steps):
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss_dict = model(**batch)
                    loss, logged_losses, _ = loss_wrapper(loss_dict, step=iter_num)
                    scaled_loss = loss / grad_accum_steps
                    
                scaled_loss.backward()
                
                # GPU-side accumulation of detached scalars purely for logging
                accum_loss = accum_loss + scaled_loss.detach()
                for k, v in logged_losses.items():
                    accum_logged_losses[k] = accum_logged_losses.get(k, 0.0) + v / grad_accum_steps
                    
                # Asynchronous pre-fetch of the next batch while backward pass computes
                batch, current_epoch, current_batch_idx = next(batch_generator)

        if cfg.training.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # -----------------------------
        # Dynamo Compilation Verdict
        # -----------------------------
        if (iter_num - start_iter) in COMPILE_MILESTONES:
            logger.info(f"\n========== TORCH.COMPILE STATUS (Step {iter_num - start_iter}) ==========")
            counters = torch._dynamo.utils.counters
            
            graph_breaks = counters.get("graph_break", {})
            total_breaks = sum(graph_breaks.values())
            num_buckets = len(cfg.dataloader.bucket_mapping)
            num_disable_points = 2  # encodec.get_latents, aligner.maximum_path_indices
            expected_breaks = num_buckets * num_disable_points

            logger.info(f"Total Traced Graph Breaks: {total_breaks} (Expected maximum: {num_buckets} buckets * {num_disable_points} disable points = {expected_breaks})")
            if total_breaks <= expected_breaks:
                logger.info("✅ Batch bucketing is stable and no unintended graph breaks occurred.")
            else:
                logger.warning("⚠️ WARNING: Too many traces! Either new graph breaks were introduced, or batch shapes are leaking.")
            logger.info("==========================================\n")

        # -----------------------------
        # Timing & Logging
        # -----------------------------
        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % cfg.setup.log_interval == 0:
            # CPU-GPU sync point due to .item() extraction
            lossf = accum_loss.item()
            
            logger.info(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
            if iter_num > 0 and cfg.setup.save_checkpoint:
                checkpoint_data = {
                    'model': unoptimized_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_cfg': model_cfg_dict,
                    'token_vocabulary_size': token_vocabulary_size,
                    'sampling_rate': sampling_rate,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'wandb_id': wandb.run.id if cfg.wandb.log else None,
                    'epoch': current_epoch,
                    'batch_idx': current_batch_idx, # Index of the pre-fetched batch for the upcoming step
                }
                
                ckpt_path = CHECKPOINTS_DIR / 'ckpt.pt'
                ckpt_tmp_path = CHECKPOINTS_DIR / 'ckpt.pt.tmp'
                ckpt_bak_path = CHECKPOINTS_DIR / 'ckpt_bak.pt'
                
                torch.save(checkpoint_data, ckpt_tmp_path)
                if ckpt_path.exists():
                    ckpt_path.replace(ckpt_bak_path)
                ckpt_tmp_path.replace(ckpt_path)
                
            if cfg.wandb.log:
                log_payload = {
                    "train/iter": iter_num,
                    "train/loss": lossf,
                    "train/time": dt * 1000,
                    "train/lr": lr
                }
                # Log individual balanced loss components as well — .item() here
                # (inside log_interval) so we don't sync the GPU every step.
                for k, v in accum_logged_losses.items():
                    log_payload[f"train/losses/{k}"] = v.item()

                # Perform expensive gradient analysis only when logging
                if analyzer is not None:
                    grad_norms, cos_sims = analyzer.compute_metrics()
                    for k, v in grad_norms.items():
                        log_payload[f"train/grad_norms/{k}"] = v
                    for k, v in cos_sims.items():
                        log_payload[f"train/cos_sims/{k}"] = v

                wandb.log(log_payload)

            # Accumulate unweighted raw losses exclusively for the analysis table
            if cfg.setup.loss_analysis_run:
                for k, v in accum_logged_losses.items():
                    if not k.endswith("_weighted"):
                        loss_analysis_accumulators[k] = loss_analysis_accumulators.get(k, 0.0) + v.item()

    # -----------------------------
    # Loss Analysis Summary Dump
    # -----------------------------
    if cfg.setup.loss_analysis_run:
        logger.info("========== LOSS ANALYSIS SUMMARY ==========")
        logger.info(f"Analyzed over {cfg.setup.max_iters} iterations.")
        logger.info("Average raw unweighted loss magnitudes:")
        for k, v in loss_analysis_accumulators.items():
            avg = v / cfg.setup.max_iters
            logger.info(f"  {k}: {avg:.4f}")
        logger.info("===========================================")


if __name__ == "__main__":
    # Enable PyTorch Memory Expansion to heavily mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train()