import os
import time
import math
import random
import logging
import itertools
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

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.inference import compute_inference_data_loss, generate_audio
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper, GradientAnalyzer
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.paths import CHECKPOINTS_DIR, PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators

logger = logging.getLogger(__name__)

COMPILE_MILESTONES = [1, 250]

def override_dict(d, val):
    for k, v in d.items():
        if isinstance(v, dict):
            override_dict(v, val)
        else:
            d[k] = val

def get_loss_section(key: str, phase: str, split: str = "") -> str:
    is_weighted = key.endswith("weighted")
    
    if "total_weighted" in key:
        group = "Losses"
    elif any(x in key for x in ["data_loss", "score_loss", "ce_rvq_loss"]):
        group = "Diffusion-Losses"
    elif any(x in key for x in ["forward_sum_loss", "bin_loss"]):
        group = "Aligner-Losses"
    else:
        group = "Losses"
        
    weight_str = "(Weighted)" if is_weighted else "(Raw)"
    if phase == "Train":
        return f"Train: {group} {weight_str}"
    else:
        return f"Evaluation: {split}-{group} {weight_str}"

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

def get_infinite_batches(loader, start_epoch=0, start_batch_idx=0, overfit_single_batch=False, grad_accum_steps=1):
    """Continuously yields batches while tracking and setting dataloader state for instant resuming."""
    epoch = start_epoch
    sampler = loader.batch_sampler
    sampler.set_epoch(epoch)
    sampler.set_start_batch_idx(start_batch_idx)

    if overfit_single_batch:
        logger.info(f"OVERFIT TEST ACTIVE: Yielding the exact same {grad_accum_steps} micro-batches endlessly.")
        loader_iter = iter(loader)
        overfit_batches = [next(loader_iter) for _ in range(grad_accum_steps)]
        
        while True:
            for b in overfit_batches:
                yield b, epoch, 0

    while True:
        for batch_idx, batch in enumerate(loader, start=sampler.start_batch_idx):
            # Yield CPU batches to buffer into the lookahead queue
            yield batch, epoch, batch_idx

        # Epoch finished
        epoch += 1
        sampler.set_epoch(epoch)
        sampler.set_start_batch_idx(0)

@torch.no_grad()
def estimate_loss(model, train_loader, dev_loader, test_loader, loss_wrapper, eval_iters, grad_accum_steps, device, cfg):
    out = {}
    model.eval()
    
    # Store main loop's train sampler state
    train_sampler = train_loader.batch_sampler
    main_train_epoch = train_sampler.epoch
    main_train_batch_idx = train_sampler.start_batch_idx

    # Override for eval to pull a random shuffled subset starting from 0
    train_sampler.set_epoch(random.randint(0, 10000))
    train_sampler.set_start_batch_idx(0)

    # We evaluate 'dev' first. If it hits the end of the combined dataset
    # before eval_iters, we restrict 'train' to that exact same number of steps.
    target_iters = eval_iters

    for split in ['dev', 'train']:
        if split == 'dev':
            # Chain both iterators so they act as one continuous dataloader
            loader_iter = itertools.chain(dev_loader, test_loader)
        else:
            loader_iter = iter(train_loader)
            
        eval_total_loss_sum = torch.zeros((), device=device)
        eval_log_dict_sums = {}
        actual_eval_iters = 0

        for _ in range(target_iters):
            eval_lookahead_queue = []
            try:
                for _ in range(grad_accum_steps):
                    eval_lookahead_queue.append(next(loader_iter))
                    
            except StopIteration:
                break # Drop incomplete logical batch and terminate this split's evaluation

            eval_denominators = compute_denominators(eval_lookahead_queue, cfg)
            
            eval_accum_loss = torch.zeros((), device=device)
            eval_accum_logged_losses = {}

            for batch in eval_lookahead_queue:
                # Skip non-tensor fields (`text: list[str]` from the collate)
                # and exclude them from the model() call since forward()
                # has an explicit kwarg list.
                tensor_batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                    loss_dict = model(**tensor_batch)
                    eval_loss, eval_logged_losses, _ = loss_wrapper(loss_dict, denominators=eval_denominators)

                eval_accum_loss += eval_loss.detach()
                for key, val in eval_logged_losses.items():
                    eval_accum_logged_losses[key] = eval_accum_logged_losses.get(key, 0.0) + val

            eval_total_loss_sum += eval_accum_loss
            for key, val in eval_accum_logged_losses.items():
                eval_log_dict_sums[key] = eval_log_dict_sums.get(key, 0.0) + val
                
            actual_eval_iters += 1
            
        if split == 'dev':
            target_iters = actual_eval_iters
            
        divisor = actual_eval_iters
        out[split] = {
            'total_loss': (eval_total_loss_sum / divisor).item(),
            'logged_losses': {key: (val / divisor).item() for key, val in eval_log_dict_sums.items()},
        }
        
    # Restore main loop's train sampler state
    train_sampler.set_epoch(main_train_epoch)
    train_sampler.set_start_batch_idx(main_train_batch_idx)

    model.train()
    return out

def create_dataloader(cfg, split: str, token_vocabulary_path: str = None):
    bucket_mapping = OmegaConf.to_container(cfg.dataloader.bucket_mapping, resolve=True)
    
    max_audio_length = cfg.dataset.max_audio_length
    max_phoneme_length = cfg.dataset.max_phoneme_length
    
    if split != cfg.dataset.train_split:    # bucket mapping is derived from train split
        largest_bucket = max(bucket_mapping, key=lambda x: x['audio_length'])
        max_audio_length = largest_bucket['audio_length']
        max_phoneme_length = largest_bucket['phoneme_length']
        logger.info(f"Overriding upper boundaries for '{split}' split to match max bucket: "
                    f"audio={max_audio_length}, phonemes={max_phoneme_length}")

    dataset = DatasetWrapper(
        dataset_source=cfg.dataset.source,
        dataset_name=cfg.dataset.name,
        split=split,
        text_column=cfg.dataset.text_column,
        audio_column=cfg.dataset.audio_column,
        filter_column=cfg.dataset.filter_column,
        filter_substring=cfg.dataset.filter_substring,
        token_vocabulary_path=token_vocabulary_path,
        min_audio_length=cfg.dataset.min_audio_length,
        max_audio_length=max_audio_length,
        min_phoneme_length=cfg.dataset.min_phoneme_length,
        max_phoneme_length=max_phoneme_length,
        sampling_rate=cfg.dataloader.sampling_rate,
        resample_on_the_fly=cfg.dataloader.resample_on_the_fly,
        num_proc_pitch=cfg.dataloader.num_proc_pitch,
        num_proc_phonemize=cfg.dataloader.num_proc_phonemize,
        num_proc_tokenize=cfg.dataloader.num_proc_tokenize,
    )
    sampler = DynamicBucketedBatchSampler(
        dataset,
        bucket_mapping=bucket_mapping,
        drop_last=cfg.dataloader.drop_last,
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
    dev_loader, dev_dataset = create_dataloader(cfg, cfg.dataset.dev_split, train_dataset.token_vocabulary_path)
    test_loader, _ = create_dataloader(cfg, cfg.dataset.test_split, train_dataset.token_vocabulary_path)

    # Shared tokenizer with espeak backend: feeds vocab_size to model construction
    # AND is attached as model._inference_tokenizer below so the eval block's
    # generate_audio() calls hide phoneme plumbing.
    inference_tokenizer = PhonemeTokenizer(
        token_vocabulary_path=train_dataset.token_vocabulary_path, with_backend=True,
    )
    token_vocabulary_size = inference_tokenizer.token_vocabulary_size

    # Eval block's static text prompts. generate_audio re-phonemizes per call
    # (~50ms × 4 prompts × eval intervals — negligible per inference.md §4.2).
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

    sampling_rate = cfg.dataloader.sampling_rate
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)

    # State initialization variables
    start_iter = 0
    best_dev_loss = 1e9
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
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']

        # Subsequent checkpoints must save the cfg the model was actually built with,
        # not the (possibly drifted) Hydra cfg captured above on resume.
        model_cfg_dict = checkpoint['model_cfg']
        start_batch_idx = checkpoint['batch_idx'] # Already points to the next batch due to pre-fetch

        logger.info(f"Resuming training at iteration {start_iter} from checkpoint in {CHECKPOINTS_DIR}...")

    model.to(device)

    # Attach inference helpers for the eval block's generate_audio() calls.
    # These are non-Parameter/Buffer attributes — don't pollute state_dict, don't
    # affect torch.compile, don't affect forward(). generate_audio() reads them.
    model._inference_tokenizer = inference_tokenizer
    model._inference_sampling_rate = sampling_rate

    logger.info(f"Phoneme vocabulary size: {token_vocabulary_size}")
    trainable_params = model.num_parameters()
    total_params = model.num_parameters(only_trainable=False)
    non_trainable_params = total_params - trainable_params
    logger.info(f"Model has {total_params / 1e6:.2f}M total parameters ({trainable_params / 1e6:.2f}M trainable, {non_trainable_params / 1e6:.2f}M non-trainable).")
    
    loss_weights_dict = OmegaConf.to_container(cfg.model.loss_weights, resolve=True)
    loss_warmup_steps_dict = OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)

    if cfg.setup.loss_analysis_run:
        logger.info("LOSS ANALYSIS RUN: Forcing all dynamic loss weights to 1.0 and warmups to 0.")
        override_dict(loss_weights_dict, 1.0)
        override_dict(loss_warmup_steps_dict, 0)
    elif cfg.setup.overfit_single_batch:
        logger.info("OVERFIT TEST: Forcing loss warmups to 0 (the 1000-iter ramp would otherwise eat half a 2000-iter run).")
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
            notes=cfg.wandb.notes,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            id=resume_wandb_id,
            resume="allow" if resume_wandb_id else None
        )
        
        wandb.define_metric("Train Iteration")
        wandb.define_metric("Evaluation Iteration")
        wandb.define_metric("Train: *", step_metric="Train Iteration")
        wandb.define_metric("Gradient Analysis: *", step_metric="Train Iteration")
        wandb.define_metric("Evaluation: *", step_metric="Evaluation Iteration")
        
        table_2_refs = []
        num_static_refs = 6
        prompt_samples_len = int(cfg.model.prompt_seconds * sampling_rate)
        
        valid_indices = []
        all_indices = list(range(len(dev_dataset)))
        random.shuffle(all_indices)
        
        for idx in all_indices:
            if len(valid_indices) == num_static_refs:
                break
            if dev_dataset.dataset[idx]["audio_length"] >= prompt_samples_len:
                valid_indices.append(idx)
                
        for idx in valid_indices:
            sample = dev_dataset[idx]
            audio_np = sample["audio"].numpy()
            
            max_start = sample["audio_length"] - prompt_samples_len
            start_idx = random.randint(0, max_start)
            prompt_audio_np = audio_np[start_idx : start_idx + prompt_samples_len]
            
            table_2_refs.append({
                "original_audio": wandb.Audio(audio_np, sample_rate=sampling_rate),
                "prompt_audio": wandb.Audio(prompt_audio_np, sample_rate=sampling_rate),
                "prompt_tensor": torch.from_numpy(prompt_audio_np),
                "text": sample["text"],
            })

    batch_generator = get_infinite_batches(
        train_loader, 
        start_epoch, 
        start_batch_idx, 
        cfg.setup.overfit_single_batch, 
        cfg.setup.gradient_accumulation_steps
    )
    
    sampler = train_loader.batch_sampler
    G = cfg.setup.gradient_accumulation_steps
    sr = sampling_rate
    exp_minutes = (G * sampler.expected_batch_audio_samples) / sr / 60.0
    std_minutes = math.sqrt(G * sampler.variance_batch_audio_samples) / sr / 60.0
    cv_logical = std_minutes / exp_minutes
    logger.info(f"Expected audio processed per logical step: ~{exp_minutes:.2f}m (± std of {std_minutes:.2f}m)")
    logger.info(f"  - Relative Fluctuation (CV): {cv_logical:.1%}. Target: < 10% for good stability.")

    loss_analysis_accumulators = {}
    logger.info("Starting training loop...")
    last_log_time = time.perf_counter()
    last_log_iter = start_iter - 1
    
    for iter_num in range(start_iter, cfg.setup.max_iters):

        # Apply LR scheduling
        lr = get_lr(iter_num, cfg) if cfg.training.decay_lr else cfg.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -----------------------------
        # Lookahead Queue & Denominators
        # -----------------------------
        lookahead_queue = []
        for _ in range(cfg.setup.gradient_accumulation_steps):
            cpu_batch, current_epoch, current_batch_idx = next(batch_generator)
            lookahead_queue.append(cpu_batch)
            
        global_denominators = compute_denominators(lookahead_queue, cfg)
            
        # -----------------------------
        # Evaluation & Checkpointing
        # -----------------------------
        if iter_num % cfg.setup.eval_interval == 0 and cfg.setup.save_checkpoint:
            logger.info(f"Running evaluation loop...")
            losses = estimate_loss(
                model, train_loader, dev_loader, test_loader, loss_wrapper, 
                cfg.setup.eval_iters, cfg.setup.gradient_accumulation_steps, device, cfg
            )
            logger.info(f"Step {iter_num}: train loss {losses['train']['total_loss']:.4f}, dev loss {losses['dev']['total_loss']:.4f}")
            
            if cfg.wandb.log:
                eval_payload = {
                    "Evaluation Iteration": iter_num,
                    "Evaluation: Metrics/Learning Rate": lr,
                    "Evaluation: Metrics/Train-Loss": losses['train']['total_loss'],
                    "Evaluation: Metrics/Dev-Loss": losses['dev']['total_loss'],
                }
                
                for k, v in losses['train']['logged_losses'].items():
                    eval_payload[f"{get_loss_section(k, 'Evaluation', 'Train')}/{k}"] = v
                for k, v in losses['dev']['logged_losses'].items():
                    eval_payload[f"{get_loss_section(k, 'Evaluation', 'Dev')}/{k}"] = v

                # --- Generation Testing ---
                logger.info("Generating audio samples for evaluation...")

                # estimate_loss() flips back to train mode before returning; bracket the
                # sampling loop in eval mode so dropout (prompt encoder, predictors,
                # WaveNet blocks) doesn't perturb inference.
                unoptimized_model.eval()
                
                # --- Table 1: Random Generation Examples ---
                table_1_rows = []
                ten_seconds_samples = int(10.0 * sampling_rate)
                five_seconds_samples = int(5.0 * sampling_rate)
                
                random_indices = list(range(len(dev_dataset)))
                random.shuffle(random_indices)
                test_idx = next(i for i in random_indices if dev_dataset.dataset[i]["audio_length"] >= ten_seconds_samples)
                sample = dev_dataset[test_idx]
                        
                audio_np = sample["audio"].numpy()
                max_start = sample["audio_length"] - ten_seconds_samples
                start_idx = random.randint(0, max_start)
                
                prompt_5s_np = audio_np[start_idx : start_idx + five_seconds_samples]
                prompt_10s_np = audio_np[start_idx : start_idx + ten_seconds_samples]
                
                prompt_idx = random.randint(0, len(custom_prompts) - 1)
                target_text = custom_prompts[prompt_idx]

                for p_len, p_np in [(5.0, prompt_5s_np), (10.0, prompt_10s_np)]:
                    audio_np, length = generate_audio(
                        unoptimized_model, p_np, target_text=target_text,
                    )

                    table_1_rows.append([
                        iter_num,
                        p_len,
                        target_text,
                        wandb.Audio(p_np, sample_rate=sampling_rate),
                        wandb.Audio(audio_np[:length], sample_rate=sampling_rate),
                    ])
                        
                # --- Table 2: Original vs. Generated ---
                table_2_rows = []
                for ref in table_2_refs:
                    audio_np, length = generate_audio(
                        unoptimized_model, ref["prompt_tensor"], target_text=ref["text"],
                    )

                    table_2_rows.append([
                        iter_num,
                        cfg.model.prompt_seconds,
                        ref["text"],
                        ref["original_audio"],
                        ref["prompt_audio"],
                        wandb.Audio(audio_np[:length], sample_rate=sampling_rate),
                    ])

                unoptimized_model.train()

                eval_payload["Evaluation: Random Generation Examples"] = wandb.Table(
                    columns=["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt", "Speech-Prompt", "Generated Audio"],
                    data=table_1_rows,
                )
                
                eval_payload["Evaluation: Original vs. Generated"] = wandb.Table(
                    columns=["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt", "Original Audio", "Speech-Prompt", "Generated Audio"],
                    data=table_2_rows,
                )
                
                wandb.log(eval_payload, step=iter_num)
                
            if iter_num > 0 and losses['dev']['total_loss'] < best_dev_loss:
                best_dev_loss = losses['dev']['total_loss']
                logger.info(f"Saving new best model to {CHECKPOINTS_DIR}")
                
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
        
        accum_loss = torch.zeros((), device=device)
        accum_logged_losses = {}
        
        analyzer = GradientAnalyzer() if (cfg.setup.gradient_analysis_run and iter_num % cfg.setup.log_interval == 0) else None
        
        if analyzer:
            logger.info(f"Performing gradient analysis for step {iter_num} (this takes extra time)...")
                
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
                
                for b_cpu in lookahead_queue:
                    b_gpu = {k: v.to(device, non_blocking=True) for k, v in b_cpu.items() if isinstance(v, torch.Tensor)}
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss_dict = model(**b_gpu)
                        _loss, _logged_losses, weighted_tensors = loss_wrapper(loss_dict, step=iter_num, denominators=global_denominators)
                        
                        if not loss_keys:
                            loss_keys = list(weighted_tensors.keys())
                            
                        loss_term = weighted_tensors[loss_keys[loss_idx]]
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
            
            for b_cpu in lookahead_queue:
                b_gpu = {k: v.to(device, non_blocking=True) for k, v in b_cpu.items() if isinstance(v, torch.Tensor)}
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss_dict = model(**b_gpu)
                    loss, logged_losses, weighted_tensors = loss_wrapper(loss_dict, step=iter_num, denominators=global_denominators)
                    
                loss.backward()
                
                accum_loss += loss.detach()
                for k, v in logged_losses.items():
                    accum_logged_losses[k] = accum_logged_losses.get(k, 0.0) + v
                    
                del b_gpu, loss_dict, loss, logged_losses, weighted_tensors
                    
        else:
            for b_cpu in lookahead_queue:
                b_gpu = {k: v.to(device, non_blocking=True) for k, v in b_cpu.items() if isinstance(v, torch.Tensor)}
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss_dict = model(**b_gpu)
                    loss, logged_losses, _ = loss_wrapper(loss_dict, step=iter_num, denominators=global_denominators)
                    
                loss.backward()
                
                # GPU-side accumulation of detached scalars purely for logging
                accum_loss += loss.detach()
                for k, v in logged_losses.items():
                    accum_logged_losses[k] = accum_logged_losses.get(k, 0.0) + v
                    
                del b_gpu, loss_dict, loss, logged_losses

        max_norm = cfg.training.grad_clip if cfg.training.grad_clip != 0.0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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
        
        if iter_num % cfg.setup.log_interval == 0:
            # CPU-GPU sync point due to .item() extraction
            lossf = accum_loss.item()
            
            current_time = time.perf_counter()
            steps_since_last = iter_num - last_log_iter
            dt_avg = (current_time - last_log_time) / steps_since_last
            
            logger.info(f"Iteration: {iter_num}, Loss: {lossf:.4f}, Avg. Time/Step: {dt_avg*1000:.2f}ms")
            
            last_log_time = current_time
            last_log_iter = iter_num
            
            if iter_num > 0 and cfg.setup.save_checkpoint:
                checkpoint_data = {
                    'model': unoptimized_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_cfg': model_cfg_dict,
                    'token_vocabulary_size': token_vocabulary_size,
                    'sampling_rate': sampling_rate,
                    'iter_num': iter_num,
                    'best_dev_loss': best_dev_loss,
                    'wandb_id': wandb.run.id if cfg.wandb.log else None,
                    'epoch': current_epoch,
                    'batch_idx': current_batch_idx + 1, # Index of the upcoming batch
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
                    "Train Iteration": iter_num,
                    "Train: Metrics/Loss": lossf,
                    "Train: Metrics/Avg. Time per Step (ms)": dt_avg * 1000,
                    "Train: Metrics/Learning Rate": lr,
                    "Train: Metrics/Gradient Norm": grad_norm.item(),
                }
                # Log individual balanced loss components as well — .item() here
                # (inside log_interval) so we don't sync the GPU every step.
                for k, v in accum_logged_losses.items():
                    log_payload[f"{get_loss_section(k, 'Train')}/{k}"] = v.item()

                # Perform expensive gradient analysis only when logging
                if analyzer is not None:
                    grad_norms, cos_sims = analyzer.compute_metrics()
                    for k, v in grad_norms.items():
                        if k.endswith("_total"):
                            log_payload[f"Gradient Analysis: L2-Norms (Total)/{k}"] = v
                        else:
                            log_payload[f"Gradient Analysis: L2-Norms (Shared)/{k}"] = v
                    for k, v in cos_sims.items():
                        log_payload[f"Gradient Analysis: Cosine-Similarity (Shared)/{k}"] = v

                wandb.log(log_payload, step=iter_num)

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

    # -----------------------------
    # Overfit Test Audio Comparison
    # -----------------------------
    # Generate audio from the overfit batch so a human reviewer can listen to
    # (original | prompt | generated) side-by-side in the wandb UI. The prompt
    # is the deterministic first `prompt_seconds` of each clip's own audio —
    # training uses a random start position, but a fixed start makes the eval
    # reproducible across runs.
    if cfg.setup.overfit_single_batch and cfg.wandb.log:
        logger.info("Generating audio for overfit-batch comparison...")
        unoptimized_model.eval()

        sr = cfg.dataloader.sampling_rate
        prompt_samples = int(cfg.model.prompt_seconds * sr)

        batch = lookahead_queue[0]

        audio_full = batch["audio"].to(device)                                # [B, T]
        audio_lengths_full = batch["audio_lengths"].to(device)                # [B]

        # -----------------------------
        # Inference data_loss diagnostic
        # -----------------------------
        # Training data_loss is a one-step prediction error from a known noisy z_t.
        # Inference chains N ODE steps from t=1 to t≈0, each step using the network's
        # prediction. Per-step errors compound. Measuring the same MSE on the sampler's
        # actual output tells us whether audio quality issues live in (a) the sampler
        # trajectory or (b) something downstream (off-manifold predictions, decoder).
        # Uses GT condition + GT prompt extracted from a fresh forward pass so the only
        # variable being measured is the sampler integration error.
        #
        # Sweep `sampling_steps` to distinguish solver-discretization error from a
        # weights-side gap (e.g. missing EMA, undertrained t bins). Monotonic drop to
        # ~training data_loss = solver-bound; flat across step counts = weights-side.
        logger.info("Computing inference_data_loss diagnostic...")
        sampling_steps_sweep = (150, 300, 600, 1000)
        sweep_losses = compute_inference_data_loss(
            unoptimized_model, batch, sampling_steps_sweep=sampling_steps_sweep,
        )
        sweep_payload = {}
        for n_steps, loss_val in sweep_losses.items():
            logger.info(f"  inference_data_loss @ {n_steps} steps = {loss_val:.6f}")
            sweep_payload[f"eval/overfit_inference_data_loss_steps_{n_steps}"] = loss_val
        wandb.log(sweep_payload, step=iter_num)

        num_compare = min(2, audio_full.shape[0])
        table_rows = []
        for i in range(num_compare):
            T_i = int(audio_lengths_full[i].item())
            prompt_T = min(prompt_samples, T_i)

            ref_audio_slice = audio_full[i, :prompt_T]                              # [T_p]

            audio_np, length = generate_audio(
                unoptimized_model, ref_audio_slice, target_text=batch["text"][i],
            )

            original_np = audio_full[i, :T_i].detach().cpu().to(torch.float32).numpy()
            prompt_np = ref_audio_slice.detach().cpu().to(torch.float32).numpy()

            table_rows.append([
                i,
                batch["text"][i],
                wandb.Audio(original_np, sample_rate=sr),
                wandb.Audio(prompt_np, sample_rate=sr),
                wandb.Audio(audio_np[:length], sample_rate=sr),
            ])

        wandb.log({
            "overfit_audio_comparison": wandb.Table(
                columns=["clip_idx", "text", "original", "prompt", "generated"],
                data=table_rows,
            )
        }, step=iter_num)
        logger.info(f"Logged {num_compare} audio comparison rows to wandb.")


if __name__ == "__main__":
    # Enable PyTorch Memory Expansion to heavily mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train()