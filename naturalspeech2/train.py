import os
import time
import math
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

import torch
from torch.utils.data import DataLoader
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_model

from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.model import NaturalSpeech2Model
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.utils.utils import LossWrapper
from naturalspeech2.paths import CHECKPOINTS_DIR

logger = logging.getLogger(__name__)

def setup_logging(out_dir):
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(out_dir, "training.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def get_lr(it, cfg):
    learning_rate = cfg.training.learning_rate
    warmup_iters = cfg.training.warmup_iters
    lr_decay_iters = cfg.training.lr_decay_iters
    min_lr = cfg.training.min_lr

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def get_infinite_batches(loader, device, start_epoch=0, start_batch_idx=0):
    """Continuously yields batches while tracking and setting dataloader state for instant resuming."""
    epoch = start_epoch
    sampler = loader.batch_sampler
    sampler.set_epoch(epoch)
    sampler.set_start_batch_idx(start_batch_idx)
    
    while True:
        for batch_idx, batch in enumerate(loader, start=sampler.start_batch_idx):
            # Move immediately to device asynchronously
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
                
            for k_b, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k_b] = v.to(device, non_blocking=True)
                    
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                loss_dict = model(**batch)
                total_loss, _ = loss_wrapper(loss_dict)
            losses[k] = total_loss.item()
        out[split] = losses.mean().item()
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

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires an NVIDIA GPU and CUDA installed, but none were detected.")
    
    device = cfg.training.device
    device_type = 'cuda'
    
    # Create checkpoints directory and setup specific logs
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(CHECKPOINTS_DIR)
    
    logger.info("Initializing DataLoaders...")
    train_loader, train_dataset = create_dataloader(cfg, cfg.dataset.train_split, cfg.dataset.token_vocabulary_path)
    val_loader, _ = create_dataloader(cfg, cfg.dataset.val_split, train_dataset.token_vocabulary_path)

    tokenizer = PhonemeTokenizer(token_vocabulary_path=train_dataset.token_vocabulary_path, with_backend=False)
    token_vocabulary_size = tokenizer.token_vocabulary_size

    # Build Model Args Dict
    model_args = {
        'device': device,
        'token_vocabulary_size': token_vocabulary_size,
        'hidden_dim': cfg.model.hidden_dim,
        'latent_dim': cfg.model.latent_dim,
        'sampling_rate': cfg.dataloader.sampling_rate,
        'rope_base': cfg.model.rope_base,
        'rope_max_seq_len': cfg.model.rope_max_seq_len,
        'min_prompt_pct': cfg.model.min_prompt_pct,
        'max_prompt_pct': cfg.model.max_prompt_pct,

        # Log Mel Spectrogram parameters
        'n_fft': cfg.model.mel.n_fft,
        'hop_length': cfg.model.mel.hop_length,
        'n_mels': cfg.model.mel.n_mels,
        'f_min': cfg.model.mel.f_min,
        'f_max': cfg.model.mel.f_max,

        # Phoneme Encoder parameters
        'phoneme_encoder_layers': cfg.model.phoneme_encoder.transformer_layers,
        'phoneme_encoder_heads': cfg.model.phoneme_encoder.attention_heads,
        'phoneme_encoder_filter_size': cfg.model.phoneme_encoder.conv1d_filter_size,
        'phoneme_encoder_kernel_size': cfg.model.phoneme_encoder.conv1d_kernel_size,
        'phoneme_encoder_dropout': cfg.model.phoneme_encoder.dropout,

        # Aligner parameters
        'aligner_attn_channels': cfg.model.aligner.attn_channels,
        'aligner_temperature': cfg.model.aligner.temperature,
        'prior_w': cfg.model.aligner.prior_w,

        # Speech Prompt Encoder parameters
        'speech_prompt_encoder_layers': cfg.model.speech_prompt_encoder.transformer_layers,
        'speech_prompt_encoder_heads': cfg.model.speech_prompt_encoder.attention_heads,
        'speech_prompt_encoder_filter_size': cfg.model.speech_prompt_encoder.conv1d_filter_size,
        'speech_prompt_encoder_kernel_size': cfg.model.speech_prompt_encoder.conv1d_kernel_size,
        'speech_prompt_encoder_dropout': cfg.model.speech_prompt_encoder.dropout,

        # Duration Predictor parameters
        'duration_predictor_conv1d_layers': cfg.model.duration_predictor.conv1d_layers,
        'duration_predictor_conv1d_kernel_size': cfg.model.duration_predictor.conv1d_kernel_size,
        'duration_predictor_attention_layers': cfg.model.duration_predictor.attention_layers,
        'duration_predictor_attention_heads': cfg.model.duration_predictor.attention_heads,
        'duration_predictor_dropout': cfg.model.duration_predictor.dropout,
    }

    # State initialization variables
    iter_num = 0
    best_val_loss = 1e9
    start_epoch = 0
    start_batch_idx = 0
    
    # Instantiate Model
    if cfg.training.init_from == 'scratch':
        logger.info("Initializing a new model from scratch...")
        model = NaturalSpeech2Model(**model_args)
    elif cfg.training.init_from == 'resume':
        logger.info(f"Resuming training from checkpoint in {CHECKPOINTS_DIR}...")
        ckpt_path = CHECKPOINTS_DIR / 'ckpt.pt'
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        
        # Extract previous model args (ensures architecture matches exactly)
        ckpt_model_args = checkpoint['model_args']
        model = NaturalSpeech2Model(**ckpt_model_args)
        
        state_dict = checkpoint['model']
                
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint.get('epoch', 0)
        start_batch_idx = checkpoint.get('batch_idx', 0) # Already points to the next batch due to pre-fetch
        
    model.to(device)
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
    ).to(device)
    
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay, 
        cfg.training.learning_rate, 
        (cfg.training.beta1, cfg.training.beta2)
    )
    
    if cfg.training.init_from == 'resume':
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
            config=OmegaConf.to_container(cfg, resolve=True), 
            id=resume_wandb_id, 
            resume="allow" if resume_wandb_id else None
        )

    batch_generator = get_infinite_batches(train_loader, device, start_epoch, start_batch_idx)
    batch, current_epoch, current_batch_idx = next(batch_generator)
    
    t0 = time.perf_counter()
    logger.info("Starting training loop...")
    for iter_num in range(iter_num, cfg.training.max_iters):
        
        # Apply LR scheduling
        lr = get_lr(iter_num, cfg) if cfg.training.decay_lr else cfg.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # -----------------------------
        # Evaluation & Checkpointing
        # -----------------------------
        if iter_num % cfg.training.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, loss_wrapper, cfg.training.eval_iters, device)
            logger.info(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if cfg.wandb.log:
                # TODO: Generation testing could go here via unoptimized_model.generate() or model.generate()
                pass
                
            if iter_num > 0 and losses['val'] < best_val_loss:
                best_val_loss = losses['val']
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
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss_dict = model(**batch)
            loss, logged_losses = loss_wrapper(loss_dict, step=iter_num)
            
        # Asynchronous pre-fetch of the next batch while backward pass computes
        batch, current_epoch, current_batch_idx = next(batch_generator)
        
        loss.backward()
        
        if cfg.training.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # -----------------------------
        # Timing & Logging
        # -----------------------------
        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % cfg.training.log_interval == 0:
            # CPU-GPU sync point due to .item() extraction
            lossf = loss.item()
            
            logger.info(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
            if iter_num > 0:
                checkpoint_data = {
                    'model': unoptimized_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
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
                # Log individual balanced loss components as well
                for k, v in logged_losses.items():
                    log_payload[f"train/losses/{k}"] = v
                    
                wandb.log(log_payload)


if __name__ == "__main__":
    # train()
    pass