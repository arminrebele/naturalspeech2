import os
import time
import math
import random
import logging
import itertools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch._dynamo
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.dataset import DatasetWrapper, BucketedCollateFn, DynamicBucketedBatchSampler
from naturalspeech2.inference import compute_inference_data_loss, generate_audio
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper, GradientAnalyzer
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.paths import CHECKPOINTS_DIR, PROJECT_ROOT
from naturalspeech2.utils.ema import EMA
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators

logger = logging.getLogger(__name__)

COMPILE_MILESTONES = [1, 250]

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


@dataclass
class EvalDeps:
    """Bag of dependencies for run_eval_block and the audio table render functions.

    Built once per eval firing; the heavy objects (model, dataset, refs) outlive
    individual eval calls and are passed by reference.
    """
    iter_num: int
    unoptimized_model: nn.Module       # for generate_audio + compute_inference_data_loss
    compiled_model: nn.Module          # for estimate_loss
    sampling_rate: int
    device: str
    cfg: DictConfig
    dev_dataset: Any                   # DatasetWrapper — typed Any to avoid forward-decl noise
    table_2_refs: list
    custom_prompts: list
    overfit_ref_batch: Optional[dict]  # cached at iter 0 when overfit_batch is in audio_tables


def render_random_dev_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Pick one random ≥10s dev clip, generate audio at 5s + 10s prompt lengths."""
    sr = deps.sampling_rate
    ten_seconds_samples = int(10.0 * sr)
    five_seconds_samples = int(5.0 * sr)

    random_indices = list(range(len(deps.dev_dataset)))
    random.shuffle(random_indices)
    test_idx = next(i for i in random_indices if deps.dev_dataset.dataset[i]["audio_length"] >= ten_seconds_samples)
    sample = deps.dev_dataset[test_idx]

    audio_np = sample["audio"].numpy()
    max_start = sample["audio_length"] - ten_seconds_samples
    start_idx = random.randint(0, max_start)

    prompt_5s_np = audio_np[start_idx : start_idx + five_seconds_samples]
    prompt_10s_np = audio_np[start_idx : start_idx + ten_seconds_samples]

    target_text = deps.custom_prompts[random.randint(0, len(deps.custom_prompts) - 1)]

    rows = []
    for p_len, p_np in [(5.0, prompt_5s_np), (10.0, prompt_10s_np)]:
        gen_audio_np, length = generate_audio(deps.unoptimized_model, p_np, target_text=target_text)
        rows.append([
            deps.iter_num,
            p_len,
            target_text,
            wandb.Audio(p_np, sample_rate=sr),
            wandb.Audio(gen_audio_np[:length], sample_rate=sr),
        ])

    return (
        "Evaluation: Random Generation Examples",
        ["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt", "Speech-Prompt", "Generated Audio"],
        rows,
    )


def render_fixed_dev_refs_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the 6 fixed dev references created at startup."""
    sr = deps.sampling_rate
    rows = []
    for ref in deps.table_2_refs:
        gen_audio_np, length = generate_audio(deps.unoptimized_model, ref["prompt_tensor"], target_text=ref["text"])
        rows.append([
            deps.iter_num,
            deps.cfg.model.prompt_seconds,
            ref["text"],
            ref["original_audio"],
            ref["prompt_audio"],
            wandb.Audio(gen_audio_np[:length], sample_rate=sr),
        ])
    return (
        "Evaluation: Original vs. Generated",
        ["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt", "Original Audio", "Speech-Prompt", "Generated Audio"],
        rows,
    )


def render_overfit_batch_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the cached overfit batch; deterministic first prompt_seconds slice as the prompt."""
    batch = deps.overfit_ref_batch
    sr = deps.sampling_rate
    prompt_samples = int(deps.cfg.model.prompt_seconds * sr)

    audio_full = batch["audio"].to(deps.device)                # [B, T]
    audio_lengths_full = batch["audio_lengths"].to(deps.device)

    num_compare = min(2, audio_full.shape[0])
    rows = []
    for i in range(num_compare):
        T_i = int(audio_lengths_full[i].item())
        prompt_T = min(prompt_samples, T_i)
        ref_audio_slice = audio_full[i, :prompt_T]

        gen_audio_np, length = generate_audio(deps.unoptimized_model, ref_audio_slice, target_text=batch["text"][i])

        original_np = audio_full[i, :T_i].detach().cpu().to(torch.float32).numpy()
        prompt_np = ref_audio_slice.detach().cpu().to(torch.float32).numpy()

        rows.append([
            deps.iter_num,
            i,
            batch["text"][i],
            wandb.Audio(original_np, sample_rate=sr),
            wandb.Audio(prompt_np, sample_rate=sr),
            wandb.Audio(gen_audio_np[:length], sample_rate=sr),
        ])

    return (
        "overfit_audio_comparison",
        ["iter", "clip_idx", "text", "original", "prompt", "generated"],
        rows,
    )


AUDIO_TABLES = {
    "overfit_batch": render_overfit_batch_table,
    "random_dev": render_random_dev_table,
    "fixed_dev_refs": render_fixed_dev_refs_table,
}


def _save_safetensors(model: nn.Module, path) -> None:
    """Save model state_dict to safetensors, cloning each tensor to fresh storage.

    `safetensors.torch.save_model` errors on nn.LSTM `weight_ih_l0` because the
    LSTM's internal `_flat_weights` aliases the named parameter — the dedup pass
    inside save_model picks a single name that doesn't cover the full storage
    and bails. Cloning each tensor before writing sidesteps that path.
    Encodec's LSTM in the frozen encoder is the concrete trigger here.
    """
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(path))


def _collect_dropout_keys(cfg_model) -> dict:
    """Walks cfg.model recursively, returns {dot_path: value} for every key ending in
    `_dropout` or named `dropout`. Used by the overfit-mode startup assertion."""
    out = {}
    def _walk(node, prefix):
        if isinstance(node, (dict, DictConfig)):
            for k, v in node.items():
                dot = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, DictConfig)):
                    _walk(v, dot)
                elif k == "dropout" or k.endswith("_dropout"):
                    out[dot] = v
    _walk(cfg_model, "")
    return out


def run_eval_block(
    deps: EvalDeps,
    train_loader,
    dev_loader,
    test_loader,
    loss_wrapper: LossWrapper,
    ema: Optional[EMA],
    best_dev_loss: float,
    incremental_audio_tables: dict,
) -> float:
    """Single eval pass under EMA-swapped weights. Returns possibly-updated best_dev_loss.

    Toggles unoptimized_model.eval() at entry and unoptimized_model.train() in a
    finally clause on exit. Caller must NOT wrap this call in ema.swap_in — the
    bracket is managed internally; EMA.swap_in is non-reentrant.
    """
    eval_start_time = time.perf_counter()
    logger.info("Running evaluation block...")
    deps.unoptimized_model.eval()
    eval_payload: dict = {}
    losses = None

    try:
        with ema.swap_in(deps.unoptimized_model) if ema is not None else nullcontext():
            if deps.cfg.setup.eval_iters > 0:
                losses = estimate_loss(
                    deps.compiled_model, train_loader, dev_loader, test_loader,
                    loss_wrapper, deps.cfg.setup.eval_iters,
                    deps.cfg.setup.gradient_accumulation_steps, deps.device, deps.cfg,
                )
                logger.info(f"Step {deps.iter_num}: train loss {losses['train']['total_loss']:.4f}, "
                            f"dev loss {losses['dev']['total_loss']:.4f}")
                eval_payload["Evaluation: Metrics/Train-Loss"] = losses['train']['total_loss']
                eval_payload["Evaluation: Metrics/Dev-Loss"] = losses['dev']['total_loss']
                for k, v in losses['train']['logged_losses'].items():
                    eval_payload[f"{get_loss_section(k, 'Evaluation', 'Train')}/{k}"] = v
                for k, v in losses['dev']['logged_losses'].items():
                    eval_payload[f"{get_loss_section(k, 'Evaluation', 'Dev')}/{k}"] = v

            if deps.cfg.setup.inference_data_loss_sweep and deps.overfit_ref_batch is not None:
                # Training data_loss is a one-step prediction error from a known noisy z_t;
                # inference chains N ODE steps from t=1 to t≈0, each using the network's prediction.
                # Sweep step counts to distinguish solver-discretization error (monotonic drop to
                # ~training data_loss) from a weights-side gap (flat across step counts).
                logger.info("Computing inference_data_loss diagnostic...")
                sweep = tuple(deps.cfg.setup.inference_data_loss_sweep)
                sweep_losses = compute_inference_data_loss(
                    deps.unoptimized_model, deps.overfit_ref_batch, sampling_steps_sweep=sweep,
                )
                for n_steps, loss_val in sweep_losses.items():
                    logger.info(f"  inference_data_loss @ {n_steps} steps = {loss_val:.6f}")
                    eval_payload[f"eval/overfit_inference_data_loss_steps_{n_steps}"] = loss_val

            if deps.cfg.setup.audio_tables:
                logger.info("Generating audio samples for evaluation...")
                for table_name in deps.cfg.setup.audio_tables:
                    wandb_key, columns, rows = AUDIO_TABLES[table_name](deps)
                    # INCREMENTAL so each eval's rows accumulate into one table
                    # (compare audio across iters); a fresh table per eval would
                    # make the panel show only the latest step's audio.
                    table = incremental_audio_tables.get(wandb_key)
                    if table is None:
                        table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
                        incremental_audio_tables[wandb_key] = table
                    for row in rows:
                        table.add_data(*row)
                    eval_payload[wandb_key] = table

            if (
                deps.cfg.setup.best_safetensors
                and losses is not None
                and losses['dev']['total_loss'] < best_dev_loss
            ):
                best_dev_loss = losses['dev']['total_loss']
                logger.info(f"Saving new best model to {CHECKPOINTS_DIR}")
                best_path = CHECKPOINTS_DIR / 'ema_best.safetensors'
                best_tmp_path = CHECKPOINTS_DIR / 'ema_best.tmp.safetensors'
                best_bak_path = CHECKPOINTS_DIR / 'ema_best_bak.safetensors'
                _save_safetensors(deps.unoptimized_model, best_tmp_path)
                if best_path.exists():
                    best_path.replace(best_bak_path)
                best_tmp_path.replace(best_path)
    finally:
        deps.unoptimized_model.train()

    if deps.cfg.wandb.log and eval_payload:
        wandb.log(eval_payload, step=deps.iter_num)

    logger.info(f"Eval block done in {time.perf_counter() - eval_start_time:.1f}s")
    return best_dev_loss


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires an NVIDIA GPU and CUDA installed, but none were detected.")

    device = cfg.setup.device
    device_type = 'cuda'

    # Seed torch + Python random so model init, diffusion t/ε sampling, prompt
    # windows, eval-block prompt picks, and DataLoader worker RNG state are
    # reproducible across runs. cuDNN-level determinism is NOT enforced (would
    # cost ~5–10% perf and force slower compiled kernels) — sufficient for
    # fair A/B comparisons. The bucketed sampler uses np.random.default_rng
    # with its own hardcoded seed so the 5 overfit batches stay fixed
    # regardless of cfg.seed.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Startup assertions — catch config-pilot-error before the first eval fires.
    if "overfit_batch" in cfg.setup.audio_tables:
        assert cfg.setup.overfit_single_batch, (
            "audio_tables contains 'overfit_batch' but overfit_single_batch=False. "
            "The overfit_batch table requires cycling-batch semantics to give a stable "
            "reference batch across eval firings."
        )
    if cfg.setup.inference_data_loss_sweep:
        assert "overfit_batch" in cfg.setup.audio_tables, (
            "inference_data_loss_sweep is non-empty but 'overfit_batch' is not in "
            "audio_tables — the sweep currently uses the cached overfit batch as its "
            "input. Set inference_data_loss_sweep: [] if you don't want the sweep."
        )
    if cfg.setup.overfit_single_batch:
        # Catches the case where model/base.yaml adds a new dropout key and the
        # per-experiment model/overfit_test.yaml forgets to override it.
        nonzero_dropouts = {k: v for k, v in _collect_dropout_keys(cfg.model).items() if v != 0.0}
        assert not nonzero_dropouts, (
            f"overfit_single_batch=True but these dropout keys are non-zero in cfg.model: "
            f"{nonzero_dropouts}. Add them to model/overfit_test.yaml overrides."
        )

    log_dir = PROJECT_ROOT / cfg.setup.log_subdir
    log_name = f"{cfg.setup.log_name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Ensure the safetensors + ckpt.pt save targets exist before any write.
    # CHECKPOINTS_DIR (= models/checkpoints) may be absent on fresh containers
    # that have models/ from encodec setup but no checkpoints subdir yet.
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
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
    # Persistent INCREMENTAL wandb.Tables, keyed by table name, so eval audio
    # accumulates across eval firings instead of each eval overwriting the last.
    incremental_audio_tables: dict = {}

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
        resume_ema_state = checkpoint.get('ema')
        resume_cur_kimg = checkpoint.get('cur_kimg', 0.0)
        logger.info("Resumed optimizer from checkpoint.")
    else:
        resume_wandb_id = None
        resume_ema_state = None
        resume_cur_kimg = 0.0

    # Free memory
    checkpoint = None

    ema = EMA(model, halflife_kimg=cfg.model.ema.halflife_kimg) if cfg.model.ema.enabled else None
    cur_kimg = resume_cur_kimg
    if ema is not None and resume_ema_state is not None:
        ema.load_state_dict(resume_ema_state)
        if ema.halflife_kimg != cfg.model.ema.halflife_kimg:
            logger.warning(
                f"halflife_kimg changed: checkpoint={ema.halflife_kimg}, "
                f"config={cfg.model.ema.halflife_kimg}. Using config value."
            )
            ema.halflife_kimg = cfg.model.ema.halflife_kimg
    elif ema is not None and cfg.setup.init_from == 'resume':
        logger.warning(
            "EMA enabled but no EMA state in checkpoint. "
            "Initializing fresh EMA from current model weights — "
            "the shadow will need to re-converge."
        )

    logger.info("Compiling the model... (this takes a minute)")
    unoptimized_model = model
    model = torch.compile(model)

    # Declared outside the wandb.log gate so render_fixed_dev_refs_table can
    # iterate it safely on `wandb.log: False` debug runs (empty list → no rows).
    table_2_refs = []
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

    if cfg.setup.loss_analysis_run:
        loss_analysis_start_iter = int(cfg.setup.max_iters * 0.8)   # Last 20% of steps for loss analysis to ensure stable logged values
        loss_analysis_accumulators = {}
        loss_analysis_steps_counted = 0
        
    if cfg.setup.gradient_analysis_run:
        grad_analysis_start_iter = int(cfg.setup.max_iters * 0.8)   # Last 20% of steps for gradient analysis to ensure stable logged values
        grad_norm_accumulators = {}
        cos_sim_accumulators = {}
        grad_analysis_steps_counted = 0
        
    logger.info("Starting training loop...")
    last_log_time = time.perf_counter()
    last_log_iter = start_iter - 1

    # Filled on the first loop iter when the overfit_batch table is active.
    # Overfit cycling (get_infinite_batches) yields the same 5 Python objects
    # forever, so caching `lookahead_queue[0]` once gives every subsequent eval
    # firing a stable reference batch.
    overfit_ref_batch = None

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
        examples_this_step = sum(b["audio"].shape[0] for b in lookahead_queue)

        # Cache the overfit batch once: overfit cycling guarantees lookahead_queue[0]
        # is the same Python object every iter, so a single capture suffices.
        if overfit_ref_batch is None and "overfit_batch" in cfg.setup.audio_tables:
            overfit_ref_batch = lookahead_queue[0]

        # -----------------------------
        # Evaluation
        # -----------------------------
        if iter_num > 0 and iter_num % cfg.setup.eval_interval == 0:
            eval_deps = EvalDeps(
                iter_num=iter_num,
                unoptimized_model=unoptimized_model,
                compiled_model=model,
                sampling_rate=sampling_rate,
                device=device,
                cfg=cfg,
                dev_dataset=dev_dataset,
                table_2_refs=table_2_refs,
                custom_prompts=custom_prompts,
                overfit_ref_batch=overfit_ref_batch,
            )
            eval_start_time = time.perf_counter()
            best_dev_loss = run_eval_block(
                eval_deps,
                train_loader, dev_loader, test_loader,
                loss_wrapper, ema, best_dev_loss,
                incremental_audio_tables,
            )
            last_log_time += time.perf_counter() - eval_start_time

        # -----------------------------
        # Forward & Backward Pass
        # -----------------------------
        
        accum_loss = torch.zeros((), device=device)
        accum_logged_losses = {}
        
        analyzer = GradientAnalyzer() if (cfg.setup.gradient_analysis_run and iter_num >= grad_analysis_start_iter and iter_num % cfg.setup.log_interval == 0) else None
        
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
        if ema is not None:
            ema.update(unoptimized_model, batch_size=examples_this_step, cur_kimg=cur_kimg)
        optimizer.zero_grad(set_to_none=True)
        cur_kimg += examples_this_step / 1000.0

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

            if cfg.wandb.log:
                log_payload = {
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

                    if iter_num >= grad_analysis_start_iter:
                        grad_analysis_steps_counted += 1
                        for k, v in grad_norms.items():
                            grad_norm_accumulators[k] = grad_norm_accumulators.get(k, 0.0) + v
                        for k, v in cos_sims.items():
                            cos_sim_accumulators[k] = cos_sim_accumulators.get(k, 0.0) + v

                if ema is not None:
                    log_payload["EMA/effective_decay"] = ema._effective_decay(
                        batch_size=examples_this_step, cur_kimg=cur_kimg,
                    )
                    log_payload["EMA/effective_halflife_kimg"] = min(cur_kimg, ema.halflife_kimg)
                    log_payload["EMA/cur_kimg"] = cur_kimg
                    # RMS drift between live and shadow weights — grows during
                    # training then plateaus once the shadow centers on the
                    # SGD oscillation around the loss minimum.
                    with torch.no_grad():
                        drift_sq = torch.zeros((), device=device)
                        n_drift = 0
                        for name, p in unoptimized_model.named_parameters():
                            if name in ema.shadow:
                                drift_sq += (p.data.float() - ema.shadow[name]).pow(2).sum()
                                n_drift += p.numel()
                    log_payload["EMA/weight_drift_rms"] = (drift_sq / max(n_drift, 1)).sqrt().item()

                wandb.log(log_payload, step=iter_num)

            # Accumulate unweighted raw losses exclusively for the analysis table
            if cfg.setup.loss_analysis_run and iter_num >= loss_analysis_start_iter:
                loss_analysis_steps_counted += 1
                for k, v in accum_logged_losses.items():
                    if not k.endswith("_weighted"):
                        loss_analysis_accumulators[k] = loss_analysis_accumulators.get(k, 0.0) + v.item()

        # -----------------------------
        # Periodic crash-recovery checkpoint (.pt) — decoupled from log_interval
        # -----------------------------
        if (
            cfg.setup.checkpoint_interval > 0
            and iter_num > 0
            and iter_num % cfg.setup.checkpoint_interval == 0
        ):
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
                'batch_idx': current_batch_idx + 1,   # Index of the upcoming batch
                'cur_kimg': cur_kimg,
            }
            if ema is not None:
                checkpoint_data['ema'] = ema.state_dict()

            ckpt_path = CHECKPOINTS_DIR / 'ckpt.pt'
            ckpt_tmp_path = CHECKPOINTS_DIR / 'ckpt.pt.tmp'
            ckpt_bak_path = CHECKPOINTS_DIR / 'ckpt_bak.pt'

            torch.save(checkpoint_data, ckpt_tmp_path)
            if ckpt_path.exists():
                ckpt_path.replace(ckpt_bak_path)
            ckpt_tmp_path.replace(ckpt_path)

    # -----------------------------
    # Loss Analysis Summary Dump
    # -----------------------------
    if cfg.setup.loss_analysis_run:
        logger.info("========== LOSS ANALYSIS SUMMARY ==========")
        logger.info(f"Analyzed over the last {loss_analysis_steps_counted} logged steps (between iterations {loss_analysis_start_iter} and {cfg.setup.max_iters - 1}).")
        logger.info("Average raw unweighted loss magnitudes:")
        for k, v in loss_analysis_accumulators.items():
            avg = v / loss_analysis_steps_counted
            logger.info(f"  {k}: {avg:.4f}")
        logger.info("===========================================")

    # -----------------------------
    # Gradient Analysis Summary Dump
    # -----------------------------
    if cfg.setup.gradient_analysis_run:
        logger.info("========== GRADIENT ANALYSIS SUMMARY ==========")
        logger.info(f"Analyzed over the last {grad_analysis_steps_counted} logged steps (between iterations {grad_analysis_start_iter} and {cfg.setup.max_iters - 1}).")
        logger.info("Average Gradient L2-Norms (Shared Backbone):")
        for k, v in grad_norm_accumulators.items():
            if not k.endswith("_total"):
                avg = v / grad_analysis_steps_counted
                logger.info(f"  {k}: {avg:.4f}")
        logger.info("Average Gradient Cosine Similarities (Shared Backbone):")
        for k, v in cos_sim_accumulators.items():
            avg = v / grad_analysis_steps_counted
            logger.info(f"  {k}: {avg:.4f}")
        logger.info("===============================================")

    # -----------------------------
    # Final eval pass + inference-iterable safetensors save
    # -----------------------------
    # The eval block runs once more at iter_num=max_iters under EMA shadow
    # weights, then a separate ema.swap_in bracket writes the final safetensors
    # (EMA.swap_in is non-reentrant, so the two cannot be nested).
    final_eval_deps = EvalDeps(
        iter_num=cfg.setup.max_iters,
        unoptimized_model=unoptimized_model,
        compiled_model=model,
        sampling_rate=sampling_rate,
        device=device,
        cfg=cfg,
        dev_dataset=dev_dataset,
        table_2_refs=table_2_refs,
        custom_prompts=custom_prompts,
        overfit_ref_batch=overfit_ref_batch,
    )
    best_dev_loss = run_eval_block(
        final_eval_deps,
        train_loader, dev_loader, test_loader,
        loss_wrapper, ema, best_dev_loss,
        incremental_audio_tables,
    )

    if cfg.setup.final_safetensors:
        final_path = CHECKPOINTS_DIR / 'ema_final.safetensors'
        with ema.swap_in(unoptimized_model) if ema is not None else nullcontext():
            _save_safetensors(unoptimized_model, final_path)
        logger.info(f"Saved final EMA weights for offline inference to {final_path}")


if __name__ == "__main__":
    # Enable PyTorch Memory Expansion to heavily mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train()