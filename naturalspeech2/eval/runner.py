"""Process-agnostic eval core, shared by the in-process trainer path and the decoupled eval
daemon. Pure compute → plain data; no wandb, no IPC, no process assumptions.

Contents:
  - estimate_loss: held-out loss over dev/test (+train subset), denominator-normalized.
  - build_fixed_refs_data / FixedRef: deterministic reference clips, GT-floor WER cached once.
  - generate_ref_audio: one ref → (generated waveform, synth WER, ASR transcription).
  - run_decoupled_eval: daemon orchestration — both live + EMA losses, EMA audio/WER, best-pick.
  - get_loss_section / _mean_skip_nan / atomic_save_safetensors: shared formatting + IO helpers.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

from naturalspeech2.eval.metrics import compute_wer
from naturalspeech2.inference import generate_audio
from naturalspeech2.utils.utils import compute_denominators


# ----------------------------------------------------------------------------
# Shared logging-key + aggregation helpers
# ----------------------------------------------------------------------------

def get_loss_section(key: str, phase: str, split: str = "") -> str:
    """W&B section for a loss leaf. Groups diffusion / aligner terms; (Raw) vs (Weighted)."""
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
    return f"Evaluation: {split}-{group} {weight_str}"


def _mean_skip_nan(xs: list) -> float:
    """Mean over non-NaN values (NaN if all dropped) — a degenerate clip's WER drops out."""
    vals = [x for x in xs if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


def atomic_save_safetensors(model: nn.Module, path, *, tmp_suffix=".tmp", bak_suffix="_bak") -> None:
    """Atomic safetensors write: clone each tensor to fresh storage (dodges save_model's LSTM
    _flat_weights dedup crash), serialize to tmp, rotate prev → bak, replace.
    """
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    tmp_path = path.with_name(path.stem + tmp_suffix + path.suffix)
    bak_path = path.with_name(path.stem + bak_suffix + path.suffix)
    save_file(state_dict, str(tmp_path))
    if path.exists():
        path.replace(bak_path)
    tmp_path.replace(path)


# ----------------------------------------------------------------------------
# Held-out loss
# ----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_loader, dev_loader, test_loader, loss_wrapper, eval_iters,
                  grad_accum_steps, device, cfg, eval_train: bool = True):
    """Denominator-normalized held-out loss. dev+test chained as one 'dev' pool (separate only
    when dev IS the train split). train subset (random, eval_iters cap) gated behind eval_train.
    Returns {split: {total_loss, logged_losses}}. Toggles model.eval()/train() around the pass.
    """
    out = {}
    model.eval()

    if cfg.dataset.train_split != cfg.dataset.dev_split:
        held_out_splits = [('dev', itertools.chain(dev_loader, test_loader))]
    else:
        held_out_splits = [('dev', dev_loader), ('test', test_loader)]

    if eval_train:
        # Store the (shared) train sampler state; perturb to a random subset from 0; restore after.
        train_sampler = train_loader.batch_sampler
        main_train_epoch = train_sampler.epoch
        main_train_batch_idx = train_sampler.start_batch_idx
        train_sampler.set_epoch(random.randint(0, 10000))
        train_sampler.set_start_batch_idx(0)
        splits = [('train', train_loader)] + held_out_splits
    else:
        splits = held_out_splits

    for split, loader in splits:
        loader_iter = iter(loader)

        eval_total_loss_sum = torch.zeros((), device=device)
        eval_log_dict_sums = {}
        actual_eval_iters = 0

        for _ in range(eval_iters):
            eval_lookahead_queue = []
            try:
                for _ in range(grad_accum_steps):
                    eval_lookahead_queue.append(next(loader_iter))
            except StopIteration:
                break  # drop incomplete logical batch, end this split

            eval_denominators = compute_denominators(eval_lookahead_queue, cfg)

            eval_accum_loss = torch.zeros((), device=device)
            eval_accum_logged_losses = {}

            for batch in eval_lookahead_queue:
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

        if actual_eval_iters == 0:
            continue  # split had fewer than grad_accum_steps batches — skip it

        divisor = actual_eval_iters
        out[split] = {
            'total_loss': (eval_total_loss_sum / divisor).item(),
            'logged_losses': {key: (val / divisor).item() for key, val in eval_log_dict_sums.items()},
        }

    if eval_train:
        train_sampler.set_epoch(main_train_epoch)
        train_sampler.set_start_batch_idx(main_train_batch_idx)

    model.train()
    return out


# ----------------------------------------------------------------------------
# Fixed reference clips (data-only) + per-ref audio generation
# ----------------------------------------------------------------------------

@dataclass
class FixedRef:
    """A deterministic eval reference clip. Raw arrays only — wandb wrapping happens at the
    logging boundary (trainer wraps wandb.Audio; daemon writes wav files)."""
    original_np: np.ndarray
    prompt_np: np.ndarray
    prompt_tensor: torch.Tensor
    text: str
    gt_wer: Optional[float]  # GT-floor WER (ASR on the real clip), cached once at build; None if WER off


def build_fixed_refs_data(dataset, n_refs, prompt_samples_len, rng, *, sampling_rate,
                          compute_gt_wer: bool) -> list[FixedRef]:
    """Pick a fixed, reproducible set of ≥prompt-length clips. Seeded rng → identical across runs
    AND across the trainer/daemon processes. GT-floor WER computed ONCE here (was re-run every
    eval on identical audio).
    """
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    refs: list[FixedRef] = []
    for idx in indices:
        if len(refs) == n_refs:
            break
        if dataset.dataset[idx]["audio_length"] < prompt_samples_len:
            continue
        sample = dataset[idx]
        audio_np = sample["audio"].numpy()
        start_idx = rng.randint(0, sample["audio_length"] - prompt_samples_len)
        prompt_audio_np = audio_np[start_idx: start_idx + prompt_samples_len]
        gt = compute_wer(audio_np, sample["text"], src_sr=sampling_rate)[0] if compute_gt_wer else None
        refs.append(FixedRef(
            original_np=audio_np,
            prompt_np=prompt_audio_np,
            prompt_tensor=torch.from_numpy(prompt_audio_np),
            text=sample["text"],
            gt_wer=gt,
        ))
    return refs


def generate_ref_audio(model, prompt_tensor, text: str, sampling_rate: int, do_wer: bool):
    """One ref → (trimmed generated waveform np, synth WER or None, ASR transcription or None).
    GT-floor is cached on the ref (build time); only the synth WER + transcription are computed here.
    The transcription is the raw ASR hypothesis on the generated audio (debugging intelligibility)."""
    gen_audio_np, length = generate_audio(model, prompt_tensor, target_text=text)
    gen = gen_audio_np[:length]
    synth_wer, hyp = compute_wer(gen, text, src_sr=sampling_rate) if do_wer else (None, None)
    return gen, synth_wer, hyp


# ----------------------------------------------------------------------------
# Decoupled-eval orchestration (daemon) → structured report
# ----------------------------------------------------------------------------

@dataclass
class AudioClip:
    """Audio cell in an eval table row. Serialized to a wav path for the parent; the trainer
    in-process path wraps it directly as wandb.Audio."""
    waveform: np.ndarray
    sample_rate: int


@dataclass
class EvalReport:
    """Everything one decoupled eval produces — serialized to the results dir for the parent."""
    snapshot_step: int
    scalars: dict = field(default_factory=dict)          # final wandb metric name -> float
    audio_tables: dict = field(default_factory=dict)     # table_key -> {"columns": [...], "rows": [[cell|AudioClip,...]]}
    new_best: bool = False
    best_dev_loss: float = float("inf")


# table_name -> (wandb title, split label for metric keys, ref source: 'dev' | 'test' | 'val')
FIXED_REF_TABLES = {
    "fixed_dev_refs":  ("Eval Audio: dev (trained-on)", "dev", "dev"),
    "fixed_test_refs": ("Eval Audio: test (held-out)", "test", "test"),
    "fixed_val_refs":  ("Eval Audio: dev+test (held-out validation)", "dev", "val"),
}


def _load_trainable(model: nn.Module, state: dict) -> None:
    """Copy a trainable-param snapshot into the resident model in place. Frozen params + buffers
    (Encodec, RoPE caches, latent stats) are reconstructed identically at build → left untouched."""
    for name, p in model.named_parameters():
        src = state.get(name)
        if src is not None:
            p.data.copy_(src.to(device=p.device, dtype=p.dtype))


def _record_losses(scalars: dict, losses: dict, suffix: str = "") -> None:
    """Flatten estimate_loss output into wandb-keyed scalars. EMA → suffix='' (primary);
    live → suffix='-live'."""
    for split_name, section in [('train', 'Train'), ('dev', 'Dev'), ('test', 'Test')]:
        if split_name not in losses:
            continue
        scalars[f"Evaluation: Metrics/{section}-Loss{suffix}"] = losses[split_name]['total_loss']
        for k, v in losses[split_name]['logged_losses'].items():
            scalars[f"{get_loss_section(k, 'Evaluation', section)}/{k}{suffix}"] = v


def run_decoupled_eval(
    *,
    model: nn.Module,
    loss_model: nn.Module,
    loss_wrapper,
    train_loader,
    dev_loader,
    test_loader,
    live_trainable: dict,
    shadow_trainable: dict,
    dev_refs: list[FixedRef],
    test_refs: list[FixedRef],
    cfg,
    device: str,
    prompt_seconds: float,
    sampling_rate: int,
    snapshot_step: int,
    prev_best_dev_loss: float,
) -> EvalReport:
    """Daemon eval over one weight snapshot, on a single resident model:
      1. load LIVE trainable → losses ('-live').
      2. load EMA shadow → losses (primary) + audio/WER (shipped model).
      3. best-ckpt gate on EMA dev loss.
    `model` is the eager handle (clean param names — used for weight-load + audio gen);
    `loss_model` is the handle for estimate_loss (compiled if enabled; shares the same params).
    Returns an EvalReport (scalars + audio rows + best flag) — caller handles IO/wandb.
    """
    report = EvalReport(snapshot_step=snapshot_step, best_dev_loss=prev_best_dev_loss)
    eval_iters = cfg.setup.eval_iters
    gas = cfg.setup.gradient_accumulation_steps

    # Set warmup-ramped weights (e.g. duration_predictor over 1000 steps) to this snapshot's step →
    # weighted losses + EMA-dev best-pick match the trainer. The daemon's LossWrapper is a fresh
    # instance (frozen at step 0) and estimate_loss calls it without a step.
    loss_wrapper._update_weights(snapshot_step)

    # --- 1. live losses (overfitting signal, comparable to the per-step train curve) ---
    if eval_iters > 0:
        _load_trainable(model, live_trainable)
        live = estimate_loss(loss_model, train_loader, dev_loader, test_loader, loss_wrapper,
                             eval_iters, gas, device, cfg, eval_train=True)
        _record_losses(report.scalars, live, suffix="-live")

    # --- 2. EMA losses (primary, best-selection + shipped quality) ---
    _load_trainable(model, shadow_trainable)
    ema_losses = None
    if eval_iters > 0:
        ema_losses = estimate_loss(loss_model, train_loader, dev_loader, test_loader, loss_wrapper,
                                   eval_iters, gas, device, cfg, eval_train=True)
        _record_losses(report.scalars, ema_losses, suffix="")

    # --- 2b. EMA audio + WER on fixed refs ---
    # estimate_loss leaves the model in train() mode → force eval() so generation runs with
    # dropout OFF (the predictors use dropout up to 0.5; sampling under it is wrong).
    model.eval()
    do_wer = "wer" in cfg.setup.eval_metrics
    num_table_rows = cfg.setup.num_table_rows
    # val = dev+test interleaved → the first num_table_rows stay balanced across both splits.
    val_refs = [r for pair in itertools.zip_longest(dev_refs, test_refs) for r in pair if r is not None]
    refs_by_source = {"dev": dev_refs, "test": test_refs, "val": val_refs}
    for table_name in cfg.setup.audio_tables:
        if table_name not in FIXED_REF_TABLES:
            continue  # random_dev / overfit_batch are in-process-only; skip in the daemon
        title, split, source = FIXED_REF_TABLES[table_name]
        refs = refs_by_source[source]
        columns = ["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt",
                   "Original Audio", "Speech-Prompt", "Generated Audio"]
        if do_wer:
            columns += ["Transcription", "WER"]
        # WER is the mean over ALL refs (well-sampled metric); only the first num_table_rows are
        # rendered as wandb rows (keeps the audio table small as num_audio_refs scales up).
        rows, synth_wers, gt_wers = [], [], []
        for i, ref in enumerate(refs):
            in_table = i < num_table_rows
            if not do_wer and not in_table:
                continue  # needed for neither the WER metric nor a table row → skip generation
            gen, synth_wer, hyp = generate_ref_audio(model, ref.prompt_tensor, ref.text, sampling_rate, do_wer)
            if do_wer:
                synth_wers.append(synth_wer)
                gt_wers.append(ref.gt_wer)
            if in_table:
                row = [snapshot_step, prompt_seconds, ref.text,
                       AudioClip(ref.original_np, sampling_rate),
                       AudioClip(ref.prompt_np, sampling_rate),
                       AudioClip(gen, sampling_rate)]
                if do_wer:
                    row += [hyp, synth_wer]
                rows.append(row)
        if do_wer:
            report.scalars[f"Evaluation: Metrics/{split}-WER"] = _mean_skip_nan(synth_wers)
            report.scalars[f"Evaluation: Metrics/{split}-WER-gt"] = _mean_skip_nan(gt_wers)
        report.audio_tables[title] = {"columns": columns, "rows": rows}

    # --- 3. best-ckpt decision (EMA dev loss) ---
    if (cfg.setup.best_safetensors and ema_losses is not None and 'dev' in ema_losses
            and ema_losses['dev']['total_loss'] < prev_best_dev_loss):
        report.new_best = True
        report.best_dev_loss = ema_losses['dev']['total_loss']

    return report
