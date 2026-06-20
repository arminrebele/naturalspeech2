"""Process-agnostic eval core, shared by the in-process trainer path and the decoupled eval
daemon. Pure compute → plain data; no wandb, no IPC, no process assumptions.

Contents:
  - estimate_loss: held-out loss over the pooled val set (dev+test) (+train subset), denominator-normalized.
  - build_fixed_refs_data / FixedRef: deterministic reference clips, GT-floor WER cached once.
  - batch_generate: length-sorted, frame-budget-packed batched generation (packer: utils.pack_by_budget).
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

from naturalspeech2.eval.metrics import compute_sim_o, compute_wer, compute_wer_batch, speaker_embedding
from naturalspeech2.inference import generate_audio_batch
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
from naturalspeech2.utils.utils import compute_denominators, pack_by_budget


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


def _dist_stats_skip_nan(xs: list) -> dict:
    """Mean + p10/median/p90 over non-NaN values (all-NaN → all NaN). Spread companions to the mean;
    median/p90 resist the occasional catastrophic-WER clip that drags the mean."""
    vals = [x for x in xs if x == x]
    if not vals:
        return {k: float("nan") for k in ("", "-p10", "-median", "-p90")}
    arr = np.asarray(vals, dtype=float)
    return {"": float(arr.mean()), "-p10": float(np.percentile(arr, 10)),
            "-median": float(np.percentile(arr, 50)), "-p90": float(np.percentile(arr, 90))}


def record_metric_dist(scalars: dict, base_key: str, values: list) -> None:
    """Write mean + p10/median/p90 of `values` to scalars under base_key + ('', -p10, -median, -p90)."""
    for suffix, v in _dist_stats_skip_nan(values).items():
        scalars[f"{base_key}{suffix}"] = v


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
    """Denominator-normalized held-out loss. dev+test are always chained into one pooled 'val' set
    (the project's real test set is external — VCTK + LibriSpeech). train subset (random, eval_iters
    cap) gated behind eval_train. Returns {split: {total_loss, logged_losses}} with
    split ∈ {'train','val'}. Toggles model.eval()/train() around the pass.
    """
    out = {}
    model.eval()

    # dev+test = one pooled 'val' held-out set. Chaining two loaders == one concat for a
    # denominator-normalized sum (batch boundaries / per-loader bucketing don't change the number).
    held_out_splits = [('val', itertools.chain(dev_loader, test_loader))]

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
# Fixed reference clips (data-only) + batched audio generation
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
    prompt_embedding: Optional[torch.Tensor]  # cached WavLM-SV speaker embedding of the prompt; None if SIM-o off


def build_fixed_refs_data(datasets, n_refs, prompt_samples_len, rng, *, sampling_rate,
                          compute_gt_wer: bool, compute_sim_emb: bool) -> list[FixedRef]:
    """Pick a fixed, reproducible set of ≥prompt-length clips, pooled across `datasets` (a list of
    DatasetWrapper — e.g. [dev, test] → the held-out 'val' set, or [train_subset] → trained-on).
    ONE seeded rng over the union of (dataset, idx) candidates → a uniform draw, identical across
    runs AND across the trainer/daemon processes (same dataset order + seed). Cached ONCE here
    (was re-run every eval on identical audio): GT-floor WER, and — when SIM-o is on — the prompt's
    WavLM-SV speaker embedding.
    """
    candidates = [(ds, i) for ds in datasets for i in range(len(ds))]
    rng.shuffle(candidates)
    refs: list[FixedRef] = []
    for ds, idx in candidates:
        if len(refs) == n_refs:
            break
        if ds.dataset[idx]["audio_length"] < prompt_samples_len:
            continue
        sample = ds[idx]
        audio_np = sample["audio"].numpy()
        start_idx = rng.randint(0, sample["audio_length"] - prompt_samples_len)
        prompt_audio_np = audio_np[start_idx: start_idx + prompt_samples_len]
        gt = compute_wer(audio_np, sample["text"], src_sr=sampling_rate)[0] if compute_gt_wer else None
        emb = speaker_embedding(prompt_audio_np, src_sr=sampling_rate) if compute_sim_emb else None
        refs.append(FixedRef(
            original_np=audio_np,
            prompt_np=prompt_audio_np,
            prompt_tensor=torch.from_numpy(prompt_audio_np),
            text=sample["text"],
            gt_wer=gt,
            prompt_embedding=emb,
        ))
    return refs


def batch_generate(model, prompts, texts, frame_proxies, frame_budget):
    """Length-sorted, frame-budget-packed batched generation → trimmed gens in INPUT order.
    frame_proxies need only be a consistent relative length estimate (exact F' unknown pre-duration)."""
    n = len(prompts)
    if n == 0:
        return []
    gens = [None] * n
    for group in pack_by_budget(frame_proxies, frame_budget):
        # on_overflow="warn": an under-trained predictor mustn't OOM-kill a multi-day run → clamp + log.
        out = generate_audio_batch(model, [prompts[i] for i in group], [texts[i] for i in group],
                                   on_overflow="warn")
        for pos, i in enumerate(group):
            gens[i] = out[pos]
    return gens


# ----------------------------------------------------------------------------
# Random val-clip showcase (random_val table) — shared core
# ----------------------------------------------------------------------------

# Eval-block static text prompts (short → very long, ~3/6/15/30 s) → probe length generalization.
# Generation re-phonemizes per call (~50 ms each — negligible). Shared by both eval paths.
EVAL_TEXT_PROMPTS = [
    "Hello, world! This is a test.",
    "The quick brown fox jumps over the lazy dog, while the sun sets.",
    ("Natural speech synthesis has come a long way in recent years. "
     "Today, we can generate highly realistic human voices from just a "
     "few seconds of reference audio, opening up new possibilities for "
     "accessibility and content creation."),
    ("In the early days of artificial intelligence, text to speech systems "
     "sounded incredibly robotic and lacked emotional nuance. Researchers "
     "spent decades studying human phonetics, prosody, and intonation. "
     "Now, thanks to advanced deep learning techniques, diffusion models, "
     "and massive datasets, the boundaries between synthesized and natural "
     "voices are becoming indistinguishable. This marks a paradigm shift "
     "in how we interact with technology on a daily basis."),
]

RANDOM_VAL_TITLE = "Evaluation: Random Generation Examples"
RANDOM_VAL_COLUMNS = ["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt",
                      "Speech-Prompt", "Generated Audio"]


def generate_random_val_clips(model, val_datasets, sampling_rate: int, custom_prompts: list):
    """One random ≥10 s clip from the pooled val set (dev+test) + one random custom text → generate
    at 5 s & 10 s prompt lengths. Returns (target_text, [(prompt_len_s, prompt_np, gen_np), ...]);
    raw arrays, wandb/AudioClip wrapping at the boundary. Re-picked each eval (advances global RNG)
    — unlike the fixed refs."""
    sr = sampling_rate
    ten, five = int(10.0 * sr), int(5.0 * sr)

    candidates = [(ds, i) for ds in val_datasets for i in range(len(ds))]
    random.shuffle(candidates)
    ds, idx = next((d, i) for d, i in candidates if d.dataset[i]["audio_length"] >= ten)
    sample = ds[idx]
    audio_np = sample["audio"].numpy()
    start = random.randint(0, sample["audio_length"] - ten)
    target_text = custom_prompts[random.randint(0, len(custom_prompts) - 1)]

    prompts = [audio_np[start: start + n] for n in (five, ten)]
    gens = generate_audio_batch(model, prompts, [target_text, target_text], on_overflow="warn")
    clips = [(p_len, prompts[j], gens[j]) for j, p_len in enumerate((5.0, 10.0))]
    return target_text, clips


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
    best_val_loss: float = float("inf")


# table_name -> (wandb title, split label for metric keys, ref source: 'train' | 'val')
FIXED_REF_TABLES = {
    "fixed_train_refs": ("Evaluation: Trained-on Audio (train)", "train", "train"),
    "fixed_val_refs":   ("Evaluation: Held-out Audio (dev+test)", "val", "val"),
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
    for split_name, section in [('train', 'Train'), ('val', 'Val')]:
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
    live_trainable: Optional[dict],
    shadow_trainable: dict,
    val_refs: list[FixedRef],
    train_refs: list[FixedRef],
    val_datasets: list,
    cfg,
    device: str,
    prompt_seconds: float,
    sampling_rate: int,
    snapshot_step: int,
    prev_best_val_loss: float,
    eval_train: bool = True,
) -> EvalReport:
    """Daemon eval over one weight snapshot, on a single resident model:
      1. load LIVE trainable → losses ('-live')  [skipped if live_trainable is None — an EMA-only
         checkpoint, e.g. a standalone benchmark of ema_*.safetensors].
      2. load EMA shadow → losses (primary) + audio/WER (shipped model).
      3. best-ckpt gate on EMA val loss.
    `model` is the eager handle (clean param names — used for weight-load + audio gen);
    `loss_model` is the handle for estimate_loss (compiled if enabled; shares the same params).
    `eval_train=False` → held-out-only losses (no train-subset pass; standalone benchmark of a
    checkpoint without the train split preprocessed — train_loader may be None).
    Returns an EvalReport (scalars + audio rows + best flag) — caller handles IO/wandb.
    """
    report = EvalReport(snapshot_step=snapshot_step, best_val_loss=prev_best_val_loss)
    eval_iters = cfg.setup.eval_iters
    # Daemon eval runs a divisor-shrunk batch (lower VRAM on the 2nd GPU); multiply grad_accum by the
    # same divisor so the logical batch (gas × batch) and total samples (eval_iters × gas × batch) are
    # unchanged vs the in-process eval. Loss is a masked sum / shared per-step denominator → invariant
    # to the micro-batch split, so the metric is unchanged (modulo ~1-sample integer rounding).
    gas = cfg.setup.gradient_accumulation_steps * cfg.setup.eval_daemon.batch_size_divisor

    # Set warmup-ramped weights (e.g. duration_predictor over 1000 steps) to this snapshot's step →
    # weighted losses + EMA-dev best-pick match the trainer. The daemon's LossWrapper is a fresh
    # instance (frozen at step 0) and estimate_loss calls it without a step.
    loss_wrapper._update_weights(snapshot_step)

    # --- 1. live losses (overfitting signal, comparable to the per-step train curve) ---
    # live_trainable=None → EMA-only checkpoint (no live counterpart): skip rather than log a
    # '-live' series that just duplicates the EMA numbers below.
    if eval_iters > 0 and live_trainable is not None:
        _load_trainable(model, live_trainable)
        live = estimate_loss(loss_model, train_loader, dev_loader, test_loader, loss_wrapper,
                             eval_iters, gas, device, cfg, eval_train=eval_train)
        _record_losses(report.scalars, live, suffix="-live")

    # --- 2. EMA losses (primary, best-selection + shipped quality) ---
    _load_trainable(model, shadow_trainable)
    ema_losses = None
    if eval_iters > 0:
        ema_losses = estimate_loss(loss_model, train_loader, dev_loader, test_loader, loss_wrapper,
                                   eval_iters, gas, device, cfg, eval_train=eval_train)
        _record_losses(report.scalars, ema_losses, suffix="")

    # --- 2b. EMA audio + WER on fixed refs ---
    # estimate_loss leaves the model in train() mode → force eval() so generation runs with
    # dropout OFF (the predictors use dropout up to 0.5; sampling under it is wrong).
    model.eval()
    do_wer = "wer" in cfg.setup.eval_metrics
    do_sim_o = "sim_o" in cfg.setup.eval_metrics
    num_table_rows = cfg.setup.num_table_rows
    refs_by_source = {"train": train_refs, "val": val_refs}
    for table_name in cfg.setup.audio_tables:
        if table_name == "random_val":
            target_text, clips = generate_random_val_clips(
                model, val_datasets, sampling_rate, EVAL_TEXT_PROMPTS)
            report.audio_tables[RANDOM_VAL_TITLE] = {
                "columns": RANDOM_VAL_COLUMNS,
                "rows": [[snapshot_step, p_len, target_text,
                          AudioClip(prompt_np, sampling_rate), AudioClip(gen_np, sampling_rate)]
                         for p_len, prompt_np, gen_np in clips],
            }
            continue
        if table_name not in FIXED_REF_TABLES:
            continue  # overfit_batch is in-process-only (needs the cached overfit batch)
        title, split, source = FIXED_REF_TABLES[table_name]
        refs = refs_by_source[source]
        columns = ["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt",
                   "Original Audio", "Speech-Prompt", "Generated Audio"]
        if do_wer:
            columns += ["Transcription", "WER"]
        if do_sim_o:
            columns += ["SIM-o"]
        # WER/SIM-o are means over ALL refs (well-sampled metrics); only the first num_table_rows
        # are rendered as wandb rows (keeps the audio table small as num_audio_refs scales up).
        # Generate (length-sorted, frame-budget-packed) for as many refs as needed: ALL when a
        # metric is on, else just the rendered rows.
        k = len(refs) if (do_wer or do_sim_o) else min(num_table_rows, len(refs))
        proxies = [len(refs[i].original_np) // ENCODER_HOP_LENGTH for i in range(k)]
        gens = batch_generate(model, [refs[i].prompt_tensor for i in range(k)],
                              [refs[i].text for i in range(k)], proxies, cfg.setup.gen_frame_budget)
        wer_list = (compute_wer_batch(gens, [refs[i].text for i in range(k)], src_sr=sampling_rate,
                                      batch_samples=cfg.setup.metric_batch_samples) if do_wer else None)
        rows, synth_wers, gt_wers, sim_os = [], [], [], []
        for i in range(k):
            ref, gen = refs[i], gens[i]
            if do_wer:
                synth_wer, hyp = wer_list[i]
                synth_wers.append(synth_wer)
                gt_wers.append(ref.gt_wer)
            if do_sim_o:
                sim_os.append(compute_sim_o(gen, ref.prompt_embedding, src_sr=sampling_rate))
            if i < num_table_rows:
                row = [snapshot_step, prompt_seconds, ref.text,
                       AudioClip(ref.original_np, sampling_rate),
                       AudioClip(ref.prompt_np, sampling_rate),
                       AudioClip(gen, sampling_rate)]
                if do_wer:
                    row += [hyp, synth_wer]
                if do_sim_o:
                    row += [sim_os[-1]]
                rows.append(row)
        if do_wer:
            record_metric_dist(report.scalars, f"Evaluation: Metrics/{split}-WER", synth_wers)
            report.scalars[f"Evaluation: Metrics/{split}-WER-gt"] = _mean_skip_nan(gt_wers)
        if do_sim_o:
            record_metric_dist(report.scalars, f"Evaluation: Metrics/{split}-SIM-o", sim_os)
        report.audio_tables[title] = {"columns": columns, "rows": rows}

    # --- 3. best-ckpt decision (EMA val loss) ---
    if (cfg.setup.best_safetensors and ema_losses is not None and 'val' in ema_losses
            and ema_losses['val']['total_loss'] < prev_best_val_loss):
        report.new_best = True
        report.best_val_loss = ema_losses['val']['total_loss']

    return report
