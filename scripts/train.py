import os
import sys
import json
import time
import math
import queue
import signal
import ctypes
import random
import logging
import threading
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

import torch
import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.inference import compute_inference_data_loss, generate_audio
from naturalspeech2.eval import resolve_metric_device, set_metric_device, ipc
from naturalspeech2.eval.metrics import compute_sim_o, compute_wer_batch
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
from naturalspeech2.eval.runner import (
    estimate_loss,
    get_loss_section,
    _mean_skip_nan,
    build_fixed_refs_data,
    batch_generate,
    generate_random_val_clips,
    atomic_save_safetensors,
    EVAL_TEXT_PROMPTS,
    RANDOM_VAL_TITLE,
    RANDOM_VAL_COLUMNS,
)
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper, GradientAnalyzer
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.paths import CHECKPOINTS_DIR, PROJECT_ROOT
from naturalspeech2.utils.ema import EMA
from naturalspeech2.utils.utils import (
    setup_file_logger, compute_denominators,
    install_prewandb_log_buffer, flush_prewandb_log_buffer,
)
from naturalspeech2.utils.compile_tracking import compile_kwargs, read_compile_stats, format_break_reasons

logger = logging.getLogger(__name__)

# Compile telemetry: unique_graphs must PLATEAU after warmup (the leak signal). Logged as a
# per-log_interval wandb series + an eval-aware leak check (see the log block in the loop).
COMPILE_WARMUP_STEP = 250  # by here all buckets + their dynamic-shape promotions have compiled

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
            # CPU batches → lookahead queue
            yield batch, epoch, batch_idx

        # Epoch finished
        epoch += 1
        sampler.set_epoch(epoch)
        sampler.set_start_batch_idx(0)


def save_resume_checkpoint(unoptimized_model, optimizer, ema, *, model_cfg_dict,
                           token_vocabulary_size, sampling_rate, iter_num, best_val_loss,
                           epoch, batch_idx, cur_kimg, wandb_log):
    """Atomic full-state checkpoint (model + optimizer + EMA + data position) for crash-recovery
    and resume — the single source of truth for the ckpt.pt format. Written by both the periodic
    in-loop save AND the end-of-run final save (callers pass the already-incremented `batch_idx`,
    i.e. the index of the upcoming batch)."""
    checkpoint_data = {
        'model': unoptimized_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_cfg': model_cfg_dict,
        'token_vocabulary_size': token_vocabulary_size,
        'sampling_rate': sampling_rate,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'wandb_id': wandb.run.id if wandb_log else None,
        'epoch': epoch,
        'batch_idx': batch_idx,
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
    return ckpt_path

@dataclass
class EvalDeps:
    """Dependencies for run_eval_block + audio-table renderers. Built once per eval firing;
    heavy objects (model, dataset, refs) passed by reference."""
    iter_num: int
    unoptimized_model: nn.Module       # for generate_audio + compute_inference_data_loss
    compiled_model: nn.Module          # for estimate_loss
    sampling_rate: int
    device: str
    cfg: DictConfig
    val_datasets: list                 # [dev_dataset, test_dataset] — pooled source for random_val
    val_refs: list                     # pooled dev+test held-out refs (fixed_val_refs)
    train_refs: list                   # trained-on refs from the train subset (fixed_train_refs)
    custom_prompts: list
    overfit_ref_batch: Optional[dict]  # cached at iter 0 when overfit_batch is in audio_tables
    metrics_out: dict = field(default_factory=dict)  # per-eval scalar metrics (WER, …) → merged into eval_payload


def render_random_val_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Pick one random ≥10s val (dev+test) clip, generate audio at 5s + 10s prompt lengths."""
    sr = deps.sampling_rate
    target_text, clips = generate_random_val_clips(
        deps.unoptimized_model, deps.val_datasets, sr, deps.custom_prompts)
    rows = [
        [deps.iter_num, p_len, target_text,
         wandb.Audio(prompt_np, sample_rate=sr), wandb.Audio(gen_np, sample_rate=sr)]
        for p_len, prompt_np, gen_np in clips
    ]
    return RANDOM_VAL_TITLE, RANDOM_VAL_COLUMNS, rows


def build_fixed_refs(datasets, n_refs, prompt_samples_len, sampling_rate, rng, do_wer, do_sim_o):
    """Trainer-side fixed refs: shared data builder (deterministic selection over the pooled
    `datasets` + GT-floor WER and, when SIM-o is on, the prompt speaker embedding cached once)
    wrapped with wandb.Audio for the original/prompt clips (reused across evals). Call only when
    wandb is active."""
    return [
        {
            "original_audio": wandb.Audio(r.original_np, sample_rate=sampling_rate),
            "prompt_audio": wandb.Audio(r.prompt_np, sample_rate=sampling_rate),
            "prompt_tensor": r.prompt_tensor,
            "prompt_embedding": r.prompt_embedding,   # cached WavLM-SV embedding (None if SIM-o off)
            "original_np": r.original_np,
            "text": r.text,
            "gt_wer": r.gt_wer,   # GT-floor WER, computed once at build
        }
        for r in build_fixed_refs_data(
            datasets, n_refs, prompt_samples_len, rng,
            sampling_rate=sampling_rate, compute_gt_wer=do_wer, compute_sim_emb=do_sim_o,
        )
    ]


def _render_fixed_refs_table(deps: EvalDeps, refs: list, title: str, split: str) -> tuple[str, list, list]:
    """Generate audio on a fixed reference set (shared by dev + test tables).

    If "wer" in cfg.setup.eval_metrics: per-clip synth WER column + per-split means (synth WER and
    the GT-floor cached on each ref) → gap (synth − floor) = honest signal.
    If "sim_o" in cfg.setup.eval_metrics: per-clip speaker-similarity column + per-split mean.
    """
    sr = deps.sampling_rate
    do_wer = "wer" in deps.cfg.setup.eval_metrics
    do_sim_o = "sim_o" in deps.cfg.setup.eval_metrics
    num_table_rows = deps.cfg.setup.num_table_rows
    columns = ["Iteration", "Speech-Prompt-Length (s)", "Text-Prompt",
               "Original Audio", "Speech-Prompt", "Generated Audio"]
    if do_wer:
        columns += ["Transcription", "WER"]
    if do_sim_o:
        columns += ["SIM-o"]

    # WER/SIM-o are means over ALL refs (well-sampled metrics); only the first num_table_rows
    # are rendered as wandb rows (keeps the audio table small as num_audio_refs scales up).
    # Generate for ALL refs when a metric is on, else just the rendered rows.
    rows, synth_wers, gt_wers, sim_os = [], [], [], []
    k = len(refs) if (do_wer or do_sim_o) else min(num_table_rows, len(refs))
    proxies = [len(refs[i]["original_np"]) // ENCODER_HOP_LENGTH for i in range(k)]
    gens = batch_generate(deps.unoptimized_model, [refs[i]["prompt_tensor"] for i in range(k)],
                          [refs[i]["text"] for i in range(k)], proxies, deps.cfg.setup.gen_frame_budget)
    wer_list = (compute_wer_batch(gens, [refs[i]["text"] for i in range(k)], src_sr=sr,
                                  batch_samples=deps.cfg.setup.metric_batch_samples) if do_wer else None)
    for i in range(k):
        ref, gen = refs[i], gens[i]
        if do_wer:
            synth_wer, hyp = wer_list[i]
            synth_wers.append(synth_wer)
            gt_wers.append(ref["gt_wer"])
        if do_sim_o:
            sim_os.append(compute_sim_o(gen, ref["prompt_embedding"], src_sr=sr))
        if i < num_table_rows:
            row = [
                deps.iter_num,
                deps.cfg.model.prompt_seconds,
                ref["text"],
                ref["original_audio"],
                ref["prompt_audio"],
                wandb.Audio(gen, sample_rate=sr),
            ]
            if do_wer:
                row += [hyp, synth_wer]
            if do_sim_o:
                row += [sim_os[-1]]
            rows.append(row)

    if do_wer:
        deps.metrics_out[f"Evaluation: Metrics/{split}-WER"] = _mean_skip_nan(synth_wers)
        deps.metrics_out[f"Evaluation: Metrics/{split}-WER-gt"] = _mean_skip_nan(gt_wers)
    if do_sim_o:
        deps.metrics_out[f"Evaluation: Metrics/{split}-SIM-o"] = _mean_skip_nan(sim_os)

    return (title, columns, rows)


def render_fixed_train_refs_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the fixed trained-on references (sampled from the train subset)."""
    return _render_fixed_refs_table(deps, deps.train_refs, "Eval Audio: train (trained-on)", "train")


def render_fixed_val_refs_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the combined dev+test held-out VALIDATION references.

    dev+test together = validation set (speaker-disjoint from train and each other; paper reports
    only on external VCTK / LibriSpeech), so one pooled table + one combined val-WER under 'val'
    (matches the chained 'val' loss in estimate_loss). Refs arrive pre-pooled in deps.val_refs."""
    return _render_fixed_refs_table(
        deps, deps.val_refs,
        "Eval Audio: dev+test (held-out validation)", "val",
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

        # on_overflow="warn": training-eval generation must survive an early under-trained predictor.
        gen_audio_np, length = generate_audio(deps.unoptimized_model, ref_audio_slice,
                                              target_text=batch["text"][i], on_overflow="warn")

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
    "random_val": render_random_val_table,
    "fixed_train_refs": render_fixed_train_refs_table,
    "fixed_val_refs": render_fixed_val_refs_table,
}


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


def compute_suggested_loss_weights(
    magnitudes: dict[str, float],
    targets: dict,
    anchor: str = "data_loss",
) -> dict:
    """Suggest loss_weights so each leaf's weighted contribution matches a target share,
    given the raw per-leaf magnitudes a loss-analysis run measured.

    targets mirrors loss_weights: flat leaves carry a scalar; groups carry group_target + per-sub
    targets. Desired share:
        flat:    c = target
        grouped: c = group_target * sub_target / sum(sub_targets in group)
    Effective weight W = c / magnitude. All weights scaled so the anchor leaf's W = 1.0, pinning
    the anchor's gradient scale (data_loss → diffusion path) so the paper LR transfers.

    Two-level like LossWrapper: sub_weight = sub_target / magnitude; group_weight = group share ×
    anchor scale. Returns a nested dict shaped like loss_weights, ready to paste into config/model.
    """
    suggested: dict = {}
    eff_unscaled: dict[str, float] = {}   # leaf -> pre-anchor effective weight

    for key, value in targets.items():
        if isinstance(value, dict):
            sub_targets = {k: v for k, v in value.items() if k != "group_target"}
            group_share = value["group_target"] / sum(sub_targets.values())
            suggested[key] = {"group_weight": group_share}   # anchor scale folded in below
            for sub_key, sub_target in sub_targets.items():
                sub_weight = sub_target / magnitudes[sub_key]
                suggested[key][sub_key] = sub_weight
                eff_unscaled[sub_key] = group_share * sub_weight
        else:
            weight = value / magnitudes[key]
            suggested[key] = weight
            eff_unscaled[key] = weight

    assert anchor in eff_unscaled, (
        f"anchor '{anchor}' is not a loss leaf; have {sorted(eff_unscaled)}"
    )
    alpha = 1.0 / eff_unscaled[anchor]

    for key, value in suggested.items():
        if isinstance(value, dict):
            value["group_weight"] *= alpha
        else:
            suggested[key] = value * alpha

    return suggested


def _should_run_eval(iter_num: int, cfg) -> bool:
    """Eval trigger: every eval_interval steps (never at step 0)."""
    return iter_num > 0 and iter_num % cfg.setup.eval_interval == 0


def _append_aligner_trial_eval(out_path, step: int, losses: dict) -> None:
    """Append one held-out eval point as a JSON line for the aligner Optuna driver: step +
    per-term (+total) held-out loss (the driver minimizes forward_sum+bin over the tail).
    dev+test are chained into one pooled 'val' record."""
    record = {"step": step}
    if "val" in losses:
        record["val_total"] = losses["val"]["total_loss"]
        record["val"] = losses["val"]["logged_losses"]
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ----------------------------------------------------------------------------
# Decoupled eval daemon — trainer side (spawn/supervise, snapshot writer, results drain)
# ----------------------------------------------------------------------------

EVAL_DAEMON_SCRIPT = PROJECT_ROOT / "scripts" / "eval_daemon.py"
EVAL_DAEMON_SHUTDOWN_TIMEOUT_S = 1200   # allow a slow final dev+test pass + audio at run end
EVAL_DAEMON_MAX_RESPAWNS = 5            # persistent crash → stop respawning (training continues, eval paused)


def _clone_trainable_cpu(model, ema):
    """Snapshot live trainable params + EMA shadow as fresh CPU tensors. Training-thread cost is a
    GPU→CPU copy of 2× trainable weights (tens–hundreds of ms once per eval cadence); the slow
    serialize/write runs on the writer thread."""
    live = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}
    shadow = {n: t.detach().to("cpu", copy=True) for n, t in ema.shadow.items()}
    return live, shadow


class SnapshotWriter:
    """Background thread: serialize weight snapshots to /dev/shm off the training thread. Main
    thread does the cheap clone + submit; this thread does the torch.save + marker bump."""
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="snapshot-writer", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            step, live, shadow = item
            ipc.write_snapshot(self.run_dir, step, live, shadow)

    def submit(self, step, live, shadow):
        self._q.put((step, live, shadow))

    def close(self):
        """Flush + join — at shutdown, so the final snapshot lands on disk before we signal."""
        self._q.put(None)
        self._thread.join()


def _pdeathsig_preexec():
    """Child hook (Linux): PR_SET_PDEATHSIG=1 → SIGTERM when the trainer dies, so a trainer
    crash/kill never leaves an orphan daemon holding GPU1. Best-effort: a failure here must not
    abort the spawn (the daemon still works without it; Ctrl-C is covered by the process group)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:
        pass


def _spawn_eval_daemon(run_dir, log_dir):
    """Launch the eval daemon as a fresh subprocess pinned to GPU1 (clean CUDA context, no fork).
    log_dir = the run's log dir (scratch vs main) → daemon writes eval_daemon.log alongside the run log."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(EVAL_DAEMON_SCRIPT), "--run-dir", str(run_dir), "--log-dir", str(log_dir)],
        env=env, cwd=str(PROJECT_ROOT), preexec_fn=_pdeathsig_preexec,
    )
    logger.info(f"Spawned eval daemon (pid {proc.pid}) on GPU1; run_dir={run_dir}")
    return proc


def _drain_and_log(run_dir, incremental_audio_tables, wandb_log):
    """Drain finished daemon evals → wandb. Always drains+cleans even when wandb is off (bounds
    /dev/shm). Audio cells are wav paths wrapped as wandb.Audio, which reads the file at construction
    → the wavs are deleted only AFTER that (in the finally, via cleanup_eval_dir). Tables stay
    INCREMENTAL across the run. No step= → eval charts use the eval/snapshot_step data field (set via
    define_metric), immune to the trainer being many steps ahead of a stale snapshot. A wandb hiccup
    here logs + continues — forwarding eval results must never crash training (the daemon's
    crash-isolation guarantee; wandb is an external boundary)."""
    for rec in ipc.drain_results(run_dir):
        try:
            if wandb_log:
                payload = dict(rec["scalars"])
                for title, table in rec["audio_tables"].items():
                    wandb_table = incremental_audio_tables.get(title)
                    if wandb_table is None:
                        wandb_table = wandb.Table(columns=table["columns"], log_mode="INCREMENTAL")
                        incremental_audio_tables[title] = wandb_table
                    for row in table["rows"]:
                        cells = [wandb.Audio(c["__audio__"]) if isinstance(c, dict) and "__audio__" in c else c
                                 for c in row]
                        wandb_table.add_data(*cells)
                    payload[title] = wandb_table
                payload["eval/snapshot_step"] = rec["step"]
                wandb.log(payload)
            logger.info(f"Drained + logged daemon eval for snapshot_step {rec['step']}.")
        except Exception:
            logger.exception(f"Failed to log drained eval (step {rec.get('step')}); skipping.")
        finally:
            ipc.cleanup_eval_dir(rec["_eval_dir"])


def run_eval_block(
    deps: EvalDeps,
    train_loader,
    dev_loader,
    test_loader,
    loss_wrapper: LossWrapper,
    ema: Optional[EMA],
    best_val_loss: float,
    incremental_audio_tables: dict,
) -> float:
    """Single eval pass under EMA-swapped weights → possibly-updated best_val_loss.

    eval() at entry, train() in a finally on exit. Caller must NOT wrap this in ema.swap_in —
    managed internally (EMA.swap_in is non-reentrant).
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
                    eval_train=not deps.cfg.setup.aligner_trial_run,
                )
                logged = ", ".join(f"{s}={losses[s]['total_loss']:.4f}"
                                   for s in ('train', 'val') if s in losses)
                logger.info(f"Step {deps.iter_num}: eval losses  {logged}")
                for split_name, section in [('train', 'Train'), ('val', 'Val')]:
                    if split_name not in losses:
                        continue
                    eval_payload[f"Evaluation: Metrics/{section}-Loss"] = losses[split_name]['total_loss']
                    for k, v in losses[split_name]['logged_losses'].items():
                        eval_payload[f"{get_loss_section(k, 'Evaluation', section)}/{k}"] = v

            if deps.cfg.setup.inference_data_loss_sweep and deps.overfit_ref_batch is not None:
                # Training data_loss = one-step error from a known noisy z_t; inference chains N ODE
                # steps from t=1 to t≈0. Sweep step counts to separate solver-discretization error
                # (drops to ~training data_loss) from a weights-side gap (flat across step counts).
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
                # estimate_loss (above) leaves the shared module in train() mode → force eval()
                # so generation runs with dropout OFF (predictors use dropout up to 0.5).
                deps.unoptimized_model.eval()
                for table_name in deps.cfg.setup.audio_tables:
                    wandb_key, columns, rows = AUDIO_TABLES[table_name](deps)
                    # INCREMENTAL → rows accumulate into one table (compare audio across iters);
                    # a fresh table per eval would show only the latest step.
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
                and 'val' in losses
                and losses['val']['total_loss'] < best_val_loss
            ):
                best_val_loss = losses['val']['total_loss']
                logger.info(f"Saving new best model to {CHECKPOINTS_DIR}")
                atomic_save_safetensors(deps.unoptimized_model, CHECKPOINTS_DIR / 'ema_best.safetensors')
    finally:
        deps.unoptimized_model.train()

    if deps.cfg.setup.aligner_trial_run and losses is not None:
        _append_aligner_trial_eval(PROJECT_ROOT / deps.cfg.setup.aligner_trial_out, deps.iter_num, losses)

    eval_payload.update(deps.metrics_out)

    if deps.cfg.wandb.log and eval_payload:
        wandb.log(eval_payload, step=deps.iter_num)

    logger.info(f"Eval block done in {time.perf_counter() - eval_start_time:.1f}s")
    return best_val_loss


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):

    # Route warnings.warn (torch TF32/Inductor notices, etc.) through logging → they land in the log
    # files too, matching console + wandb (Python's "once per location" filter still dedups them).
    logging.captureWarnings(True)

    # Buffer every log emitted before wandb.init() (dataloaders, dataset preprocessing, model/param/
    # compile/daemon lines) so they can be replayed into the wandb Logs tab — wandb only hooks stdout
    # at init, so without this they'd never reach wandb. Replayed (wandb on) or dropped (off) below.
    prewandb_log_buffer = install_prewandb_log_buffer()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires an NVIDIA GPU and CUDA installed, but none were detected.")

    device = cfg.setup.device
    device_type = 'cuda'

    # Seed torch + Python random → reproducible model init, diffusion t/ε, prompt windows,
    # eval prompt picks, DataLoader worker RNG. cuDNN determinism NOT enforced (~5–10% perf
    # cost) — fine for A/B. Bucketed sampler has its own hardcoded np.random seed, so the 5
    # overfit batches stay fixed regardless of cfg.seed.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Warmup is canonically a fraction of the run (cfg.setup.warmup_ratio); resolve to an absolute
    # step count unless a config pins warmup_iters explicitly (e.g. overfit_test=0). ISR's
    # post-warmup LR = peak·√(warmup/it) is run-length-independent only for a fixed warmup.
    if cfg.setup.warmup_iters is None:
        cfg.setup.warmup_iters = round(cfg.setup.warmup_ratio * cfg.setup.max_iters)

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
        # Catches model/base.yaml adding a dropout key that overfit_test.yaml forgot to override.
        nonzero_dropouts = {k: v for k, v in _collect_dropout_keys(cfg.model).items() if v != 0.0}
        assert not nonzero_dropouts, (
            f"overfit_single_batch=True but these dropout keys are non-zero in cfg.model: "
            f"{nonzero_dropouts}. Add them to model/overfit_test.yaml overrides."
        )

    log_dir = PROJECT_ROOT / cfg.setup.log_subdir
    log_name = f"{cfg.setup.log_name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Ensure save targets exist before any write. CHECKPOINTS_DIR may be absent on fresh
    # containers (models/ from encodec setup, no checkpoints subdir yet).
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logger(logger, log_dir / log_name, root=True)

    # Daemon offloads eval to GPU1. overfit/gradient_analysis/aligner_trial need trainer-local eval
    # state so they stay in-process; loss_analysis's eval is plain held-out loss → daemon-compatible.
    daemon_incompatible_run = (
        cfg.setup.overfit_single_batch
        or cfg.setup.gradient_analysis_run or cfg.setup.aligner_trial_run
    )
    use_eval_daemon = (
        cfg.setup.eval_daemon.enabled
        and torch.cuda.device_count() > 1
        and cfg.model.ema.enabled
        and not daemon_incompatible_run
    )
    daemon_proc = None
    snapshot_writer = None
    eval_run_dir = None
    daemon_respawn_count = 0
    if use_eval_daemon:
        logger.info("Decoupled eval daemon ENABLED (eval runs on GPU1; training never pauses).")
    else:
        # In-process WER ASR honors metric_device (auto → CPU on single-GPU, never the train card).
        set_metric_device(resolve_metric_device(cfg.setup.metric_device))
        if cfg.setup.eval_daemon.enabled and not daemon_incompatible_run:
            logger.info(f"Eval daemon requested but inactive (device_count="
                        f"{torch.cuda.device_count()}, ema_enabled={cfg.model.ema.enabled}); "
                        "falling back to in-process eval.")

    logger.info("Initializing DataLoaders...")
    train_loader, train_dataset = create_dataloader(cfg, cfg.dataset.train_split, cfg.dataset.token_vocabulary_path)
    dev_loader, dev_dataset = create_dataloader(cfg, cfg.dataset.dev_split, train_dataset.token_vocabulary_path)
    test_loader, test_dataset = create_dataloader(cfg, cfg.dataset.test_split, train_dataset.token_vocabulary_path)

    # Shared tokenizer (espeak backend): feeds vocab_size to model construction AND attached
    # as model._inference_tokenizer so eval-block generate_audio() hides phoneme plumbing.
    inference_tokenizer = PhonemeTokenizer(
        token_vocabulary_path=train_dataset.token_vocabulary_path, with_backend=True,
    )
    token_vocabulary_size = inference_tokenizer.token_vocabulary_size

    # Eval block's static text prompts (random_val table); shared with the daemon via runner.
    custom_prompts = EVAL_TEXT_PROMPTS

    sampling_rate = cfg.dataloader.sampling_rate
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)

    # State initialization variables
    start_iter = 0
    best_val_loss = 1e9
    start_epoch = 0
    start_batch_idx = 0
    # Persistent INCREMENTAL wandb.Tables (keyed by name) → eval audio accumulates across firings.
    incremental_audio_tables: dict = {}

    # Instantiate Model
    if cfg.setup.init_from == 'scratch':
        logger.info("Initializing a new model from scratch...")
        # Fresh run → drop the daemon's persistent best-tracking from a prior run in this dir, so the
        # ema_best gate restarts from inf (matches the in-process best_val_loss=1e9 reset). Stale weight
        # files are left untouched (never read on scratch; overwritten as the run progresses).
        (CHECKPOINTS_DIR / "eval_state.json").unlink(missing_ok=True)
        model_cfg = model_cfg_from_omegaconf(cfg.model)
        model = NaturalSpeech2Model(
            model_cfg,
            token_vocabulary_size=token_vocabulary_size,
            sampling_rate=sampling_rate,
        )
    elif cfg.setup.init_from == 'resume':
        ckpt_path = CHECKPOINTS_DIR / 'ckpt.pt'
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Rebuild from the checkpoint's own cfg → architecture matches even if base.yaml drifted.
        ckpt_cfg = model_cfg_from_omegaconf(checkpoint['model_cfg'])
        model = NaturalSpeech2Model(
            ckpt_cfg,
            token_vocabulary_size=checkpoint['token_vocabulary_size'],
            sampling_rate=checkpoint['sampling_rate'],
        )

        state_dict = checkpoint['model']

        model.load_state_dict(state_dict)
        start_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
        start_epoch = checkpoint['epoch']

        # Save the cfg the model was built with, not the (possibly drifted) Hydra cfg.
        model_cfg_dict = checkpoint['model_cfg']
        start_batch_idx = checkpoint['batch_idx'] # Already points to the next batch due to pre-fetch

        logger.info(f"Resuming training at iteration {start_iter} from checkpoint in {CHECKPOINTS_DIR}...")

    model.to(device)

    # Attach inference helpers for eval-block generate_audio(). Plain attributes (not
    # Parameter/Buffer) → don't touch state_dict, torch.compile, or forward().
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

    unoptimized_model = model
    if cfg.setup.compile.enabled:
        kwargs = compile_kwargs(cfg.setup.compile)
        logger.info(f"Compiling the model... (this takes a minute) "
                    f"[dynamic={cfg.setup.compile.dynamic}, mode={cfg.setup.compile.mode}]")
        model = torch.compile(model, **kwargs)
    else:
        logger.info("torch.compile disabled (setup.compile.enabled=false) — running eager.")

    # Outside the wandb.log gate so the renderers iterate safely on wandb.log=False debug runs
    # (empty list → no rows). In daemon mode the daemon builds its own refs → trainer skips them
    # (avoids loading the WER ASR on the training card). Build only the ref sets audio_tables needs.
    val_refs = []
    train_refs = []
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

        # Replay the buffered pre-init logs into the now-hooked stdout → wandb Logs tab, in original
        # order, just ahead of the live stream (sys.stdout read here to catch wandb's wrapped stream).
        flush_prewandb_log_buffer(prewandb_log_buffer, sys.stdout)

        if use_eval_daemon:
            # Eval logged from drained daemon results against a custom x-axis (the true snapshot
            # step) → eval curves don't collapse when the trainer is many steps ahead of a stale eval.
            wandb.define_metric("eval/snapshot_step")
            wandb.define_metric("Evaluation: *", step_metric="eval/snapshot_step")

        if not use_eval_daemon:
            num_static_refs = cfg.setup.num_audio_refs
            prompt_samples_len = int(cfg.model.prompt_seconds * sampling_rate)
            # One seeded RNG over the pooled dev+test candidates → identical across runs (cross-run
            # A/B) and matching the daemon's draw. val_refs = held-out; train_refs = trained-on subset.
            do_wer = "wer" in cfg.setup.eval_metrics
            do_sim_o = "sim_o" in cfg.setup.eval_metrics
            if "fixed_val_refs" in cfg.setup.audio_tables:
                val_refs = build_fixed_refs([dev_dataset, test_dataset], num_static_refs, prompt_samples_len, sampling_rate, random.Random(cfg.seed), do_wer, do_sim_o)
            if "fixed_train_refs" in cfg.setup.audio_tables:
                train_refs = build_fixed_refs([train_dataset], num_static_refs, prompt_samples_len, sampling_rate, random.Random(cfg.seed), do_wer, do_sim_o)
    else:
        # wandb off → no Logs tab to replay into; drop the buffer so it stops accumulating.
        flush_prewandb_log_buffer(prewandb_log_buffer)

    # Spawn the eval daemon (handshake first → daemon reads it at startup). Pinned to GPU1.
    if use_eval_daemon:
        run_tag = wandb.run.id if cfg.wandb.log else f"pid{os.getpid()}"
        eval_run_dir = ipc.resolve_run_dir(cfg.setup.eval_daemon.snapshot_dir, run_tag)
        ipc.write_daemon_init(eval_run_dir, {
            "model_cfg": model_cfg_dict,
            "token_vocabulary_size": token_vocabulary_size,
            "token_vocabulary_path": str(train_dataset.token_vocabulary_path),
            "sampling_rate": sampling_rate,
            "daemon_device": "cuda:0",   # under CUDA_VISIBLE_DEVICES=1 == physical GPU1
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        })
        snapshot_writer = SnapshotWriter(eval_run_dir)
        daemon_proc = _spawn_eval_daemon(eval_run_dir, log_dir)

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
        loss_analysis_start_iter = int(cfg.setup.max_iters * 0.8)   # last 20% of steps (stable logged values)
        loss_analysis_accumulators = {}
        loss_analysis_steps_counted = 0

    if cfg.setup.aligner_trial_run:
        # Truncate any stale dump from a prior run.
        aligner_trial_out_path = PROJECT_ROOT / cfg.setup.aligner_trial_out
        aligner_trial_out_path.parent.mkdir(parents=True, exist_ok=True)
        aligner_trial_out_path.write_text("")
        
    if cfg.setup.gradient_analysis_run:
        # Last 20% (stable logged values), but capped to ~200 analyzed steps (window ÷ log_interval)
        # so layering gradient_analysis onto a LONG productive run (e.g. a train_5M continuation)
        # doesn't balloon the 9×-cost per-term passes over thousands of steps. Identical to 0.8× for
        # the standalone 10k run (log_interval=10 → 200·10=2000 = the last 20%).
        grad_analysis_start_iter = max(int(cfg.setup.max_iters * 0.8), cfg.setup.max_iters - 200 * cfg.setup.log_interval)
        grad_norm_accumulators = {}
        cos_sim_accumulators = {}
        grad_analysis_steps_counted = 0
        
    logger.info("Starting training loop...")
    last_log_time = time.perf_counter()
    last_log_iter = start_iter - 1
    prev_unique_graphs = None           # eval-aware recompile-leak check (None until warmup baseline)
    prev_break_reasons = set()          # text-log break reasons only when the SET changes
    eval_ran_since_graph_check = False  # in-process eval compiles eval-mode graphs once → not a leak

    # Filled on the first loop iter when overfit_batch table is active. Overfit cycling yields
    # the same 5 objects forever, so caching lookahead_queue[0] once gives a stable ref batch.
    overfit_ref_batch = None

    # Seed the loop-locals so the end-of-run final checkpoint is still coherent if the loop body
    # never executes (a resume with start_iter >= max_iters, e.g. re-resuming a completed run
    # without raising max_iters) instead of reading unbound names.
    current_epoch, current_batch_idx = start_epoch, start_batch_idx

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

        # Cache overfit batch once — cycling guarantees lookahead_queue[0] is the same object each iter.
        if overfit_ref_batch is None and "overfit_batch" in cfg.setup.audio_tables:
            overfit_ref_batch = lookahead_queue[0]

        # -----------------------------
        # Evaluation
        # -----------------------------
        if _should_run_eval(iter_num, cfg):
            if use_eval_daemon:
                # Decoupled: clone weights (cheap, on this thread) + hand to the writer thread.
                # Training continues; the daemon evals on GPU1 and results drain at log cadence
                # below. The small clone cost is left in the step-time metric (so it stays visible).
                live_cpu, shadow_cpu = _clone_trainable_cpu(unoptimized_model, ema)
                snapshot_writer.submit(iter_num, live_cpu, shadow_cpu)
                logger.info(f"Submitted eval snapshot (iter {iter_num}) to the GPU1 daemon; "
                            f"result surfaces at the next log_interval after it finishes.")
            else:
                eval_deps = EvalDeps(
                    iter_num=iter_num,
                    unoptimized_model=unoptimized_model,
                    compiled_model=model,
                    sampling_rate=sampling_rate,
                    device=device,
                    cfg=cfg,
                    val_datasets=[dev_dataset, test_dataset],
                    val_refs=val_refs,
                    train_refs=train_refs,
                    custom_prompts=custom_prompts,
                    overfit_ref_batch=overfit_ref_batch,
                )
                eval_start_time = time.perf_counter()
                best_val_loss = run_eval_block(
                    eval_deps,
                    train_loader, dev_loader, test_loader,
                    loss_wrapper, ema, best_val_loss,
                    incremental_audio_tables,
                )
                last_log_time += time.perf_counter() - eval_start_time
                eval_ran_since_graph_check = True  # a unique_graphs bump at the next check is eval-mode, not a leak

        # -----------------------------
        # Forward & Backward Pass
        # -----------------------------
        
        accum_loss = torch.zeros((), device=device)
        accum_logged_losses = {}
        
        analyzer = GradientAnalyzer() if (cfg.setup.gradient_analysis_run and iter_num >= grad_analysis_start_iter and iter_num % cfg.setup.log_interval == 0) else None
        
        if analyzer:
            logger.info(f"Performing gradient analysis for step {iter_num} (this takes extra time)...")
                
            # Capture RNG states for fair per-term analysis passes
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state(device)
                
            # 2. Independent passes for each loss term (keys discovered dynamically)
            optimizer.zero_grad(set_to_none=True)
            loss_keys = []
            loss_idx = 0
            
            while True:
                # Restore RNG → identical forward per loss term
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
                    
                    # Free graphs immediately → avoid VRAM high-water spikes
                    del b_gpu, loss_dict, _loss, _logged_losses, weighted_tensors, loss_term
                
                # .grad → CPU
                analyzer.extract_gradients(unoptimized_model, loss_keys[loss_idx])
                optimizer.zero_grad(set_to_none=True)
                
                loss_idx += 1
                if loss_idx >= len(loss_keys):
                    break
                
            # 3. Standard update pass over the same data; restore RNG so the real step aligns with analysis
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
                
                # accumulate detached scalars for logging
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

            # --- torch.compile telemetry (CPU-only counter reads → free; Dynamo maintains them
            # regardless). unique_graphs must PLATEAU after warmup; a grow at a NON-eval step = shape
            # leak. The first in-process eval compiles a one-time eval-mode graph family (benign).
            cstats = None
            if cfg.setup.compile.enabled:
                cstats = read_compile_stats()
                rel_step = iter_num - start_iter
                # Verbose status: first log (compile-happened) + first post-warmup log (all buckets in).
                if rel_step <= 1 or (prev_unique_graphs is None and rel_step >= COMPILE_WARMUP_STEP):
                    logger.info(f"torch.compile: {cstats['unique_graphs']} graphs | {cstats['graph_breaks_total']} "
                                f"breaks [{format_break_reasons(cstats['break_reasons'])}] | "
                                f"cache_size_limit={cstats['cache_size_limit']}")
                # Intended-breaks tripwire: log only when the reason SET changes (no per-step strings).
                reason_set = set(cstats["break_reasons"])
                if reason_set != prev_break_reasons:
                    added = reason_set - prev_break_reasons
                    if added and prev_break_reasons:
                        logger.warning(f"⚠️ New torch.compile graph-break reason(s): {sorted(added)} — "
                                       f"expected only the 2 intended @torch.compiler.disable sites.")
                    prev_break_reasons = reason_set
                # Eval-aware leak check: a unique_graphs grow at a non-eval step past warmup = real leak.
                ug = cstats["unique_graphs"]
                if rel_step >= COMPILE_WARMUP_STEP:
                    if prev_unique_graphs is not None and ug > prev_unique_graphs:
                        if eval_ran_since_graph_check:
                            logger.info(f"unique_graphs {prev_unique_graphs}→{ug} after an eval — one-time "
                                        f"eval-mode graph compilation, not a leak.")
                        else:
                            logger.warning(f"⚠️ unique_graphs grew {prev_unique_graphs}→{ug} at a non-eval step after "
                                           f"warmup — batch shapes are LEAKING (recompiling beyond the "
                                           f"{len(cfg.dataloader.bucket_mapping)} buckets). Re-run with "
                                           f"TORCH_LOGS=recompiles to see the failing guard.")
                    prev_unique_graphs = ug
                eval_ran_since_graph_check = False

            # Gradient-analysis metrics: compute ONCE + accumulate here, OUTSIDE the wandb block.
            # This mode REQUIRES wandb.log=false (resume re-opens the train_5M run, already at its
            # last step → live logs would be dropped as non-monotonic), so the end-of-run summary
            # must NOT depend on the wandb path. `analyzer is not None` ⟹ already in the window.
            grad_norms = cos_sims = None
            if analyzer is not None:
                grad_norms, cos_sims = analyzer.compute_metrics()
                grad_analysis_steps_counted += 1
                for k, v in grad_norms.items():
                    grad_norm_accumulators[k] = grad_norm_accumulators.get(k, 0.0) + v
                for k, v in cos_sims.items():
                    cos_sim_accumulators[k] = cos_sim_accumulators.get(k, 0.0) + v

            if cfg.wandb.log:
                log_payload = {
                    "Train: Metrics/Loss": lossf,
                    "Train: Metrics/Avg. Time per Step (ms)": dt_avg * 1000,
                    "Train: Metrics/Learning Rate": lr,
                    "Train: Metrics/Gradient Norm": grad_norm.item(),
                }
                if cstats is not None:
                    log_payload["Compile/unique_graphs"] = cstats["unique_graphs"]
                    log_payload["Compile/graph_breaks_total"] = cstats["graph_breaks_total"]
                    log_payload["Compile/n_break_reasons"] = cstats["n_break_reasons"]
                    log_payload["Compile/cache_size_limit"] = cstats["cache_size_limit"]
                # Balanced loss components; .item() inside log_interval → no per-step GPU sync.
                for k, v in accum_logged_losses.items():
                    log_payload[f"{get_loss_section(k, 'Train')}/{k}"] = v.item()

                # Each raw term's share of summed raw magnitudes — flat once balance stabilizes
                # (drift signal in the main run).
                raw_terms = {k: v for k, v in accum_logged_losses.items() if not k.endswith("_weighted")}
                raw_total = sum(raw_terms.values())
                for k, v in raw_terms.items():
                    log_payload[f"Train: Loss Composition (Raw Fraction)/{k}"] = (v / raw_total).item()

                # Gradient analysis (computed + accumulated above, wandb-independent); log it.
                if analyzer is not None:
                    for k, v in grad_norms.items():
                        if k.endswith("_total"):
                            log_payload[f"Gradient Analysis: L2-Norms (Total)/{k}"] = v
                        else:
                            log_payload[f"Gradient Analysis: L2-Norms (Shared)/{k}"] = v
                    for k, v in cos_sims.items():
                        log_payload[f"Gradient Analysis: Cosine-Similarity (Shared)/{k}"] = v

                if ema is not None:
                    log_payload["EMA/effective_decay"] = ema._effective_decay(
                        batch_size=examples_this_step, cur_kimg=cur_kimg,
                    )
                    log_payload["EMA/effective_halflife_kimg"] = min(cur_kimg, ema.halflife_kimg)
                    log_payload["EMA/cur_kimg"] = cur_kimg
                    # RMS drift between live and shadow weights — grows, then plateaus as the
                    # shadow centers on the SGD oscillation around the loss minimum.
                    with torch.no_grad():
                        drift_sq = torch.zeros((), device=device)
                        n_drift = 0
                        for name, p in unoptimized_model.named_parameters():
                            if name in ema.shadow:
                                drift_sq += (p.data.float() - ema.shadow[name]).pow(2).sum()
                                n_drift += p.numel()
                    log_payload["EMA/weight_drift_rms"] = (drift_sq / max(n_drift, 1)).sqrt().item()

                wandb.log(log_payload, step=iter_num)

            # Accumulate raw unweighted losses for the analysis table
            if cfg.setup.loss_analysis_run and iter_num >= loss_analysis_start_iter:
                loss_analysis_steps_counted += 1
                for k, v in accum_logged_losses.items():
                    if not k.endswith("_weighted"):
                        loss_analysis_accumulators[k] = loss_analysis_accumulators.get(k, 0.0) + v.item()

            # Drain finished daemon evals → wandb; respawn the daemon if it crashed (capped,
            # non-blocking). Training never waits on the daemon.
            if use_eval_daemon:
                if daemon_proc.poll() is not None:
                    daemon_respawn_count += 1
                    if daemon_respawn_count <= EVAL_DAEMON_MAX_RESPAWNS:
                        logger.warning(f"Eval daemon exited (code {daemon_proc.returncode}); "
                                       f"respawning ({daemon_respawn_count}/{EVAL_DAEMON_MAX_RESPAWNS}).")
                        daemon_proc = _spawn_eval_daemon(eval_run_dir, log_dir)
                    elif daemon_respawn_count == EVAL_DAEMON_MAX_RESPAWNS + 1:
                        logger.error("Eval daemon exceeded max respawns; leaving it down "
                                     "(training continues, eval paused).")
                _drain_and_log(eval_run_dir, incremental_audio_tables, cfg.wandb.log)

        # -----------------------------
        # Periodic crash-recovery checkpoint (.pt) — decoupled from log_interval
        # -----------------------------
        if (
            cfg.setup.checkpoint_interval > 0
            and iter_num > 0
            and iter_num % cfg.setup.checkpoint_interval == 0
        ):
            save_resume_checkpoint(
                unoptimized_model, optimizer, ema,
                model_cfg_dict=model_cfg_dict, token_vocabulary_size=token_vocabulary_size,
                sampling_rate=sampling_rate, iter_num=iter_num, best_val_loss=best_val_loss,
                epoch=current_epoch, batch_idx=current_batch_idx + 1,   # upcoming batch
                cur_kimg=cur_kimg, wandb_log=cfg.wandb.log,
            )

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

        # Suggested loss_weights to hit configured target shares (loss_balance_targets), anchored data_loss=1.0.
        targets = OmegaConf.to_container(cfg.model.loss_balance_targets, resolve=True)
        magnitudes = {k: v / loss_analysis_steps_counted for k, v in loss_analysis_accumulators.items()}
        suggested = compute_suggested_loss_weights(magnitudes, targets, anchor="data_loss")
        logger.info("Suggested loss_weights (anchor: data_loss = 1.0) — paste into config/model:")
        for key, val in suggested.items():
            if isinstance(val, dict):
                logger.info(f"  {key}:")
                logger.info(f"    group_weight: {val['group_weight']:.4f}")
                for sub_key, sub_val in val.items():
                    if sub_key != "group_weight":
                        logger.info(f"    {sub_key}: {sub_val:.4f}")
            else:
                logger.info(f"  {key}: {val:.4f}")
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
    # Final eval at iter_num=max_iters. In-process: run the eval block under EMA. Daemon: write a
    # final snapshot + signal shutdown → the daemon runs the final eval (+ ema_best), then drain it.
    # ema_final weights are written by the trainer below regardless (no dependency on the daemon).
    if use_eval_daemon:
        logger.info("Signaling eval daemon shutdown + final eval...")
        live_cpu, shadow_cpu = _clone_trainable_cpu(unoptimized_model, ema)
        snapshot_writer.submit(cfg.setup.max_iters, live_cpu, shadow_cpu)
        snapshot_writer.close()                       # flush the final snapshot to disk first
        ipc.signal_shutdown(eval_run_dir, cfg.setup.max_iters)
        try:
            daemon_proc.wait(timeout=EVAL_DAEMON_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("Eval daemon did not exit within timeout; terminating.")
            daemon_proc.terminate()
        _drain_and_log(eval_run_dir, incremental_audio_tables, cfg.wandb.log)
    else:
        # swap_in writes the final safetensors below separately (non-reentrant → can't nest the two).
        final_eval_deps = EvalDeps(
            iter_num=cfg.setup.max_iters,
            unoptimized_model=unoptimized_model,
            compiled_model=model,
            sampling_rate=sampling_rate,
            device=device,
            cfg=cfg,
            val_datasets=[dev_dataset, test_dataset],
            val_refs=val_refs,
            train_refs=train_refs,
            custom_prompts=custom_prompts,
            overfit_ref_batch=overfit_ref_batch,
        )
        best_val_loss = run_eval_block(
            final_eval_deps,
            train_loader, dev_loader, test_loader,
            loss_wrapper, ema, best_val_loss,
            incremental_audio_tables,
        )

    # Final full-state checkpoint at the endpoint. The periodic save only fires on
    # checkpoint_interval multiples, so without this a COMPLETED run could only resume from the
    # last multiple — losing up to checkpoint_interval steps of model+optimizer+EMA+data state
    # (e.g. a 40000-step run last checkpointed at 35000). Resume from here by raising max_iters to
    # continue training. Gated like the periodic save so diagnostic modes (interval=0) stay write-free.
    if cfg.setup.checkpoint_interval > 0:
        final_ckpt_path = save_resume_checkpoint(
            unoptimized_model, optimizer, ema,
            model_cfg_dict=model_cfg_dict, token_vocabulary_size=token_vocabulary_size,
            sampling_rate=sampling_rate, iter_num=cfg.setup.max_iters - 1, best_val_loss=best_val_loss,
            epoch=current_epoch, batch_idx=current_batch_idx + 1, cur_kimg=cur_kimg,
            wandb_log=cfg.wandb.log,
        )
        logger.info(f"Saved final crash-recovery checkpoint (iter {cfg.setup.max_iters - 1}) to {final_ckpt_path}")

    if cfg.setup.final_safetensors:
        final_path = CHECKPOINTS_DIR / 'ema_final.safetensors'
        with ema.swap_in(unoptimized_model) if ema is not None else nullcontext():
            atomic_save_safetensors(unoptimized_model, final_path)
        logger.info(f"Saved final EMA weights for offline inference to {final_path}")


if __name__ == "__main__":
    # Memory expansion → mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train()