import os
import sys
import json
import time
import math
import queue
import signal
import ctypes
import itertools
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
import torch._dynamo
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.inference import compute_inference_data_loss, generate_audio
from naturalspeech2.eval import resolve_metric_device, set_metric_device, ipc
from naturalspeech2.eval.runner import (
    estimate_loss,
    get_loss_section,
    _mean_skip_nan,
    build_fixed_refs_data,
    generate_ref_audio,
    atomic_save_safetensors,
)
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper, GradientAnalyzer
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.paths import CHECKPOINTS_DIR, PROJECT_ROOT
from naturalspeech2.utils.ema import EMA
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators

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
    dev_dataset: Any                   # DatasetWrapper — typed Any to avoid forward-decl noise
    table_2_refs: list
    test_refs: list
    custom_prompts: list
    overfit_ref_batch: Optional[dict]  # cached at iter 0 when overfit_batch is in audio_tables
    metrics_out: dict = field(default_factory=dict)  # per-eval scalar metrics (WER, …) → merged into eval_payload


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


def build_fixed_refs(dataset, n_refs, prompt_samples_len, sampling_rate, rng, do_wer):
    """Trainer-side fixed refs: shared data builder (deterministic selection + GT-floor WER cached
    once) wrapped with wandb.Audio for the original/prompt clips (reused across evals). Call only
    when wandb is active."""
    return [
        {
            "original_audio": wandb.Audio(r.original_np, sample_rate=sampling_rate),
            "prompt_audio": wandb.Audio(r.prompt_np, sample_rate=sampling_rate),
            "prompt_tensor": r.prompt_tensor,
            "original_np": r.original_np,
            "text": r.text,
            "gt_wer": r.gt_wer,   # GT-floor WER, computed once at build
        }
        for r in build_fixed_refs_data(
            dataset, n_refs, prompt_samples_len, rng,
            sampling_rate=sampling_rate, compute_gt_wer=do_wer,
        )
    ]


def _render_fixed_refs_table(deps: EvalDeps, refs: list, title: str, split: str) -> tuple[str, list, list]:
    """Generate audio on a fixed reference set (shared by dev + test tables).

    If "wer" in cfg.setup.eval_metrics: per-clip synth WER column + per-split means (synth WER and
    the GT-floor cached on each ref) → gap (synth − floor) = honest signal.
    """
    sr = deps.sampling_rate
    do_wer = "wer" in deps.cfg.setup.eval_metrics
    num_table_rows = deps.cfg.setup.num_table_rows
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
        gen, synth_wer, hyp = generate_ref_audio(deps.unoptimized_model, ref["prompt_tensor"], ref["text"], sr, do_wer)
        if do_wer:
            synth_wers.append(synth_wer)
            gt_wers.append(ref["gt_wer"])
        if in_table:
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
            rows.append(row)

    if do_wer:
        deps.metrics_out[f"Evaluation: Metrics/{split}-WER"] = _mean_skip_nan(synth_wers)
        deps.metrics_out[f"Evaluation: Metrics/{split}-WER-gt"] = _mean_skip_nan(gt_wers)

    return (title, columns, rows)


def render_fixed_dev_refs_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the fixed dev references (trained-on when training on dev)."""
    return _render_fixed_refs_table(deps, deps.table_2_refs, "Eval Audio: dev (trained-on)", "dev")


def render_fixed_test_refs_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the fixed held-out test references (never trained on)."""
    return _render_fixed_refs_table(deps, deps.test_refs, "Eval Audio: test (held-out)", "test")


def render_fixed_val_refs_table(deps: EvalDeps) -> tuple[str, list, list]:
    """Generate audio on the combined dev+test held-out VALIDATION references.

    dev+test together = validation set (speaker-disjoint from train and each other; paper reports
    only on external VCTK / LibriSpeech), so one table + one combined val-WER under 'dev' (matches
    the chained 'dev' loss in estimate_loss)."""
    val_refs = [r for pair in itertools.zip_longest(deps.table_2_refs, deps.test_refs)
                for r in pair if r is not None]
    return _render_fixed_refs_table(
        deps, val_refs,
        "Eval Audio: dev+test (held-out validation)", "dev",
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
    "fixed_test_refs": render_fixed_test_refs_table,
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
    """Eval-trigger schedule. Normal: every eval_interval. Dropout trials: only the converged
    tail (from dropout_eval_start_frac onward) at eval_interval — the decision metric needs only
    the tail, skipping the early run removes most eval overhead."""
    if iter_num <= 0:
        return False
    if cfg.setup.dropout_trial_run and iter_num < int(cfg.setup.dropout_eval_start_frac * cfg.setup.max_iters):
        return False
    return iter_num % cfg.setup.eval_interval == 0


def _append_dropout_trial_eval(out_path, step: int, losses: dict) -> None:
    """Append one eval point as a JSON line for the dropout-trial orchestrator: step +
    per-term (+total) held-out loss(es). Training on train → dev+test chained as one 'dev'
    record; only when dev IS the train split do dev and test appear separately."""
    record = {"step": step}
    for split in ("dev", "test"):
        if split in losses:
            record[f"{split}_total"] = losses[split]["total_loss"]
            record[split] = losses[split]["logged_losses"]
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


def _spawn_eval_daemon(run_dir):
    """Launch the eval daemon as a fresh subprocess pinned to GPU1 (clean CUDA context, no fork)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(EVAL_DAEMON_SCRIPT), "--run-dir", str(run_dir)],
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
    best_dev_loss: float,
    incremental_audio_tables: dict,
) -> float:
    """Single eval pass under EMA-swapped weights → possibly-updated best_dev_loss.

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
                    eval_train=not deps.cfg.setup.dropout_trial_run,
                )
                logged = ", ".join(f"{s}={losses[s]['total_loss']:.4f}"
                                   for s in ('train', 'dev', 'test') if s in losses)
                logger.info(f"Step {deps.iter_num}: eval losses  {logged}")
                for split_name, section in [('train', 'Train'), ('dev', 'Dev'), ('test', 'Test')]:
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
                and 'dev' in losses
                and losses['dev']['total_loss'] < best_dev_loss
            ):
                best_dev_loss = losses['dev']['total_loss']
                logger.info(f"Saving new best model to {CHECKPOINTS_DIR}")
                atomic_save_safetensors(deps.unoptimized_model, CHECKPOINTS_DIR / 'ema_best.safetensors')
    finally:
        deps.unoptimized_model.train()

    if deps.cfg.setup.dropout_trial_run and losses is not None:
        _append_dropout_trial_eval(PROJECT_ROOT / deps.cfg.setup.dropout_trial_out, deps.iter_num, losses)

    eval_payload.update(deps.metrics_out)

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
    setup_file_logger(logger, log_dir / log_name)

    # Decoupled eval daemon vs in-process sync eval. Daemon = standard 2-GPU training run only;
    # single-GPU AND every diagnostic mode (overfit/loss/grad/dropout) keep in-process eval (they
    # need trainer-local state and run on one card). Requires EMA (best-selection uses the shadow).
    is_diagnostic_run = (
        cfg.setup.overfit_single_batch or cfg.setup.loss_analysis_run
        or cfg.setup.gradient_analysis_run or cfg.setup.dropout_trial_run
    )
    use_eval_daemon = (
        cfg.setup.eval_daemon.enabled
        and torch.cuda.device_count() > 1
        and cfg.model.ema.enabled
        and not is_diagnostic_run
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
        if cfg.setup.eval_daemon.enabled and not is_diagnostic_run:
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

    # Eval block's static text prompts. generate_audio re-phonemizes per call
    # (~50 ms × 4 prompts × eval intervals — negligible).
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
    # Persistent INCREMENTAL wandb.Tables (keyed by name) → eval audio accumulates across firings.
    incremental_audio_tables: dict = {}

    # Instantiate Model
    if cfg.setup.init_from == 'scratch':
        logger.info("Initializing a new model from scratch...")
        # Fresh run → drop the daemon's persistent best-tracking from a prior run in this dir, so the
        # ema_best gate restarts from inf (matches the in-process best_dev_loss=1e9 reset). Stale weight
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
        best_dev_loss = checkpoint['best_dev_loss']
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

    logger.info("Compiling the model... (this takes a minute)")
    unoptimized_model = model
    model = torch.compile(model)

    # Outside the wandb.log gate so render_fixed_dev_refs_table iterates safely on
    # wandb.log=False debug runs (empty list → no rows). In daemon mode the daemon builds its own
    # refs → trainer skips them (avoids loading the WER ASR on the training card).
    table_2_refs = []
    test_refs = []
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

        if use_eval_daemon:
            # Eval logged from drained daemon results against a custom x-axis (the true snapshot
            # step) → eval curves don't collapse when the trainer is many steps ahead of a stale eval.
            wandb.define_metric("eval/snapshot_step")
            wandb.define_metric("Evaluation: *", step_metric="eval/snapshot_step")

        if not use_eval_daemon:
            num_static_refs = cfg.setup.num_audio_refs
            prompt_samples_len = int(cfg.model.prompt_seconds * sampling_rate)
            # Dedicated seeded RNGs → eval reference clips identical across runs, decoupled from other
            # global-random usage (cross-run A/B). Training on dev: dev refs = trained-on, test = held-out.
            do_wer = "wer" in cfg.setup.eval_metrics
            table_2_refs = build_fixed_refs(dev_dataset, num_static_refs, prompt_samples_len, sampling_rate, random.Random(cfg.seed), do_wer)
            test_refs = build_fixed_refs(test_dataset, num_static_refs, prompt_samples_len, sampling_rate, random.Random(cfg.seed + 1), do_wer)

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
        daemon_proc = _spawn_eval_daemon(eval_run_dir)

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

    if cfg.setup.dropout_trial_run:
        # Truncate any stale dump from a prior run.
        dropout_trial_out_path = PROJECT_ROOT / cfg.setup.dropout_trial_out
        dropout_trial_out_path.parent.mkdir(parents=True, exist_ok=True)
        dropout_trial_out_path.write_text("")
        
    if cfg.setup.gradient_analysis_run:
        grad_analysis_start_iter = int(cfg.setup.max_iters * 0.8)   # last 20% of steps (stable logged values)
        grad_norm_accumulators = {}
        cos_sim_accumulators = {}
        grad_analysis_steps_counted = 0
        
    logger.info("Starting training loop...")
    last_log_time = time.perf_counter()
    last_log_iter = start_iter - 1

    # Filled on the first loop iter when overfit_batch table is active. Overfit cycling yields
    # the same 5 objects forever, so caching lookahead_queue[0] once gives a stable ref batch.
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
            else:
                eval_deps = EvalDeps(
                    iter_num=iter_num,
                    unoptimized_model=unoptimized_model,
                    compiled_model=model,
                    sampling_rate=sampling_rate,
                    device=device,
                    cfg=cfg,
                    dev_dataset=dev_dataset,
                    table_2_refs=table_2_refs,
                    test_refs=test_refs,
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
                # Balanced loss components; .item() inside log_interval → no per-step GPU sync.
                for k, v in accum_logged_losses.items():
                    log_payload[f"{get_loss_section(k, 'Train')}/{k}"] = v.item()

                # Each raw term's share of summed raw magnitudes — flat once balance stabilizes
                # (drift signal in the main run).
                raw_terms = {k: v for k, v in accum_logged_losses.items() if not k.endswith("_weighted")}
                raw_total = sum(raw_terms.values())
                for k, v in raw_terms.items():
                    log_payload[f"Train: Loss Composition (Raw Fraction)/{k}"] = (v / raw_total).item()

                # Gradient analysis only when logging (expensive)
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
                        daemon_proc = _spawn_eval_daemon(eval_run_dir)
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
            dev_dataset=dev_dataset,
            table_2_refs=table_2_refs,
            test_refs=test_refs,
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
            atomic_save_safetensors(unoptimized_model, final_path)
        logger.info(f"Saved final EMA weights for offline inference to {final_path}")


if __name__ == "__main__":
    # Memory expansion → mitigate fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train()