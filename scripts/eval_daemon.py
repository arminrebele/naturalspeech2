"""Decoupled eval daemon — runs the eval block on the 2nd GPU so the trainer never pauses.

Launched by scripts/train.py as a fresh subprocess with CUDA_VISIBLE_DEVICES=1 (clean CUDA
context → its cuda:0 == physical GPU1). All trainer↔daemon comms are files via
naturalspeech2.eval.ipc. Build-once, then watch the snapshot marker: on each new step, copy the
trainer's trainable weights into the resident model and eval LIVE + EMA, write results back, and
own ema_best + best-tracking. Crash-isolated: an eval failure kills only this process; the
trainer's supervisor respawns it. Eager by default (setup.eval_daemon.compile opts into compile).
"""
import argparse
import logging
import random
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper
from naturalspeech2.utils.compile_tracking import compile_kwargs, read_compile_stats, format_break_reasons
from naturalspeech2.eval import set_metric_device
from naturalspeech2.eval import ipc
from naturalspeech2.eval.runner import (
    build_fixed_refs_data,
    run_decoupled_eval,
    atomic_save_safetensors,
)
from naturalspeech2.paths import run_checkpoint_dir
from naturalspeech2.utils.utils import setup_file_logger, generate_dummy_batch
from naturalspeech2.utils.warning_filters import install_warning_filters

logger = logging.getLogger("eval_daemon")


def build(run_dir: Path):
    """Read the handshake + build everything that persists across snapshots (model, loaders, refs,
    ASR). Returns the loop state."""
    init = ipc.read_daemon_init(run_dir)
    cfg = OmegaConf.create(init["cfg"])
    device = init["daemon_device"]
    sr = init["sampling_rate"]
    seed = cfg.seed

    assert torch.cuda.is_available(), "eval daemon requires an NVIDIA GPU"
    set_metric_device(device)            # WER ASR on the daemon's own card
    torch.manual_seed(seed)
    random.seed(seed)

    # Model from the trainer's STORED cfg (matches a resume rebuild, not drifted Hydra cfg).
    model_cfg = model_cfg_from_omegaconf(init["model_cfg"])
    model = NaturalSpeech2Model(
        model_cfg,
        token_vocabulary_size=init["token_vocabulary_size"],
        sampling_rate=sr,
    ).to(device)
    model.eval()
    model._inference_tokenizer = PhonemeTokenizer(
        token_vocabulary_path=init["token_vocabulary_path"], with_backend=True,
    )
    model._inference_sampling_rate = sr

    # estimate_loss handle: compiled (faster forwards) or the eager model itself. Weight-load +
    # audio gen always go through the eager `model` (clean param names; compile prefixes them).
    if cfg.setup.eval_daemon.compile:
        logger.info(f"Compiling daemon eval model (this takes a minute)... "
                    f"[dynamic={cfg.setup.compile.dynamic}, mode={cfg.setup.compile.mode}]")
        loss_model = torch.compile(model, **compile_kwargs(cfg.setup.compile))
        _prewarm_eval_compile(loss_model, cfg, init["token_vocabulary_size"], device)
    else:
        loss_model = model

    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True),
        loss_warmup_hold_steps=OmegaConf.to_container(cfg.model.loss_warmup_hold_steps, resolve=True),
    ).to(device)

    nw = cfg.setup.eval_daemon.num_workers
    bsd = cfg.setup.eval_daemon.batch_size_divisor
    tok_path = init["token_vocabulary_path"]
    train_loader, train_dataset = create_dataloader(cfg, cfg.dataset.train_split, tok_path, num_workers=nw, batch_size_divisor=bsd)
    dev_loader, dev_dataset = create_dataloader(cfg, cfg.dataset.dev_split, tok_path, num_workers=nw, batch_size_divisor=bsd)
    test_loader, test_dataset = create_dataloader(cfg, cfg.dataset.test_split, tok_path, num_workers=nw, batch_size_divisor=bsd)
    if bsd > 1:
        base_gas = cfg.setup.gradient_accumulation_steps
        logger.info(f"Eval batch reduced on the 2nd GPU: bucket batch_size //{bsd}, grad_accum x{bsd} "
                    f"(base {base_gas} -> {base_gas * bsd}) -> logical batch + total samples unchanged, "
                    f"~{bsd}x lower forward VRAM.")

    # Fixed refs — same single seed + dataset order as the trainer → identical pooled draw; GT-floor
    # WER + prompt SIM-o embedding cached once. Build only the ref sets the configured audio_tables need
    # (production has no fixed_train_refs → don't sample/ASR/SIM-o over the train slice for nothing).
    do_wer = "wer" in cfg.setup.eval_metrics
    do_sim_o = "sim_o" in cfg.setup.eval_metrics
    n_refs = cfg.setup.num_audio_refs
    prompt_samples_len = int(cfg.model.prompt_seconds * sr)
    val_refs, train_refs = [], []
    if "fixed_val_refs" in cfg.setup.audio_tables:
        val_refs = build_fixed_refs_data([dev_dataset, test_dataset], n_refs, prompt_samples_len, random.Random(seed),
                                         sampling_rate=sr, compute_gt_wer=do_wer, compute_sim_emb=do_sim_o)
    if "fixed_train_refs" in cfg.setup.audio_tables:
        train_refs = build_fixed_refs_data([train_dataset], n_refs, prompt_samples_len, random.Random(seed),
                                           sampling_rate=sr, compute_gt_wer=do_wer, compute_sim_emb=do_sim_o)

    return {
        "cfg": cfg, "device": device, "sr": sr,
        # Per-run checkpoint subdir — same derivation as the trainer (run_checkpoint_dir(log_name))
        # so both write ema_best + eval_state into the same lineage dir.
        "ckpt_dir": run_checkpoint_dir(cfg.setup.log_name),
        "model": model, "loss_model": loss_model, "loss_wrapper": loss_wrapper,
        "train_loader": train_loader, "dev_loader": dev_loader, "test_loader": test_loader,
        "val_datasets": [dev_dataset, test_dataset], "val_refs": val_refs, "train_refs": train_refs,
    }


def _prewarm_eval_compile(loss_model, cfg, vocab_size, device):
    """Compile every eval-mode graph up front (one no_grad dummy fwd per bucket, B shrunk by the
    daemon's batch_size_divisor to match the real eval). torch.compile is lazy → without this the
    FIRST snapshot pays the ~1-min compile, and since the snapshot writer keeps only the newest step,
    a slow first eval can make the daemon SKIP the next snapshot(s). Model is already in eval mode (set
    in build) → this compiles exactly the graphs estimate_loss reuses (later snapshots = cache hits)."""
    bsd = cfg.setup.eval_daemon.batch_size_divisor
    t0 = time.perf_counter()
    lo = 1
    with torch.no_grad():
        for b in sorted(cfg.dataloader.bucket_mapping, key=lambda x: x.audio_length):
            batch = generate_dummy_batch(
                batch_size=max(1, b.batch_size // bsd), audio_samples=b.audio_length,
                phoneme_samples=b.phoneme_length, min_audio_samples=lo,
                vocab_size=vocab_size, device=device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                loss_model(**batch)
            lo = b.audio_length + 1
    s = read_compile_stats()
    logger.info(f"Daemon eval pre-warm: {s['unique_graphs']} graphs, {s['graph_breaks_total']} breaks "
                f"[{format_break_reasons(s['break_reasons'])}] in {time.perf_counter() - t0:.0f}s.")


def evaluate_snapshot(ctx: dict, snap: dict, best_val_loss: float) -> float:
    """Eval one snapshot, write results + (if improved) ema_best. Returns the (possibly new) best."""
    cfg, step = ctx["cfg"], snap["step"]
    logger.info(f"Evaluating snapshot step {step} ...")
    t0 = time.perf_counter()
    report = run_decoupled_eval(
        model=ctx["model"], loss_model=ctx["loss_model"], loss_wrapper=ctx["loss_wrapper"],
        train_loader=ctx["train_loader"], dev_loader=ctx["dev_loader"], test_loader=ctx["test_loader"],
        live_trainable=snap["live"], shadow_trainable=snap["shadow"],
        val_refs=ctx["val_refs"], train_refs=ctx["train_refs"], val_datasets=ctx["val_datasets"],
        cfg=cfg, device=ctx["device"], prompt_seconds=cfg.model.prompt_seconds,
        sampling_rate=ctx["sr"], snapshot_step=step, prev_best_val_loss=best_val_loss,
    )
    # Independent daemon compile telemetry (separate process → own Dynamo counters). Scalars ride the
    # existing results→wandb drain (snapshot_step x-axis); reasons + cross-snapshot leak → daemon log.
    if cfg.setup.eval_daemon.compile:
        cstats = read_compile_stats()
        for key in ("unique_graphs", "graph_breaks_total", "n_break_reasons", "cache_size_limit"):
            report.scalars[f"Evaluation: Compile/{key}"] = cstats[key]
        reasons = set(cstats["break_reasons"])
        if reasons != ctx.get("_compile_reasons"):
            added = reasons - (ctx.get("_compile_reasons") or set())
            if added and ctx.get("_compile_reasons"):
                logger.warning(f"⚠️ New daemon graph-break reason(s): {sorted(added)} — "
                               f"expected only the 2 intended disable sites.")
            ctx["_compile_reasons"] = reasons
        prev_ug = ctx.get("_compile_ug")
        if prev_ug is not None and cstats["unique_graphs"] > prev_ug:
            logger.warning(f"⚠️ Daemon unique_graphs grew {prev_ug}→{cstats['unique_graphs']} across "
                           f"snapshots — eval shapes leaking (recompiling beyond the eval buckets).")
        ctx["_compile_ug"] = cstats["unique_graphs"]
    if report.new_best:
        best_val_loss = report.best_val_loss
        # model currently holds the EMA shadow (loaded last in run_decoupled_eval) → save as ema_best.
        atomic_save_safetensors(ctx["model"], ctx["ckpt_dir"] / "ema_best.safetensors")
        ipc.save_eval_state(ctx["ckpt_dir"] / "eval_state.json",
                            {"best_val_loss": best_val_loss, "best_step": step})
        logger.info(f"New best val loss {best_val_loss:.4f} at step {step} → wrote ema_best.")
    ipc.write_results(run_dir=ctx["run_dir"], report=report)
    logger.info(f"Snapshot step {step} done in {time.perf_counter() - t0:.1f}s.")
    return best_val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="RAM-backed IPC dir (from the trainer)")
    ap.add_argument("--log-dir", required=True, help="run's log dir (scratch vs main) for eval_daemon.log")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    log_dir = Path(args.log_dir)

    setup_file_logger(logger, log_dir / "eval_daemon.log", root=True)
    logging.captureWarnings(True)   # warnings.warn → logging → eval_daemon.log (matches console/wandb)
    install_warning_filters()       # drop the same known-benign torch/phonemizer/s3prl spam as the trainer

    # Child process → an uncaught traceback goes to stderr, which (unlike the trainer's) wandb never
    # captures. Route it through logging → eval_daemon.log → the live-uploaded wandb file, so a daemon
    # crash is visible post-mortem. Then let the process die (no swallow). Ctrl-C stays default.
    def _log_uncaught(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Uncaught exception in eval daemon — crashing:", exc_info=(exc_type, exc, tb))
    sys.excepthook = _log_uncaught

    logger.info(f"Eval daemon starting; run_dir={run_dir}")

    ctx = build(run_dir)
    ctx["run_dir"] = run_dir
    # eval_state.json (best-tracking) + ema_best live in this run's lineage subdir (derived in build()).
    ckpt_dir = ctx["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = ipc.load_eval_state(ckpt_dir / "eval_state.json").get("best_val_loss", float("inf"))
    poll = ctx["cfg"].setup.eval_daemon.poll_interval_s
    logger.info(f"Eval daemon ready (best_val_loss={best_val_loss}); watching for snapshots.")

    last_step = -1
    while True:
        control = ipc.read_control(run_dir)
        marker = ipc.read_marker(run_dir)

        if marker is not None and marker > last_step:
            snap = ipc.read_snapshot(run_dir)
            if snap is not None and snap["step"] > last_step:
                best_val_loss = evaluate_snapshot(ctx, snap, best_val_loss)
                last_step = snap["step"]
                continue   # immediately check for a newer snapshot (coalesce)

        if control and control.get("shutdown"):
            if marker is None or last_step >= marker:
                logger.info("Shutdown signal received and caught up; exiting.")
                break
            continue       # a newer (final) snapshot exists → pick it up next iteration

        time.sleep(poll)


if __name__ == "__main__":
    main()
