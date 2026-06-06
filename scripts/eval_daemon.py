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
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper
from naturalspeech2.eval import set_metric_device
from naturalspeech2.eval import ipc
from naturalspeech2.eval.runner import (
    build_fixed_refs_data,
    run_decoupled_eval,
    atomic_save_safetensors,
)
from naturalspeech2.paths import CHECKPOINTS_DIR
from naturalspeech2.utils.utils import setup_file_logger

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
        logger.info("Compiling daemon eval model (this takes a minute)...")
        loss_model = torch.compile(model)
    else:
        loss_model = model

    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True),
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
        "model": model, "loss_model": loss_model, "loss_wrapper": loss_wrapper,
        "train_loader": train_loader, "dev_loader": dev_loader, "test_loader": test_loader,
        "val_datasets": [dev_dataset, test_dataset], "val_refs": val_refs, "train_refs": train_refs,
    }


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
    if report.new_best:
        best_val_loss = report.best_val_loss
        # model currently holds the EMA shadow (loaded last in run_decoupled_eval) → save as ema_best.
        atomic_save_safetensors(ctx["model"], CHECKPOINTS_DIR / "ema_best.safetensors")
        ipc.save_eval_state(CHECKPOINTS_DIR / "eval_state.json",
                            {"best_val_loss": best_val_loss, "best_step": step})
        logger.info(f"New best val loss {best_val_loss:.4f} at step {step} → wrote ema_best.")
    ipc.write_results(run_dir=ctx["run_dir"], report=report)
    logger.info(f"Snapshot step {step} done in {time.perf_counter() - t0:.1f}s.")
    return best_val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="RAM-backed IPC dir (from the trainer)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logger(logger, CHECKPOINTS_DIR / "eval_daemon.log")
    logger.info(f"Eval daemon starting; run_dir={run_dir}")

    ctx = build(run_dir)
    ctx["run_dir"] = run_dir
    best_val_loss = ipc.load_eval_state(CHECKPOINTS_DIR / "eval_state.json").get("best_val_loss", float("inf"))
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
