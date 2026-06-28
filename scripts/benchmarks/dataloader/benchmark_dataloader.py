import time
import os
import sys
import threading
import logging
import torch
import numpy as np
import psutil
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.model import NaturalSpeech2Model, LossWrapper
from naturalspeech2.utils.utils import compute_denominators
from naturalspeech2.utils.compile_tracking import read_compile_stats

logger = logging.getLogger(__name__)

NUM_BENCHMARK_STEPS = int(os.environ["NUM_BENCHMARK_STEPS"])
WARMUP_STEPS = int(os.environ["WARMUP_STEPS"])

# Host-resource safeguards (the only OOM risk here is host RAM from the DataLoader workers — bucket
# batch sizes are pre-validated, so VRAM can't OOM and num_workers doesn't change it).
#   - run_benchmark_dataloader.sh stops the ascending num_workers sweep on this exit code (a higher
#     worker count would only use more host RAM, so the rest would fail too).
#   - Per-step RAM check (main thread) aborts gracefully before the OS OOM killer fires (which could
#     kill a co-tenant job on a shared box).
#   - Watchdog thread hard-exits on any of: a wedged step (no progress > STEP_TIMEOUT), host RAM ≥
#     RAM_ABORT_PERCENT, or swap GROWTH ≥ SWAP_ABORT_GB. The swap guard catches the thrashing the other
#     two miss — the kernel can hold RAM% under the limit by paging out, and steps that merely crawl
#     (not fully wedge for STEP_TIMEOUT) never trip the stall timer, yet the box is already unusable.
ABORT_EXIT_CODE = 42
RAM_ABORT_PERCENT = float(os.environ.get("RAM_ABORT_PERCENT", "90"))
SWAP_ABORT_GB = float(os.environ.get("SWAP_ABORT_GB", "2"))
STEP_TIMEOUT_SECONDS = float(os.environ.get("STEP_TIMEOUT_SECONDS", "300"))

@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def benchmark(cfg: DictConfig):

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required for benchmarking.")
    device = cfg.setup.device

    # Init W&B up front so the whole run — init logs, the loop/bias warnings, the per-step series and the
    # final summary — is captured in the run's output.log, not only the console.
    if cfg.wandb.log:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Build the loader through the SAME entry point training uses (create_dataloader → chunked train
    # store, bucketed sampler, collate, pin_memory), so the timed read path is exactly production's.
    # num_workers + resample_on_the_fly come from cfg (swept by run_benchmark_dataloader.sh); the
    # train subset is whatever dataset.max_train_clips was preprocessed (HF split-slicing is unsupported).
    # build_if_missing=False → READ-ONLY: the benchmark consumes an already-preprocessed split and fails
    # loud if it is absent, never kicking off a (potentially huge) preprocessing run for the setting.
    logger.info("--- Initializing Dataset + Dataloader (training path, read-only) ---")
    loader, dataset = create_dataloader(
        cfg, cfg.dataset.train_split, cfg.dataset.token_vocabulary_path, build_if_missing=False
    )

    # Unbiased-I/O guard: if the run consumes more batches than one epoch holds, the loader loops and
    # re-reads now-cached clips → warm-cache reads bias the disk timing low. Warn so the user grows the
    # preprocessed subset (dataset.max_train_clips) or lowers NUM_BENCHMARK_STEPS.
    total_steps = NUM_BENCHMARK_STEPS + WARMUP_STEPS
    if len(loader) < total_steps:
        logger.warning(
            f"Loader holds only {len(loader)} batches/epoch but {total_steps} steps were requested → it "
            f"will loop and warm the page cache mid-run, biasing I/O timings. Preprocess more clips or "
            f"lower NUM_BENCHMARK_STEPS."
        )

    logger.info("--- Initializing Model ---")
    model_cfg = model_cfg_from_omegaconf(cfg.model)
    model = NaturalSpeech2Model(
        model_cfg,
        token_vocabulary_size=dataset.tokenizer.token_vocabulary_size,
        sampling_rate=cfg.dataloader.sampling_rate,
    ).to(device)

    logger.info("Compiling model...")
    model = torch.compile(model)

    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True)
    ).to(device)

    # Match training's optimizer (param groups + betas + lr from cfg.training) so the timed GPU step
    # is representative, not a hardcoded AdamW.
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay,
        cfg.training.learning_rate,
        (cfg.training.beta1, cfg.training.beta2),
    )

    iter_times = []
    cpu_times = []
    gpu_times = []
    gpu_fwd_times = []
    gpu_h2d_times = []
    gpu_bwd_times = []
    cpu_gpu_ratios = []
    starved_steps = 0

    def _abort(code: int, status: str):
        # Clean exit on resource exhaustion: close the W&B run, then signal the shell to stop the sweep.
        logger.error(f"Aborting num_workers={cfg.dataloader.num_workers} ({status}); stopping the sweep.")
        if cfg.wandb.log and wandb.run is not None:
            wandb.run.summary["summary/status"] = status
            wandb.finish(exit_code=1)
        sys.exit(code)

    logger.info(f"\nStarting benchmark: {NUM_BENCHMARK_STEPS} steps ({WARMUP_STEPS} warmup)")

    # Watchdog thread, armed BEFORE the first fetch so a worker deadlock on startup is covered too. It
    # hard-exits (os._exit frees the box without waiting on possibly-stuck cleanup; the shell stops the
    # sweep on the code) on any of three triggers — the main-thread per-step check below stays the graceful
    # path, while this is the net that fires even when the loop is too wedged or laggy to self-check:
    #   - STALL: no step progress for STEP_TIMEOUT_SECONDS — a fully wedged step.
    #   - RAM:   host RAM ≥ RAM_ABORT_PERCENT.
    #   - SWAP:  swap grew ≥ SWAP_ABORT_GB since startup — thrashing onset, the one the other two miss (the
    #            kernel can keep RAM% under the limit by paging out, and a box that merely crawls never trips
    #            the stall timer). Baseline-subtracted: pre-existing / co-tenant swap doesn't count.
    last_progress = [time.monotonic()]
    watchdog_stop = threading.Event()
    swap_baseline = psutil.swap_memory().used

    def _watchdog():
        while not watchdog_stop.wait(timeout=5.0):
            stalled = time.monotonic() - last_progress[0]
            ram_percent = psutil.virtual_memory().percent
            swap_growth_gb = (psutil.swap_memory().used - swap_baseline) / 1e9
            if stalled > STEP_TIMEOUT_SECONDS:
                logger.error(
                    f"Stall watchdog: no step progress for {stalled:.0f}s (> {STEP_TIMEOUT_SECONDS:.0f}s) "
                    f"at num_workers={cfg.dataloader.num_workers} — likely RAM thrashing or a stuck "
                    f"worker. Hard-exiting to free the box."
                )
                os._exit(ABORT_EXIT_CODE)
            if ram_percent >= RAM_ABORT_PERCENT:
                logger.error(
                    f"RAM watchdog (thread): host RAM at {ram_percent:.0f}% (≥ {RAM_ABORT_PERCENT:.0f}%) "
                    f"at num_workers={cfg.dataloader.num_workers}. Hard-exiting to free the box."
                )
                os._exit(ABORT_EXIT_CODE)
            if swap_growth_gb >= SWAP_ABORT_GB:
                logger.error(
                    f"Swap watchdog: {swap_growth_gb:.1f} GB pushed to swap since start "
                    f"(≥ {SWAP_ABORT_GB:.0f} GB) at num_workers={cfg.dataloader.num_workers} — the box is "
                    f"thrashing (RAM% can stay under the limit while the kernel swaps). Hard-exiting."
                )
                os._exit(ABORT_EXIT_CODE)

    threading.Thread(target=_watchdog, daemon=True).start()

    loader_iter = iter(loader)
    batch = next(loader_iter)
    denominators = compute_denominators([batch], cfg)
    # Only tensors go to the GPU / into the model; the collate also returns `text` (list[str]) for the
    # eval block, which the forward doesn't accept.
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_fwd_end = torch.cuda.Event(enable_timing=True)
    gpu_h2d_start = torch.cuda.Event(enable_timing=True)
    gpu_bwd_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_BENCHMARK_STEPS + WARMUP_STEPS):
        last_progress[0] = time.monotonic()   # feed the stall watchdog

        # Proactive host-RAM watchdog: abort cleanly before the OS OOM killer fires (it could kill a
        # co-tenant job on a shared box). High num_workers × prefetch buffers are the usual culprit.
        ram_percent = psutil.virtual_memory().percent
        if ram_percent >= RAM_ABORT_PERCENT:
            logger.error(f"Host RAM at {ram_percent:.0f}% (≥ {RAM_ABORT_PERCENT:.0f}%).")
            _abort(ABORT_EXIT_CODE, "RAM_ABORT")

        # Snapshot Dynamo compile counter pre-forward (unique_graphs = recompile signal)
        compile_count_before = read_compile_stats()["unique_graphs"]

        start_iter = time.perf_counter()

        gpu_start.record()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = model(**batch)
            loss, _, _ = loss_wrapper(loss_dict, step=i, denominators=denominators)
        gpu_fwd_end.record()

        looped = False
        try:
            batch_cpu = next(loader_iter)
        except StopIteration:
            logger.warning(f"Step {i}: Dataloader ran out of unique data and looped! Page cache may now artificially inflate disk I/O speeds.")
            loader_iter = iter(loader)
            batch_cpu = next(loader_iter)
            looped = True

        denominators = compute_denominators([batch_cpu], cfg)

        gpu_h2d_start.record()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch_cpu.items() if isinstance(v, torch.Tensor)}

        gpu_bwd_start.record()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gpu_end.record()

        # Capture CPU time before blocking for the GPU
        cpu_end = time.perf_counter()

        torch.cuda.synchronize(device) # Crucial for accurate GPU timing
        end_iter = time.perf_counter()

        if i >= WARMUP_STEPS:
            # wall-clock iter time (incl. sync)
            it = end_iter - start_iter
            # CPU dispatch + dataloading
            ct = cpu_end - start_iter

            # GPU compute (excl. CPU idle, incl. PCIe H2D)
            gt_fwd = gpu_start.elapsed_time(gpu_fwd_end)
            gt_h2d = gpu_h2d_start.elapsed_time(gpu_bwd_start)
            gt_bwd = gpu_bwd_start.elapsed_time(gpu_end)
            gt = (gt_fwd + gt_h2d + gt_bwd) / 1000.0

            # Skip on loop — worker spin-up poisons P95
            if looped:
                logger.warning(f"Step {i}: Skipping metrics due to dataloader worker spin-up latency.")
                continue

            # Detect Dynamo recompile this step
            compile_count_after = read_compile_stats()["unique_graphs"]
            if compile_count_after > compile_count_before:
                logger.warning(f"Step {i}: Detected graph recompile ({it:.2f}s). Skipping metrics.")
                continue

            iter_times.append(it)
            cpu_times.append(ct)
            gpu_times.append(gt)
            gpu_fwd_times.append(gt_fwd / 1000.0)
            gpu_h2d_times.append(gt_h2d / 1000.0)
            gpu_bwd_times.append(gt_bwd / 1000.0)

            # CPU dispatch + dataloading must beat GPU compute (fully mask dispatch overhead). The ratio
            # ct/gt is the headroom: < 1 healthy, → 1 fragile, > 1 starved (more telling than the flag).
            ratio = ct / gt
            cpu_gpu_ratios.append(ratio)
            is_starved = ct > gt
            if is_starved:
                starved_steps += 1

            if cfg.wandb.log:
                wandb.log({
                    "Dataloader Benchmark/Total Iteration Time": it,
                    "Dataloader Benchmark/CPU Dispatch Time": ct,
                    "Dataloader Benchmark/GPU Compute Time": gt,
                    "Dataloader Benchmark/CPU-GPU Ratio": ratio,
                    "Dataloader Benchmark/Starved Flag": 1 if is_starved else 0,
                }, step=i - WARMUP_STEPS)

    watchdog_stop.set()   # disarm before the (network-bound) summary upload

    valid_steps = len(iter_times)
    logger.info("\n--- Benchmark Results ---")
    if valid_steps > 0:
        logger.info(f"Total Iter Time   : {np.mean(iter_times):.4f}s avg | P95: {np.percentile(iter_times, 95):.4f}s")
        logger.info(f"CPU Dispatch Time : {np.mean(cpu_times):.4f}s avg | P95: {np.percentile(cpu_times, 95):.4f}s")
        logger.info(f"GPU Compute Time  : {np.mean(gpu_times):.4f}s avg | P95: {np.percentile(gpu_times, 95):.4f}s")
        logger.info(f"  ├─ Forward Pass : {np.mean(gpu_fwd_times):.4f}s avg | P95: {np.percentile(gpu_fwd_times, 95):.4f}s")
        logger.info(f"  ├─ H2D Transfer : {np.mean(gpu_h2d_times):.6f}s avg | P95: {np.percentile(gpu_h2d_times, 95):.6f}s")
        logger.info(f"  └─ Backward Pass: {np.mean(gpu_bwd_times):.4f}s avg | P95: {np.percentile(gpu_bwd_times, 95):.4f}s")
        logger.info(f"CPU/GPU Ratio     : {np.mean(cpu_gpu_ratios):.3f} avg | P95: {np.percentile(cpu_gpu_ratios, 95):.3f} (headroom: < 1 healthy)")
        logger.info(f"Starvation Rate   : {(starved_steps / valid_steps) * 100:.1f}% ({starved_steps}/{valid_steps} steps)")
        logger.info(f"Overall Status    : {'STARVED ❌' if starved_steps > 0 else 'HEALTHY ✅'}")

        if cfg.wandb.log:
            # Structured run-level summary → the W&B runs table compares worker counts at a glance.
            wandb.run.summary.update({
                "summary/num_workers": cfg.dataloader.num_workers,
                "summary/setting": "otf" if cfg.dataloader.resample_on_the_fly else "pre",
                "summary/valid_steps": valid_steps,
                "summary/batches_per_epoch": len(loader),
                "summary/iter_time_mean": float(np.mean(iter_times)),
                "summary/iter_time_p95": float(np.percentile(iter_times, 95)),
                "summary/iter_time_max": float(np.max(iter_times)),
                "summary/cpu_time_mean": float(np.mean(cpu_times)),
                "summary/cpu_time_p95": float(np.percentile(cpu_times, 95)),
                "summary/cpu_time_max": float(np.max(cpu_times)),
                "summary/gpu_time_mean": float(np.mean(gpu_times)),
                "summary/cpu_gpu_ratio_mean": float(np.mean(cpu_gpu_ratios)),
                "summary/cpu_gpu_ratio_p95": float(np.percentile(cpu_gpu_ratios, 95)),
                "summary/starvation_rate": starved_steps / valid_steps,
                "summary/status": "STARVED" if starved_steps > 0 else "HEALTHY",
            })
    else:
        logger.warning("No valid steps recorded (all skipped as warmup/loop/recompile). Nothing to report.")

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    benchmark()
