"""Compile-mode A/B + shape-leak / fragmentation stress test.

Two jobs in one GPU harness (dummy per-bucket batches, real fwd+bwd+opt step):
  1. A/B `setup.compile` modes (dynamic × mode) → steady-state ms/step + PEAK VRAM + unique_graphs +
     warmup wall. Buckets are iso-compute (~80k batch·frames each) → a simple mean over a few shuffled
     rounds is representative of the real training step.
  2. Per-mode leak check: a full extra pass over all buckets must add NO new graphs (the plateau =
     no shape leak). Replaces the old stale `total_breaks == num_buckets` verdict.

The shuffled round-robin also exercises the allocator across all bucket sizes back-to-back
(fragmentation stress); an OOM is caught per-mode, reported with the memory summary, and recorded as
data ("this mode doesn't fit the current batch budget") rather than killing the sweep.

Env knobs: COMPILE_SWEEP=auto (current config only) | fast (auto+static) | all (default, incl.
cudagraphs); COMPILE_TIMED_ROUNDS (default 6); COMPILE_MAX_WARMUP_PASSES (default 4).
NOTE: max-autotune modes spend minutes autotuning kernels at warmup — expected, one-off.
"""
import os
import random
import logging
import time
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import torch._dynamo

from naturalspeech2.paths import DATA_DIR, PROJECT_ROOT
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import LossWrapper, NaturalSpeech2Model
from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.utils.utils import setup_file_logger, compute_denominators, generate_dummy_batch
from naturalspeech2.utils.compile_tracking import read_compile_stats, format_break_reasons

logger = logging.getLogger(__name__)

# (label, dynamic, mode). dynamic: None=auto / False=static-per-bucket. mode: torch.compile mode.
_FULL_SWEEP = [
    ("auto / default (current)",          None,  "default"),
    ("static",                            False, "default"),
    ("static + autotune (no cudagraph)",  False, "max-autotune-no-cudagraphs"),
    ("static + autotune (cudagraphs)",    False, "max-autotune"),
]
TIMED_ROUNDS = int(os.environ.get("COMPILE_TIMED_ROUNDS", "6"))        # rounds × num_buckets timed steps
MAX_WARMUP_PASSES = int(os.environ.get("COMPILE_MAX_WARMUP_PASSES", "4"))


def _select_sweep() -> list:
    sel = os.environ.get("COMPILE_SWEEP", "all").lower()
    if sel == "auto":
        return _FULL_SWEEP[:1]
    if sel == "fast":
        return _FULL_SWEEP[:2]
    return _FULL_SWEEP


def _enhanced_buckets(cfg) -> list:
    """Buckets sorted by audio_length, each tagged with the min audio samples that route to it."""
    out, lo = [], 1
    for b in sorted(cfg.dataloader.bucket_mapping, key=lambda x: x.audio_length):
        out.append({"audio_length": b.audio_length, "phoneme_length": b.phoneme_length,
                    "batch_size": b.batch_size, "min_audio_samples": lo})
        lo = b.audio_length + 1
    return out


def _make_batch(bucket, vocab_size, device):
    return generate_dummy_batch(
        batch_size=bucket["batch_size"], audio_samples=bucket["audio_length"],
        phoneme_samples=bucket["phoneme_length"], min_audio_samples=bucket["min_audio_samples"],
        vocab_size=vocab_size, device=device)


def _train_step(model, loss_wrapper, optimizer, batch, cfg):
    denominators = compute_denominators([batch], cfg)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_dict = model(**batch)
        loss, _, _ = loss_wrapper(loss_dict, denominators=denominators)
    loss.backward()
    if cfg.training.grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _bench_mode(label, dynamic, mode, cfg, buckets, vocab_size, device) -> dict:
    """One compile config: fresh model → plateau-gated warmup → leak assertion → timed steady-state."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    model = NaturalSpeech2Model(model_cfg_from_omegaconf(cfg.model),
                                token_vocabulary_size=vocab_size,
                                sampling_rate=cfg.dataloader.sampling_rate).to(device)
    kwargs = {}
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    if mode != "default":
        kwargs["mode"] = mode
    model = torch.compile(model, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True),
    ).to(device)

    # Warmup: cycle all buckets until unique_graphs is stable for a full pass (statics + dynamic
    # promotion + AOTAutograd backward all compiled). Capped so a pathological leak can't spin.
    torch.cuda.reset_peak_memory_stats(device)
    t_warm0 = time.perf_counter()
    prev_ug, stable = -1, False
    for _ in range(MAX_WARMUP_PASSES):
        for b in buckets:
            _train_step(model, loss_wrapper, optimizer, _make_batch(b, vocab_size, device), cfg)
        torch.cuda.synchronize(device)
        ug = read_compile_stats()["unique_graphs"]
        stable = (ug == prev_ug)
        prev_ug = ug
        if stable:
            break
    warmup_s = time.perf_counter() - t_warm0
    warmup_peak = torch.cuda.max_memory_reserved(device) / 1024**3

    # Leak assertion (Fix #3): one more full pass must add no new graphs.
    ug_before = read_compile_stats()["unique_graphs"]
    for b in buckets:
        _train_step(model, loss_wrapper, optimizer, _make_batch(b, vocab_size, device), cfg)
    torch.cuda.synchronize(device)
    plateaued = read_compile_stats()["unique_graphs"] == ug_before

    # Timed steady-state: shuffled round-robin (per-bucket timing + allocator jump stress); skip any
    # step that recompiles (would poison the mean). Peak reset here → STEADY-STATE peak (excludes the
    # one-off autotune transient) = the number the max-batch budget cares about.
    torch.cuda.reset_peak_memory_stats(device)
    per_bucket = {i: [] for i in range(len(buckets))}
    for _ in range(TIMED_ROUNDS):
        order = list(range(len(buckets)))
        random.shuffle(order)
        for i in order:
            batch = _make_batch(buckets[i], vocab_size, device)
            ug0 = read_compile_stats()["unique_graphs"]
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _train_step(model, loss_wrapper, optimizer, batch, cfg)
            torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if read_compile_stats()["unique_graphs"] > ug0:
                continue
            per_bucket[i].append(dt_ms)
    steady_peak = torch.cuda.max_memory_reserved(device) / 1024**3
    steady_alloc = torch.cuda.max_memory_allocated(device) / 1024**3

    all_dt = [dt for v in per_bucket.values() for dt in v]
    stats = read_compile_stats()
    result = {
        "label": label, "dynamic": dynamic, "mode": mode, "oom": False,
        "mean_ms": float(np.mean(all_dt)) if all_dt else float("nan"),
        "p95_ms": float(np.percentile(all_dt, 95)) if all_dt else float("nan"),
        "per_bucket_ms": {i: float(np.mean(v)) for i, v in per_bucket.items() if v},
        "warmup_s": warmup_s, "warmup_peak_gb": warmup_peak,
        "steady_peak_gb": steady_peak, "steady_alloc_gb": steady_alloc,
        "unique_graphs": stats["unique_graphs"], "graph_breaks_total": stats["graph_breaks_total"],
        "break_reasons": stats["break_reasons"], "plateaued": plateaued,
    }
    del model, optimizer, loss_wrapper
    torch.cuda.empty_cache()
    return result


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def stress_test(cfg: DictConfig):
    log_file = PROJECT_ROOT / "logs" / "benchmarks" / "stress_test_fragmentation.log"
    setup_file_logger(logger, log_file, mode="w", format_str="%(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required for the compile-mode benchmark.")
    device = cfg.setup.device

    vocab_path = cfg.dataset.token_vocabulary_path or (DATA_DIR / cfg.dataset.name / "token_vocabulary.json")
    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=False)
    vocab_size = tokenizer.token_vocabulary_size

    buckets = _enhanced_buckets(cfg)
    sweep = _select_sweep()
    logger.info(f"Compile-mode benchmark: {len(buckets)} buckets, {len(sweep)} mode(s), "
                f"{TIMED_ROUNDS} timed rounds. (Set COMPILE_SWEEP=auto|fast|all.)")

    results = []
    for label, dynamic, mode in sweep:
        logger.info(f"\n--- {label}  (dynamic={dynamic}, mode={mode}) ---")
        try:
            r = _bench_mode(label, dynamic, mode, cfg, buckets, vocab_size, device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.error(f"❌ OOM in mode '{label}' at the current batch budget.")
            logger.error(f"Peak reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
            logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
            torch.cuda.empty_cache()
            results.append({"label": label, "dynamic": dynamic, "mode": mode, "oom": True})
            continue
        results.append(r)
        logger.info(f"  mean {r['mean_ms']:.1f}ms (p95 {r['p95_ms']:.1f}) | steady peak "
                    f"{r['steady_peak_gb']:.1f}GB | graphs {r['unique_graphs']} | "
                    f"breaks {r['graph_breaks_total']} [{format_break_reasons(r['break_reasons'])}] | "
                    f"warmup {r['warmup_s']:.0f}s | plateau {'✅' if r['plateaued'] else '❌ LEAK'}")

    # --- comparison table ---
    logger.info("\n========== COMPILE-MODE A/B SUMMARY ==========")
    ok = [r for r in results if not r["oom"]]
    base = next((r["mean_ms"] for r in ok if r["dynamic"] is None and r["mode"] == "default"), None)
    for r in results:
        if r["oom"]:
            logger.info(f"{r['label']:<36} OOM at current batch budget")
            continue
        delta = f"{(base / r['mean_ms'] - 1) * 100:+.1f}% vs auto" if base and r["mean_ms"] == r["mean_ms"] else "  n/a"
        leak = "✅" if r["plateaued"] else "❌ LEAK"
        logger.info(f"{r['label']:<36} {r['mean_ms']:6.1f}ms  {delta:>14} | peak "
                    f"{r['steady_peak_gb']:5.1f}GB (warmup {r['warmup_peak_gb']:.1f}) | "
                    f"graphs {r['unique_graphs']:>3} | warmup {r['warmup_s']:4.0f}s | {leak}")
    logger.info("\nVRAM note: dynamic=False adds static graphs (peak ≈ unchanged); reduce-overhead and "
                "both max-autotune (cudagraph) modes reserve a static memory POOL per shape → higher "
                "STEADY peak. If the chosen mode's steady peak exceeds the current budget, RE-RUN "
                "find_max_batch_sizes before training (the buckets were sized against the old peak).")
    logger.info("==========================================\n")


if __name__ == "__main__":
    # Memory expansion → mitigate fragmentation across bucket-size jumps.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    stress_test()
