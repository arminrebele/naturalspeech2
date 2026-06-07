"""torch.compile knobs + Dynamo telemetry — shared by train.py, eval_daemon.py, benchmarks.

Two concerns, one home:
- `compile_kwargs`: turn `setup.compile` config into `torch.compile(**kwargs)`. Empty for defaults
  → the call is byte-identical to a bare `torch.compile(model)`.
- `read_compile_stats` / `format_break_reasons`: snapshot Dynamo's counters. CPU-side Python ints
  Dynamo maintains regardless of observation → reading is free (no GPU sync). `unique_graphs` is the
  leak signal (must PLATEAU after warmup); `n_break_reasons` is the intended-breaks tripwire.
"""
import torch
import torch._dynamo


def compile_kwargs(compile_cfg) -> dict:
    """`setup.compile` cfg → torch.compile kwargs. Empty dict ⇒ today's default behavior.
    dynamic: null=auto (static→symbolic promotion) / false=static-per-bucket / true=symbolic.
    mode: default | reduce-overhead | max-autotune-no-cudagraphs | max-autotune."""
    kwargs = {}
    if compile_cfg.dynamic is not None:
        kwargs["dynamic"] = bool(compile_cfg.dynamic)
    if compile_cfg.mode and compile_cfg.mode != "default":
        kwargs["mode"] = compile_cfg.mode
    return kwargs


def read_compile_stats() -> dict:
    """Snapshot Dynamo counters (free, CPU-only). unique_graphs = leak signal (plateaus once every
    bucket has compiled); n_break_reasons = intended-breaks tripwire (flat at the intended count)."""
    counters = torch._dynamo.utils.counters
    break_reasons = dict(counters.get("graph_break", {}))
    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "frames_total": counters["frames"]["total"],
        "graph_breaks_total": sum(break_reasons.values()),
        "n_break_reasons": len(break_reasons),
        "break_reasons": break_reasons,
        "cache_size_limit": torch._dynamo.config.cache_size_limit,
    }


def format_break_reasons(break_reasons: dict) -> str:
    """One-line 'N× reason; ...' breakdown, descending by count."""
    if not break_reasons:
        return "(none)"
    return "; ".join(f"{n}× {r}" for r, n in sorted(break_reasons.items(), key=lambda kv: -kv[1]))
