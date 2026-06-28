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


def _reason_first_line(reason: str) -> str:
    """Dynamo reason keys are multi-line blobs (Explanation/Hint/docs URL); the first line is the stable
    identity. Matching on it keeps reason-set logic robust to the variable debug-context tail."""
    return reason.split("\n", 1)[0].strip()


def format_break_reasons(break_reasons: dict) -> str:
    """One-line 'N× reason; ...' breakdown, descending by count — first line of each reason only,
    else this "one-liner" spans ~40 lines. Full per-break detail via TORCH_LOGS=graph_breaks."""
    if not break_reasons:
        return "(none)"
    return "; ".join(f"{n}× {_reason_first_line(r)}" for r, n in sorted(break_reasons.items(), key=lambda kv: -kv[1]))


# The intended structural graph breaks for this model, matched on each reason's first line. They appear
# INCREMENTALLY over the first ~hundreds of steps (a reason only registers once its code path + shape
# first executes), so the tripwire compares against THIS fixed baseline, not the previous step's set —
# else an expected reason showing up late false-fires. Anything outside this set is a real regression.
EXPECTED_BREAK_REASON_PREFIXES = frozenset({
    "Attempted to call function marked as skipped",         # torchaudio MelSpectrogram torch.jit.isinstance
    "Skip calling `torch.compiler.disable()`d function",    # intended @torch.compiler.disable site (encodec / aligner)
    "Skip inlining `torch.compiler.disable()`d function",   # intended @torch.compiler.disable site (encodec / aligner)
    "Dynamic shape operator",                               # aligner CTC forward-sum: data-dependent output shape
    "Operator does not support running with fake tensors",  # aligner CTC: aten._use_cudnn_ctc_loss fake-tensor probe
})


def unexpected_break_reasons(break_reasons: dict) -> set:
    """First lines of any graph-break reasons OUTSIDE the intended structural baseline — i.e. genuine
    regressions / new inefficiencies. Empty in a healthy run regardless of when each baseline reason
    first appeared."""
    return {_reason_first_line(r) for r in break_reasons} - EXPECTED_BREAK_REASON_PREFIXES
