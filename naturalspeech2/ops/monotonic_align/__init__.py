"""Cython-accelerated monotonic alignment search for TTS aligners.

Vendored verbatim from Glow-TTS (Kim et al., NeurIPS 2020):
  https://github.com/jaywalnut310/glow-tts/tree/master/monotonic_align
MIT licensed; see LICENSE.glow_tts in this directory.

The kernel runs the same monotonic stay-or-move-by-1 Viterbi DP we used to
express as a Python `for f in range(F)` loop, but as a single C call with
OpenMP-parallelised batches (`prange(num_threads=B)` — explicitly overrides
any OMP_NUM_THREADS env var). Primary motivation: a 2249-iter Python loop is
hostile to `torch.compile` (Inductor would either compile-time-blow-up on the
unrolled FX graph or recompile per bucket length). Secondary: per-batch CUDA
launch overhead drops from a per-frame storm to a single CPU op plus one
`.cpu()` move; exact wall-clock improvement to be measured on first GPU run.
"""
import numpy as np
import torch


def maximum_path(
    value: torch.Tensor,    # [B, P, F] FP32, contiguous
    t_xs: torch.Tensor,     # [B] int32 — valid phonemes per item (acoustic axis)
    t_ys: torch.Tensor,     # [B] int32 — valid frames per item
) -> torch.Tensor:
    """Cython-optimised monotonic alignment search.

    Returns:
        path_indices: [B, F] int64, on value.device.
                      Padded frames (y >= t_ys[b]) are 0 — the kernel only
                      writes 1s in the valid F-range, so an all-zero column's
                      argmax returns 0 (matches the previous "padded-frames-
                      clamped-to-0" contract).

    The argmax over the [B, P, F] one-hot path runs on CPU before the back-
    transfer, so we move ~`B·F·8` bytes back to GPU instead of the full
    `B·P·F·4` bytes. At B=8, P=80, F=2250 that's ~140 KB instead of ~5.8 MB.

    The kernel only iterates valid `(x < t_xs[b], y < t_ys[b])` cells and
    never reads padded positions, so we don't need to pre-zero `value` at
    padded positions — whatever's there (e.g. `-1e9` mask floor) is invisible
    to the DP.

    Imported lazily so MacBook-side code editing / static analysis works
    without the `.so` (which is built only inside the Docker container).
    """
    try:
        from naturalspeech2.ops.monotonic_align.core import maximum_path_c
    except ImportError as e:
        raise ImportError(
            "monotonic_align Cython extension is not built. "
            "If you're inside the Docker container, restart it (entrypoint.sh "
            "rebuilds when missing). Otherwise build manually:\n"
            "    cd naturalspeech2/ops/monotonic_align && "
            "python setup.py build_ext --inplace"
        ) from e

    device = value.device
    value_np = value.detach().cpu().numpy().astype(np.float32, copy=False)
    path_np = np.zeros_like(value_np, dtype=np.int32)
    t_xs_np = t_xs.detach().cpu().numpy().astype(np.int32, copy=False)
    t_ys_np = t_ys.detach().cpu().numpy().astype(np.int32, copy=False)
    maximum_path_c(path_np, value_np, t_xs_np, t_ys_np)
    indices_np = path_np.argmax(axis=1).astype(np.int64, copy=False)   # [B, F] CPU
    return torch.from_numpy(indices_np).to(device=device)
