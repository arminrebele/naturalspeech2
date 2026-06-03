"""Cython-accelerated monotonic alignment search for TTS aligners.

Vendored verbatim from Glow-TTS (Kim et al., NeurIPS 2020):
  https://github.com/jaywalnut310/glow-tts/tree/master/monotonic_align
MIT licensed; see LICENSE.glow_tts in this directory.

Monotonic stay-or-move-by-1 Viterbi DP as one C call with OpenMP-parallel batches
(prange(num_threads=B), overriding OMP_NUM_THREADS). Primary motivation: a 2249-iter
Python loop is hostile to torch.compile (Inductor blow-up or per-bucket recompiles).
Secondary: one CPU op + one .cpu() move instead of a storm of tiny CUDA kernels.
Cost: 0.55 ms typical (B=8,F=375,P=60), 5.4 ms worst-case bucket (B=8,F=2250,P=120).
"""
import numpy as np
import torch


def maximum_path(
    value: torch.Tensor,    # [B, P, F] FP32, contiguous
    t_xs: torch.Tensor,     # [B] int32 — valid phonemes per item (acoustic axis)
    t_ys: torch.Tensor,     # [B] int32 — valid frames per item
) -> torch.Tensor:
    """Cython-optimised monotonic alignment search.

    Returns path_indices [B, F] int64 on value.device. Padded frames (y ≥ t_ys[b]) are 0
    (kernel writes 1s only in the valid F-range → all-zero column's argmax = 0).

    argmax over the [B, P, F] one-hot runs on CPU before back-transfer → move ~B·F·8 bytes
    to GPU, not the full B·P·F·4 (~140 KB vs ~5.8 MB at B=8, P=80, F=2250).

    Kernel iterates only valid (x < t_xs[b], y < t_ys[b]) cells → no need to pre-zero padded
    positions (a -1e9 mask floor there is invisible to the DP).

    Imported lazily so editing / static analysis works without the compiled .so (built only in Docker).
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
