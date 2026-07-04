"""EMA of model weights for diffusion sampling.

Karras-style (EDM Appendix B.4): halflife in #training-examples w/ linear rampup
→ per-step decay derived at update time, stable across batch-size/grad-accum.
"""

from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """Karras-style EMA: halflife-in-examples + rampup.

    Tracks only trainable params (requires_grad) — buffers skipped (not in
    named_parameters). Shadow FP32 (.float() at update is no-op if param already FP32).
    """

    def __init__(self, model: nn.Module, halflife_kimg: float = 50.0):
        self.halflife_kimg = halflife_kimg
        self.shadow: dict[str, torch.Tensor] = {}
        self._stored: dict[str, torch.Tensor] | None = None
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone().float()

    def _effective_decay(self, batch_size: int, cur_kimg: float) -> float:
        effective_halflife_kimg = min(cur_kimg, self.halflife_kimg)
        if effective_halflife_kimg <= 0.0:
            return 0.0
        return 0.5 ** (batch_size / (effective_halflife_kimg * 1000.0))

    @torch.no_grad()
    def update(self, model: nn.Module, batch_size: int, cur_kimg: float) -> None:
        decay = self._effective_decay(batch_size, cur_kimg)
        shadows, params = [], []
        for name, p in model.named_parameters():
            if name in self.shadow:
                shadows.append(self.shadow[name])
                params.append(p.data.float())
        # foreach: same elementwise ops as per-tensor mul_/add_, batched into a few
        # multi-tensor kernels instead of 2 launches per parameter.
        if decay == 0.0:
            torch._foreach_copy_(shadows, params)
        else:
            torch._foreach_mul_(shadows, decay)
            torch._foreach_add_(shadows, params, alpha=1.0 - decay)

    @contextmanager
    def swap_in(self, model: nn.Module):
        if self._stored is not None:
            raise RuntimeError("EMA.swap_in is not reentrant")
        self._stored = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if name in self.shadow
        }
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.dtype))
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in self.shadow:
                    p.data.copy_(self._stored[name])
            self._stored = None

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "halflife_kimg": self.halflife_kimg}

    def load_state_dict(self, state: dict) -> None:
        # Move loaded shadow onto the live shadow's device. Checkpoints load
        # map_location="cpu" → without this, the first update()'s in-place add_
        # vs on-GPU params raises cross-device. (Mirrors Optimizer.load_state_dict.)
        device = next(iter(self.shadow.values())).device
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}
        self.halflife_kimg = state["halflife_kimg"]

    @property
    def num_parameters(self) -> int:
        return sum(t.numel() for t in self.shadow.values())
