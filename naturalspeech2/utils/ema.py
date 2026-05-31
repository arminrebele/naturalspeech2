"""Exponential moving average of model weights for diffusion sampling.

Karras-style EMA (EDM Appendix B.4): halflife specified in number of
training examples with a linear rampup, so per-step decay is derived at
update time and stays consistent across batch-size / grad-accum changes.
"""

from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """Karras-style EMA with halflife-in-examples + rampup.

    Tracks only trainable params (requires_grad=True) — same set the
    optimizer updates. Buffers are not in named_parameters() so they're
    skipped automatically. Shadow is FP32; the .float() at update time
    is a no-op when the live param is already FP32.
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
        one_minus_decay = 1.0 - decay
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            if decay == 0.0:
                self.shadow[name].copy_(p.data.float())
            else:
                self.shadow[name].mul_(decay).add_(
                    p.data.float(), alpha=one_minus_decay,
                )

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
        # Coerce the loaded shadow onto the device the live shadow already lives
        # on (set in __init__ from the on-device model). Checkpoints load with
        # map_location="cpu", so state["shadow"] is on CPU; without this move the
        # first update()'s in-place add_ against on-GPU params raises a
        # cross-device error. Mirrors torch.optim.Optimizer.load_state_dict,
        # which casts loaded state onto each param's device.
        device = next(iter(self.shadow.values())).device
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}
        self.halflife_kimg = state["halflife_kimg"]

    @property
    def num_parameters(self) -> int:
        return sum(t.numel() for t in self.shadow.values())
