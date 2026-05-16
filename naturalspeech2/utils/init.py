from torch import nn

from naturalspeech2.modules.layers import RMSNorm


def standard_init(module: nn.Module) -> None:
    """N(0, 0.02) on Linear/Conv1d/Embedding weights; 0 on biases; 1 on RMSNorm. Walks submodules recursively."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)
