import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from naturalspeech2.modules import RMSNorm, MultiHeadCrossAttention, Conv1D


# ausgabe maskieren: out = out.masked_fill(~mask, 0.0)
# duration predictor forward: nach jedem residul maskeiren: x = (x + layer_out) * phoneme_encodings_mask.to(x.dtype)
# vor dem final to_duration auch: x = (x + layer_out) * phoneme_encodings_mask.to(x.dtype)
# conv1d residual connections!

class DurationPredictor(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            conv1d_layers: int = 30,
            conv1d_kernel_size: int = 3,
            attention_layers: int = 10,
            attention_heads: int = 8,
            dropout: float = 0.5,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.final_norm = RMSNorm(hidden_dim)

        self.convs = nn.ModuleList([
            Conv1D(hidden_dim, hidden_dim, conv1d_kernel_size)
            for _ in range(conv1d_layers)
        ])

        self.attns = nn.ModuleList([
            MultiHeadCrossAttention(hidden_dim, attention_heads, dropout)
            for _ in range(attention_layers)
        ])

        self.to_duration = nn.Linear(hidden_dim, 1)

    def forward(
            self,
            phoneme_encodings,      # [B, P, hidden_dim]
            phoneme_encodings_mask, # [B, P, 1]
            prompt_encodings,       # [B, F, hidden_dim]
            prompt_encodings_mask   # [B, F, 1]
    ):
        # TODO: implement
        pass
    