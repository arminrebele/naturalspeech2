import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from naturalspeech2.modules.layers import RMSNorm, MultiHeadCrossAttention, Conv1D


class PitchPredictor(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            conv1d_layers: int = 30,
            conv1d_kernel_size: int = 5,
            attention_layers: int = 10,
            attention_heads: int = 8,
            conv_dropout: float = 0.5,
            attn_weights_dropout: float = 0.5,
            attn_out_dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_per_group = conv1d_layers // attention_layers  # 3
        self.conv_norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(conv1d_layers)])
        self.convs = nn.ModuleList([Conv1D(hidden_dim, hidden_dim, conv1d_kernel_size) for _ in range(conv1d_layers)])
        self.attn_norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(attention_layers)])
        self.attns = nn.ModuleList([
            MultiHeadCrossAttention(hidden_dim, attention_heads, attn_weights_dropout)
            for _ in range(attention_layers)
        ])
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.attn_out_dropout = nn.Dropout(attn_out_dropout)
        self.final_norm = RMSNorm(hidden_dim)
        self.to_pitch = Conv1D(hidden_dim, 1, 1)

    def forward(
            self,
            expanded_phoneme_encodings,       # [B, F, hidden_dim]
            expanded_phoneme_encodings_mask,  # [B, F, 1] | bool
            prompt_encodings,                 # [B, Fp, hidden_dim]
            prompt_encodings_mask,            # [B, Fp, 1]
    ):
        x = expanded_phoneme_encodings
        mask = expanded_phoneme_encodings_mask.to(x.dtype)

        for group_idx in range(len(self.attns)):            # 10 groups: 3 conv blocks + 1 cross-attn each
            for conv_idx in range(self.conv_per_group):
                layer_idx = group_idx * self.conv_per_group + conv_idx
                residual = x
                x = self.conv_norms[layer_idx](x)
                x = self.convs[layer_idx](x, expanded_phoneme_encodings_mask)
                x = F.silu(x)
                x = residual + self.conv_dropout(x)
                x = x * mask

            residual = x
            attn_out = self.attns[group_idx](
                self.attn_norms[group_idx](x),
                prompt_encodings,
                prompt_encodings_mask,
            )
            x = residual + self.attn_out_dropout(attn_out)
            x = x * mask

        x = self.final_norm(x)
        x = self.to_pitch(x, expanded_phoneme_encodings_mask)  # [B, F, 1]
        x = rearrange(x * expanded_phoneme_encodings_mask.to(x.dtype), 'b f 1 -> b f')  # [B, F]
        return x
