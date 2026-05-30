from torch import nn
import torch.nn.functional as F

from einops import rearrange

from naturalspeech2.modules.layers import RMSNorm, MultiHeadCrossAttention, Conv1D
from naturalspeech2.utils.initialization import standard_init


class DurationPredictor(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            conv1d_layers: int = 30,
            conv1d_kernel_size: int = 3,
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
        self.to_duration = Conv1D(hidden_dim, 1, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard N(0, 0.02) base, then Fixup-style zero-init of every residual-output
        # projection (convs[i] feeds the conv residual; attns[i].to_out feeds the cross-attn
        # residual) and the final prediction head (to_duration). All 40 residual writes share
        # the same stream — every block is identity-at-init.
        standard_init(self)
        for conv in self.convs:
            nn.init.zeros_(conv.conv1d.weight)
        for attn in self.attns:
            nn.init.zeros_(attn.to_out.weight)
        nn.init.zeros_(self.to_duration.conv1d.weight)

    def forward(
            self,
            phoneme_encodings,       # [B, P, hidden_dim]
            phoneme_encodings_mask,  # [B, P, 1]            | dtyppe = bool (True/False)
            prompt_encodings,        # [B, Fp, hidden_dim]
            prompt_encodings_mask,   # [B, Fp, 1]
    ):
        x = phoneme_encodings
        mask = phoneme_encodings_mask.to(x.dtype)

        for group_idx in range(len(self.attns)):            # loop over 10 groups. Each group = 3 conv blocks + 1 attention block
            for conv_idx in range(self.conv_per_group):     # loop over 3 conv blocks within the group
                layer_idx = group_idx * self.conv_per_group + conv_idx
                residual = x
                x = self.conv_norms[layer_idx](x)
                x = self.convs[layer_idx](x, phoneme_encodings_mask)
                x = F.silu(x)
                x = residual + self.conv_dropout(x)
                x = x * mask

            # cross-attention: Q = phoneme path, K/V = prompt_encodings
            residual = x
            attn_out = self.attns[group_idx](
                self.attn_norms[group_idx](x),
                prompt_encodings,
                prompt_encodings_mask,
            )
            x = residual + self.attn_out_dropout(attn_out)
            x = x * mask

        x = self.final_norm(x)
        x = self.to_duration(x, phoneme_encodings_mask)  # [B, P, 1]
        x = rearrange(x * phoneme_encodings_mask.to(x.dtype), 'b p 1 -> b p')  # [B, P]
        return x
