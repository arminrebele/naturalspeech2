import torch
from torch import nn

from naturalspeech2.modules.layers import TransformerEncoderLayer, RMSNorm
from naturalspeech2.utils.initialization import standard_init

class PhonemeEncoder(nn.Module):
    def __init__(
            self,
            token_vocabulary_size: int = None,
            hidden_dim: int = 512,
            transformer_layers: int = 6,
            attention_heads: int = 8,
            conv1d_filter_size: int = 2048,
            conv1d_kernel_size: int = 9,
            conv_dropout: float = 0.2,
            attn_weights_dropout: float = 0.2,
            attn_out_dropout: float = 0.2,
            rope_base: float = 10000.0,
            rope_max_seq_len: int = 3000,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(token_vocabulary_size, hidden_dim, padding_idx=0)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim,
                attention_heads,
                conv1d_filter_size,
                conv1d_kernel_size,
                conv_dropout,
                attn_weights_dropout,
                attn_out_dropout,
                rope_base,
                rope_max_seq_len,
                )
            for _ in range(transformer_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard N(0, 0.02) base, then Fixup-style zero-init of the two residual-output
        # projections per TransformerEncoderLayer: the self-attention output projection
        # (multi_head_attention.to_out) and the FFN output projection (conv2). Each
        # transformer layer is identity-at-init → the 6-layer stack is identity-at-init.
        standard_init(self)
        for layer in self.transformer_layers:
            nn.init.zeros_(layer.multi_head_attention.to_out.weight)
            nn.init.zeros_(layer.conv2.conv1d.weight)

    def forward(
            self,
            phoneme_tokens: torch.Tensor,           # [B, P]
            phoneme_tokens_mask: torch.Tensor,      # [B, P, 1]
            phoneme_tokens_lengths: torch.Tensor,   # [B]
    ):
        phoneme_tokens_emb = self.token_embedding(phoneme_tokens) # [B, P, hidden_dim]

        for layer in self.transformer_layers:
            phoneme_tokens_emb = layer(phoneme_tokens_emb, phoneme_tokens_mask)
        
        phoneme_tokens_emb = self.final_norm(phoneme_tokens_emb)

        phoneme_tokens_emb = phoneme_tokens_emb * phoneme_tokens_mask.to(phoneme_tokens_emb.dtype)
        
        return phoneme_tokens_emb # [B, P, hidden_dim]
