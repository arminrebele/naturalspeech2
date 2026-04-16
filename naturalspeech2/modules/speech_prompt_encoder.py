import torch
from torch import nn

from naturalspeech2.modules.layers import TransformerEncoderLayer, RMSNorm, Conv1D

class SpeechPromptEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            latent_dim: int = 128,
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
        self.input_projection = Conv1D(latent_dim, hidden_dim, kernel_size=1)

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

    def forward(
            self,
            prompt_latents: torch.Tensor,         # [B, F, latent_dim]
            prompt_latents_mask: torch.Tensor,    # [B, F, 1] bool
            prompt_latents_lengths: torch.Tensor,
    ):
        x = self.input_projection(prompt_latents, prompt_latents_mask)  # [B, F, hidden_dim]

        for layer in self.transformer_layers:
            x = layer(x, prompt_latents_mask)

        x = self.final_norm(x)
        x = x * prompt_latents_mask.to(x.dtype)

        return x  # [B, F, hidden_dim]
