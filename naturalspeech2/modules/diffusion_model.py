import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from naturalspeech2.modules.layers import RMSNorm, MultiHeadCrossAttention, Conv1D


class TimestepEmbedding(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            time_dim: int = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        half_dim = time_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32)
            / (half_dim - 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
            self,
            t,  # [B]
    ):
        angles = rearrange(t.float(), 'b -> b 1') * rearrange(self.frequencies, 'd -> 1 d')
        timestep_vector = torch.cat([angles.sin(), angles.cos()], dim=-1) # [B, time_dim]
        timestep_embedding = self.mlp(timestep_vector) # [B, D]
        return timestep_embedding


class _WaveNetBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            filter_size: int = 1024,
            kernel_size: int = 3,
            dilation: int = 2,
            attention_heads: int = 8,
            attn_weights_dropout: float = 0.2,
            attn_out_dropout: float = 0.2,
            gate_dropout: float = 0.2,
    ):
        super().__init__()

        self.norm = RMSNorm(hidden_dim)
        self.timestep_projection = nn.Linear(hidden_dim, hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, filter_size)
        self.dilated_conv = Conv1D(
            hidden_dim,
            filter_size,
            kernel_size,
            dilation=dilation,
        )
        self.cross_attention = MultiHeadCrossAttention(
            hidden_dim,
            attention_heads,
            attn_weights_dropout=attn_weights_dropout,
        )
        self.film_projection = nn.Linear(hidden_dim, filter_size * 2)
        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)

        self.gate_dropout = nn.Dropout(gate_dropout)
        self.attn_out_dropout = nn.Dropout(attn_out_dropout)

    def forward(
            self,
            h,                      # [B, F, D]
            timestep_embedding,     # [B, D]
            condition,              # [B, F, D]
            prompt_summary_tokens,  # [B, 32, D]
            h_mask,                 # [B, F, 1] bool
    ):
        float_h_mask = h_mask.to(h.dtype)
        residual = h

        h_norm = self.norm(h)
        h_t = h_norm + rearrange(self.timestep_projection(timestep_embedding), 'b d -> b 1 d')
        h_t = h_t * float_h_mask

        conv_out = self.dilated_conv(h_t, h_mask) + self.condition_projection(condition)
        conv_out = conv_out * float_h_mask  # [B, F, 1024]

        attn_out = self.cross_attention(   
            h_t,
            prompt_summary_tokens,
            kv_mask=None,
        )
        attn_out = self.attn_out_dropout(attn_out) # [B, F, D=512]
        film = self.film_projection(attn_out)      # [B, F, 2048]
        scale, bias = film.chunk(2, dim=-1)        # each [B, F, 1024]
        film_out = conv_out * (1.0 + scale) + bias
        film_out = film_out * float_h_mask         # [B, F, 1024]

        h_filter, h_gate = film_out.chunk(2, dim=-1)
        gated = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        gated = gated * float_h_mask

        skip = gated                                                         # clean, no dropout
        wavenet_block_out = (residual + self.gate_dropout(gated)) * float_h_mask   # [B, F, D]

        return wavenet_block_out, skip


class DiffusionModel(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            hidden_dim: int = 512,
            wavenet_layers: int = 40,
            wavenet_kernel_size: int = 3,
            wavenet_dilation: int = 2,
            wavenet_filter_size: int = 1024,
            attention_heads: int = 8,
            query_tokens: int = 32,
            attn_weights_dropout: float = 0.2,
            attn_out_dropout: float = 0.2,
            wavenet_attn_weights_dropout: float = 0.2,
            wavenet_attn_out_dropout: float = 0.2,
            wavenet_gate_dropout: float = 0.2,
            beta_min: float = 0.05,
            beta_max: float = 20.0,
            sampling_steps: int = 150,
            sampling_temperature: float = 1.44,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sampling_steps = sampling_steps
        self.sampling_temperature = sampling_temperature
        self.timestep_eps = 1e-3

        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim=hidden_dim)

        self.query_tokens = nn.Parameter(torch.randn(1, query_tokens, hidden_dim) * 0.02)
        self.prompt_encodings_norm = RMSNorm(hidden_dim)
        self.attn = MultiHeadCrossAttention(
            hidden_dim,
            attention_heads,
            attn_weights_dropout=attn_weights_dropout,
        )
        self.attn_out_dropout = nn.Dropout(attn_out_dropout)

        self.wavenet_blocks = nn.ModuleList([
            _WaveNetBlock(
                hidden_dim=hidden_dim,
                filter_size=wavenet_filter_size,
                kernel_size=wavenet_kernel_size,
                dilation=wavenet_dilation,
                attention_heads=attention_heads,
                attn_weights_dropout=wavenet_attn_weights_dropout,
                attn_out_dropout=wavenet_attn_out_dropout,
                gate_dropout=wavenet_gate_dropout,
            )
            for _ in range(wavenet_layers)
        ])

        self.output_projection_1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection_2 = nn.Linear(hidden_dim, latent_dim)

    def forward(
            self,
            target_latents,                 # [B, Ft, latent_dim]
            c,                              # [B, Ft, D]
            prompt_encodings,               # [B, Fp, D]
            target_latents_mask,            # [B, Ft, 1] bool
            prompt_encodings_mask,          # [B, Fp, 1] bool
    ):
        batch_size = target_latents.shape[0]
        t = (         #  t ∈ [ε, 1−ε], time_step_eps = 0.01 prevents exact 0 or 1 which can cause issues in the noise schedule math
            torch.rand(batch_size, device=target_latents.device)
            * (1.0 - 2.0 * self.timestep_eps)
            + self.timestep_eps
        )

        z_t, _ = self._forward_diffusion(target_latents, t)
        prompt_summary_tokens = self._compute_prompt_summary_tokens(
            prompt_encodings,
            prompt_encodings_mask,
        )
        z0_hat = self._predict_z0(
            z_t,
            t,
            c,
            prompt_summary_tokens,
            target_latents_mask,
        )

        diff_sq = (z0_hat.float() - target_latents.float()) ** 2
        loss_mask = target_latents_mask.to(diff_sq.dtype)
        valid_scalars = loss_mask.sum().clamp(min=1.0) * self.latent_dim
        return (diff_sq * loss_mask).sum() / valid_scalars

    def _compute_prompt_summary_tokens(
            self,
            prompt_encodings,  # [B, Fp, hidden_dim]
            prompt_encodings_mask,       # [B, Fp, 1] bool
    ):
        query_tokens = repeat(
            self.query_tokens,
            '1 m d -> b m d',
            b=prompt_encodings.shape[0],
        )
        prompt_summary_tokens = self.attn(
            query_tokens,
            self.prompt_encodings_norm(prompt_encodings),
            prompt_encodings_mask,
        )
        return self.attn_out_dropout(prompt_summary_tokens)

    def _predict_z0(
            self,
            z_t,                    # [B, Ft, latent_dim]
            t,                      # [B]
            c,                      # [B, Ft, D]
            prompt_summary_tokens,  # [B, query_tokens, hidden_dim]
            target_latents_mask,            # [B, Ft, 1] bool
    ):
        h = self.input_projection(z_t)
        h = h * target_latents_mask.to(h.dtype)
        timestep_embedding = self.timestep_embedding(t)

        skip_sum = 0
        for wavenet_block in self.wavenet_blocks:
            h, skip = wavenet_block(
                h,
                timestep_embedding,
                c,
                prompt_summary_tokens,
                target_latents_mask,
            )
            skip_sum = skip_sum + skip

        skip_avg = skip_sum / math.sqrt(len(self.wavenet_blocks))
        out = F.silu(skip_avg)
        out = self.output_projection_1(out)
        out = F.silu(out)
        out = self.output_projection_2(out)
        return out * target_latents_mask.to(out.dtype)

    def _forward_diffusion(
            self,
            z_0,  # [B, Ft, latent_dim]
            t,    # [B]
    ):
        # z_t = √α̅(t) · z_0 + √(1 − α̅(t)) · ε,    ε ~ N(0, I)
        epsilon = torch.randn_like(z_0, dtype=torch.float32)
        alpha_bar = rearrange(self._alpha_bar(t), 'b -> b 1 1')
        z_t = alpha_bar.sqrt() * z_0.float() + (1.0 - alpha_bar).sqrt() * epsilon
        return z_t, epsilon

    def _alpha_bar(
            self,
            t,  # [] or [B]
    ):
        # ∫₀ᵗ β(s) ds = t · β_min + ½ t² (β_max − β_min)
        # α̅(t)       = exp(−∫₀ᵗ β(s) ds)
        t = t.float()
        integral_beta = t * self.beta_min + 0.5 * t.square() * (self.beta_max - self.beta_min)
        return torch.exp(-integral_beta)

    def _beta(
            self,
            t,  # [] or [B]
    ):
        # β(t) = β_min + t (β_max − β_min)
        return self.beta_min + t.float() * (self.beta_max - self.beta_min)
