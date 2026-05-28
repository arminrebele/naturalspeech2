import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from naturalspeech2.modules.layers import RMSNorm, MultiHeadCrossAttention, Conv1D
from naturalspeech2.utils.initialization import standard_init


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
            bias=True,
        )
        self.cross_attention = MultiHeadCrossAttention(
            hidden_dim,
            attention_heads,
            attn_weights_dropout=attn_weights_dropout,
        )
        self.film_projection = nn.Linear(hidden_dim, filter_size * 2)
        # ReZero (Bachlechner et al. 2020): per-block learnable scalar, init=0. Multiplies
        # the gated branch before the residual add — block is identity-at-init regardless of
        # what the branch computes.
        self.alpha = nn.Parameter(torch.zeros(1))

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
        wavenet_block_out = (residual + self.alpha * self.gate_dropout(gated)) * float_h_mask   # [B, F, D]

        return wavenet_block_out, skip


class DiffusionModel(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            hidden_dim: int = 512,
            time_dim: int = 128,
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
            timestep_eps: float = 1e-3,
            min_snr_gamma: float = 5.0,         # min-SNR(γ) clip on the implicit α̅/σ² weight in score loss (Hang et al. 2023); see forward()
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sampling_steps = sampling_steps
        self.sampling_temperature = sampling_temperature
        self.timestep_eps = timestep_eps
        self.min_snr_gamma = min_snr_gamma

        self.input_projection = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.timestep_embedding = TimestepEmbedding(hidden_dim=hidden_dim, time_dim=time_dim)

        self.query_tokens = nn.Parameter(torch.randn(1, query_tokens, hidden_dim) * 0.02)
        self.prompt_encodings_norm = RMSNorm(hidden_dim)
        self.cross_attention = MultiHeadCrossAttention(
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

        self.output_projection_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection_2 = nn.Linear(hidden_dim, latent_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero the final ẑ₀ head (DiT-style: data_loss starts at σ²(z₀)) and FiLM in every
        # WaveNet block. ReZero α scalars are already zero from _WaveNetBlock.__init__.
        standard_init(self)
        nn.init.zeros_(self.output_projection_2.weight)
        for block in self.wavenet_blocks:
            nn.init.zeros_(block.film_projection.weight)
            nn.init.zeros_(block.film_projection.bias)

    def forward(
            self,
            target_latents,                 # [B, Ft, latent_dim]    in normalized space
            target_latents_mask,            # [B, Ft, 1] bool
            prompt_encodings,               # [B, Fp, D]
            prompt_encodings_mask,          # [B, Fp, 1] bool
            c,                              # [B, Ft, D]
            target_codebook_indices,        # [B, Ft, Q] | GT codebook indices per quantizer
            codebook_embeddings,            # [Q, K=1024, latent_dim=128]   | frozen Encodec codebook vectors (raw)
            latent_mean,                    # [latent_dim] | per-channel mean for unnormalizing ẑ₀ before CE-RVQ
            latent_std,                     # [latent_dim] | per-channel std
    ):
        batch_size = target_latents.shape[0]
        t = (         #  t ∈ [ε, 1−ε], time_step_eps = 0.01 prevents exact 0 or 1 which can cause issues in the noise schedule math
            torch.rand(batch_size, device=target_latents.device)
            * (1.0 - 2.0 * self.timestep_eps)
            + self.timestep_eps
        )

        z_t, epsilon = self._forward_diffusion(target_latents, t)       # [B, Ft, latent_dim]       noisy latents
        prompt_summary_tokens = self._compute_prompt_summary_tokens(
            prompt_encodings,
            prompt_encodings_mask,
        )
        z0_hat = self._predict_z0(
            z_t,
            target_latents_mask,
            t,
            c,
            prompt_summary_tokens,
        )

        # Data loss:     L_data = ‖ẑ₀ − z₀‖²,    masked mean over valid scalars.
        diff_sq = (z0_hat.float() - target_latents.float()) ** 2
        loss_mask = target_latents_mask.to(diff_sq.dtype)
        data_loss = (diff_sq * loss_mask).sum()

        # Score loss:
        #   ŝ(z_t, t)          = (√α̅(t) · ẑ₀ − z_t) / (1 − α̅(t))        (predicted score, derived from ẑ₀)
        #   ∇ log p_t(z_t|z₀)  = −ε / √(1 − α̅(t))                        (true conditional score)
        #   L_score            = ‖ŝ − ∇ log p_t‖²
        #
        # Algebraically: L_score = (α̅ / σ²) · ‖ẑ₀ − z₀‖²  where σ = 1 − α̅.
        # The implicit (α̅/σ²) weight blows up as t → 0 (~1.3k at t=0.05, ~2.8e8 at
        # t=1e-3), spiking gradients and biasing the loss toward low-t (easy) modes.
        #
        # Stabilization:
        #   - Min-SNR(γ) clip (Hang et al. 2023): cap the effective weight at γ via
        #     a per-sample factor min(γ·σ²/α̅, 1). With γ=5, low-t contribution is
        #     bounded; high-t (where σ²/α̅ ≥ 1) is unchanged.
        #   - Clamp (1 − α̅) at 1e-5 as NaN-guard at timestep_eps boundary.
        alpha_bar = rearrange(self._alpha_bar(t), 'b -> b 1 1')
        sigma = (1.0 - alpha_bar).clamp(min=1e-5)
        score_hat = (alpha_bar.sqrt() * z0_hat.float() - z_t) / sigma
        score_target = -epsilon / sigma.sqrt()

        min_snr_clip = (self.min_snr_gamma * sigma.pow(2) / alpha_bar).clamp(max=1.0)
        score_diff_sq = (score_hat - score_target) ** 2 * min_snr_clip
        score_loss = (score_diff_sq * loss_mask).sum()

        # CE-RVQ loss:
        #   Per quantizer j, score the partial residual ẑ₀ − Σᵢ<ⱼ eᵢ against every
        #   codebook entry via −L2 + softmax, and cross-entropy against the GT code
        #   index. Supervises the discrete decoding path that plain MSE misses.
        #
        #   Codebook embeddings live in raw Encodec space, so ẑ₀ must be unnormalized
        #   inside _ce_rvq_loss before residual computation.
        ce_rvq_loss = self._ce_rvq_loss(
            z0_hat,
            target_codebook_indices,
            codebook_embeddings,
            target_latents_mask,
            latent_mean,
            latent_std,
        )

        return {
            "data_loss": data_loss,
            "score_loss": score_loss,
            "ce_rvq_loss": ce_rvq_loss,
        }

    @torch.no_grad()
    def sample(
            self,
            condition,                  # [B, Ft, D]
            condition_mask,             # [B, Ft, 1] bool
            prompt_encodings,           # [B, Fp, D]
            prompt_encodings_mask,      # [B, Fp, 1] bool
            sampling_steps: int | None = None,
    ):
        # Probability-flow ODE reverse solve of the VP-SDE, Euler steps over [1, ε].
        #
        #   dz/dt   = -½β(t)·(√ᾱ(t)·ẑ₀ − ᾱ(t)·z_t) / (1 − ᾱ(t))
        #   z_{t-Δt} = z_t + Δt·½β(t)·(√ᾱ(t)·ẑ₀ − ᾱ(t)·z_t) / (1 − ᾱ(t))
        #
        # z_T ~ N(0, τ⁻¹·I) with τ = sampling_temperature (paper §4.3).
        n_steps = sampling_steps if sampling_steps is not None else self.sampling_steps

        B, Ft, _ = condition.shape
        device = condition.device
        float_mask = condition_mask.to(torch.float32)

        z = torch.randn(B, Ft, self.latent_dim, device=device, dtype=torch.float32)
        z = z / math.sqrt(self.sampling_temperature)
        z = z * float_mask

        prompt_summary_tokens = self._compute_prompt_summary_tokens(
            prompt_encodings,
            prompt_encodings_mask,
        )

        timesteps = torch.linspace(1.0, self.timestep_eps, n_steps + 1, device=device, dtype=torch.float32)

        for i in range(n_steps):
            t_scalar = timesteps[i]
            dt = t_scalar - timesteps[i + 1]
            t = t_scalar.expand(B)

            z0_hat = self._predict_z0(
                z,
                condition_mask,
                t,
                condition,
                prompt_summary_tokens,
            ).float()

            alpha_bar = rearrange(self._alpha_bar(t), 'b -> b 1 1')
            sqrt_alpha_bar = alpha_bar.sqrt()
            one_minus_alpha_bar = (1.0 - alpha_bar).clamp(min=1e-5)
            beta = rearrange(self._beta(t), 'b -> b 1 1')

            drift = 0.5 * beta * (sqrt_alpha_bar * z0_hat - alpha_bar * z) / one_minus_alpha_bar
            z = z + dt * drift
            z = z * float_mask

        return z

    def _compute_prompt_summary_tokens(
            self,
            prompt_encodings,           # [B, Fp, D]
            prompt_encodings_mask,      # [B, Fp, 1] bool
    ):
        query_tokens = repeat(
            self.query_tokens,
            '1 m d -> b m d',
            b=prompt_encodings.shape[0],
        )
        prompt_summary_tokens = self.cross_attention(
            query_tokens,
            self.prompt_encodings_norm(prompt_encodings),
            prompt_encodings_mask,
        )
        return self.attn_out_dropout(prompt_summary_tokens)

    def _predict_z0(
            self,
            z_t,                    # [B, Ft, latent_dim]
            target_latents_mask,    # [B, Ft, 1] bool
            t,                      # [B]
            c,                      # [B, Ft, D]
            prompt_summary_tokens,  # [B, query_tokens, D]
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

    def _ce_rvq_loss(
            self,
            z0_hat,                   # [B, Ft, latent_dim] | in normalized space
            target_codebook_indices,  # [B, Ft, Q]           | GT codebook indices per quantizer
            codebook_embeddings,      # [Q, K, latent_dim]   | raw Encodec codebook vectors
            target_latents_mask,      # [B, Ft, 1] bool
            latent_mean,              # [latent_dim] | per-channel mean for unnormalization
            latent_std,               # [latent_dim] | per-channel std
    ):
        # For each quantizer j:
        #   r_j      = ẑ₀ − Σᵢ<ⱼ eᵢ                 (partial residual, GT earlier codes — no error cascade)
        #   logit_k  = 2·r_j·Cⱼ[k] − ‖Cⱼ[k]‖²       (= −‖r−C[k]‖² + const; const drops under softmax)
        #   loss_j   = CE(softmax_k(logit), codes_j)
        # Single Python loop over Q=32 (static), vectorized over (B, Ft, K) per step.
        # Running cumsum avoids materializing a full [Q, B, Ft, latent_dim] residual tensor.
        # Unnormalize ẑ₀ to raw codebook space before residual computation — codebook
        # embeddings are raw Encodec vectors and the residual interpretation only holds there.
        z0_hat = z0_hat.float() * latent_std + latent_mean
        Q = target_codebook_indices.shape[2]
        codebooks = codebook_embeddings[:Q].float()                         # [Q, K, latent_dim]
        codebook_sq_norms = codebooks.pow(2).sum(dim=-1)                    # [Q, K]

        flat_mask = rearrange(target_latents_mask, 'b ft 1 -> (b ft)').float()
        num_valid = target_latents_mask.sum().float().clamp(min=1.0)

        running_cum = torch.zeros_like(z0_hat)                              # [B, Ft, latent_dim]   Σᵢ<ⱼ eᵢ
        total_ce = z0_hat.new_zeros(())

        for j in range(Q):
            gt_embed_j = codebooks[j][target_codebook_indices[:, :, j]]     # [B, Ft, latent_dim]
            residual_j = z0_hat - running_cum                               # [B, Ft, latent_dim]
            running_cum = running_cum + gt_embed_j

            logits = 2.0 * torch.einsum('btd,kd->btk', residual_j, codebooks[j])  # [B, Ft, K]
            logits = logits - codebook_sq_norms[j]

            logits_flat = rearrange(logits, 'b ft k -> (b ft) k')
            targets_flat = rearrange(target_codebook_indices[:, :, j], 'b ft -> (b ft)')
            ce_per_scalar = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            total_ce = total_ce + (ce_per_scalar * flat_mask).sum()

        return total_ce

    def _forward_diffusion(
            self,
            z_0,  # [B, Ft, latent_dim]     target latents
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
        # Standard VP-SDE convention: α̅(t) = exp(−∫₀ᵗ β(s) ds), with
        # ∫₀ᵗ β(s) ds = t · β_min + ½ t² (β_max − β_min). Forward marginal
        # z_t = √α̅·z₀ + √(1 − α̅)·ε then matches the paper (ρ(z₀,t) = √α̅·z₀,
        # Σ_t = 1 − α̅).
        t = t.float()
        integral_beta = t * self.beta_min + 0.5 * t.square() * (self.beta_max - self.beta_min)
        return torch.exp(-integral_beta)

    def _beta(
            self,
            t,  # [] or [B]
    ):
        # β(t) = β_min + t (β_max − β_min)
        return self.beta_min + t.float() * (self.beta_max - self.beta_min)
