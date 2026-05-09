import torch
from torch import nn
from einops import rearrange, repeat

from naturalspeech2.modules.layers import Conv1D, RMSNorm



class Aligner(nn.Module):
    def __init__(
        self,
        audio_dim: int = 80,
        hidden_dim: int = 512,
        attn_channels: int = 80,
        temperature: float = 0.0005,
        prior_w: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prior_w = prior_w

        self.aligner_net = AlignerNet(
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            attn_channels=attn_channels,
            temperature=temperature,
            dropout=dropout,
        )

    def forward(
        self,
        audio_encodings,            # [B, F, audio_dim]
        frame_mask,                 # [B, F, 1] bool
        frame_lengths,              # [B] long
        phoneme_encodings,          # [B, P, hidden_dim]
        phoneme_encodings_mask,     # [B, P, 1] bool
        phoneme_encodings_lengths,  # [B] long
    ):
        B, F, _ = audio_encodings.shape
        P = phoneme_encodings.shape[1]
        device = audio_encodings.device

        learned_scores = self.aligner_net(  # FP32 [B, F, P]
            audio_encodings,
            frame_mask,
            phoneme_encodings,
            phoneme_encodings_mask,
        )

        # Mask invalid phoneme columns before softmax (large negative, not -inf,
        # so the FP32 log_softmax stays well-defined on rows that have at least
        # one valid column).
        mask_value = -torch.finfo(learned_scores.dtype).max
        phoneme_col_mask = rearrange(phoneme_encodings_mask.bool(), 'b p 1 -> b 1 p')  # [B, 1, P]
        learned_scores = learned_scores.masked_fill(~phoneme_col_mask, mask_value)

        prior_logprobs = compute_beta_binomial_prior(  # FP32 [B, F, P]
            frame_lengths,
            phoneme_encodings_lengths,
            frames_max=F,
            phoneme_encodings_max=P,
            w=self.prior_w,
        )

        # Decision 3: a single posterior-label object feeds CTC, Viterbi, and bin loss.
        # Form is `learned_scores + prior` (not `log_softmax(learned_scores) + prior`)
        # because the per-frame log-softmax constant is irrelevant for Viterbi (V8) and
        # would otherwise cost an extra normalization for CTC.
        posterior_label_logits = learned_scores + prior_logprobs  # [B, F, P] FP32
        posterior_label_logprobs = posterior_label_logits.log_softmax(dim=-1)  # [B, F, P] FP32

        attn_mask = frame_mask & phoneme_col_mask  # [B, F, P]

        with torch.no_grad():
            path_indices = maximum_path_indices(  # [B, F] long, padded frames clamped to 0
                posterior_label_logits,
                attn_mask,
                frame_lengths,
                phoneme_encodings_lengths,
            )

            frame_valid = rearrange(frame_mask, 'b f 1 -> b f').long()  # [B, F]
            durations = torch.zeros(B, P, device=device, dtype=torch.long).scatter_add_(
                dim=1,
                index=path_indices,
                src=frame_valid,
            )  # [B, P]

        return (
            durations,                  # [B, P] long
            path_indices,               # [B, F] long
            posterior_label_logprobs,   # [B, F, P] FP32
            posterior_label_logits,     # [B, F, P] FP32  -> ForwardSumLoss
        )



class AlignerNet(nn.Module):
    def __init__(
        self,
        audio_dim: int = 80,
        hidden_dim: int = 512,
        attn_channels: int = 80,
        temperature: float = 0.0005,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        self.audio_norm = RMSNorm(audio_dim)
        self.phoneme_norm = RMSNorm(hidden_dim)

        self.audio_conv1 = Conv1D(audio_dim, audio_dim * 2, kernel_size=3, bias=True)
        self.audio_conv2 = Conv1D(audio_dim * 2, audio_dim, kernel_size=3, bias=True)
        self.audio_proj = Conv1D(audio_dim, attn_channels, kernel_size=1, bias=True)

        self.phoneme_conv1 = Conv1D(hidden_dim, hidden_dim * 2, kernel_size=3, bias=True)
        self.phoneme_conv2 = Conv1D(hidden_dim * 2, hidden_dim, kernel_size=3, bias=True)
        self.phoneme_proj = Conv1D(hidden_dim, attn_channels, kernel_size=1, bias=True)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def _encode(
        self,
        x,            # [B, T, D]
        mask,         # [B, T, 1] bool
        norm,
        conv1,
        conv2,
        proj,
    ):
        m = mask.to(x.dtype)
        x = norm(x) * m

        # Conv1D pre-masks; we re-mask after each post-conv activation because
        # conv bias produces nonzero values at padded positions.
        x = conv1(x, mask)
        x = self.act(x) * m
        x = self.dropout(x)

        x = conv2(x, mask)
        x = self.act(x) * m
        x = self.dropout(x)

        x = proj(x, mask) * m
        return x

    def forward(
        self,
        audio_encodings,         # [B, F, audio_dim]
        frame_mask,              # [B, F, 1] bool
        phoneme_encodings,       # [B, P, hidden_dim]
        phoneme_encodings_mask,  # [B, P, 1] bool
    ):
        audio_features = self._encode(
            audio_encodings, frame_mask,
            self.audio_norm, self.audio_conv1, self.audio_conv2, self.audio_proj,
        )  # [B, F, attn_channels]

        phoneme_features = self._encode(
            phoneme_encodings, phoneme_encodings_mask,
            self.phoneme_norm, self.phoneme_conv1, self.phoneme_conv2, self.phoneme_proj,
        )  # [B, P, attn_channels]

        # Decision 2 / Decision 6: squared-L2 via GEMM in FP32 even under BF16 autocast.
        # `temperature * dist_sq` lands learned scores in roughly [-1, 0] for moderately
        # separated features — small enough that BF16 quantization noise (~0.01) is too
        # coarse for the downstream log_softmax + log_prior arithmetic.
        a = audio_features.float()
        p = phoneme_features.float()
        a_sq = a.square().sum(dim=-1, keepdim=True)                      # [B, F, 1]
        p_sq = rearrange(p.square().sum(dim=-1), 'b p -> b 1 p')         # [B, 1, P]
        cross = torch.bmm(a, rearrange(p, 'b p c -> b c p'))             # [B, F, P]
        dist_sq = (a_sq + p_sq - 2.0 * cross).clamp_min(0.0)
        learned_scores = -self.temperature * dist_sq                     # [B, F, P] FP32
        return learned_scores



_PRIOR_PAD = -1e9   # finite stand-in for log 0; exp(-1e9) underflows to 0 in fp32,
                    # but stays finite under batched CTC backward (which evaluates
                    # exp(log_probs) * grad over every cell, including padded ones —
                    # `-inf` there yields 0 * NaN = NaN gradients in PyTorch's CTC).


def compute_beta_binomial_prior(
    frame_lengths: torch.Tensor,              # [B] long
    phoneme_encodings_lengths: torch.Tensor,  # [B] long
    frames_max: int,
    phoneme_encodings_max: int,
    w: float = 1.0,
) -> torch.Tensor:
    """
    Paper: One TTS Alignment To Rule Them All (Equations 12, 13).
    Returns FP32 log-prior [B, F, P] regardless of autocast context — `lgamma`
    differences lose precision rapidly in BF16 for moderate F.
    """
    device = frame_lengths.device

    T = rearrange(frame_lengths.float(), 'b -> b 1 1')                              # [B, 1, 1]
    N = rearrange(phoneme_encodings_lengths.float(), 'b -> b 1 1') - 1.0            # [B, 1, 1]

    t_grid = rearrange(
        torch.arange(1, frames_max + 1, device=device, dtype=torch.float32),
        't -> 1 t 1',
    )                                                                               # [1, F, 1]
    k_grid = rearrange(
        torch.arange(0, phoneme_encodings_max, device=device, dtype=torch.float32),
        'k -> 1 1 k',
    )                                                                               # [1, 1, P]

    alpha = (w * t_grid).clamp_min(1e-5)                                            # [1, F, 1]
    beta = (w * (T - t_grid + 1.0)).clamp_min(1e-5)                                 # [B, F, 1]

    N_minus_k = (N - k_grid).clamp_min(0.0)                                         # [B, 1, P]

    # log C(N, k)
    log_binom_coeff = (
        torch.lgamma(N + 1.0)
        - torch.lgamma(k_grid + 1.0)
        - torch.lgamma(N_minus_k + 1.0)
    )

    nk_beta = (N - k_grid + beta).clamp_min(1e-5)
    # log B(k + α, N − k + β)
    log_beta_numerator = (
        torch.lgamma(k_grid + alpha)
        + torch.lgamma(nk_beta)
        - torch.lgamma(k_grid + alpha + nk_beta)
    )

    # log B(α, β)
    log_beta_denominator = (
        torch.lgamma(alpha)
        + torch.lgamma(beta)
        - torch.lgamma(alpha + beta)
    )

    log_prior = log_binom_coeff + log_beta_numerator - log_beta_denominator         # [B, F, P]

    t_mask = (t_grid <= T)                                                          # [B, F, 1]
    k_mask = (k_grid < N + 1.0)                                                     # [B, 1, P]
    mask = t_mask & k_mask                                                          # [B, F, P]

    return torch.where(mask, log_prior, log_prior.new_full((), _PRIOR_PAD))



def maximum_path_indices(
    scores: torch.Tensor,                       # [B, F, P] FP32
    attn_mask: torch.Tensor,                    # [B, F, P] bool
    frame_lengths: torch.Tensor,                # [B] long
    phoneme_encodings_lengths: torch.Tensor,    # [B] long
) -> torch.Tensor:
    """
    Monotonic Viterbi (stay or move-by-1). Returns compact path indices [B, F] long.
    Padded frame indices are clamped to 0 so downstream gather/scatter cannot fault.

    Note: the DP is invariant to a per-frame additive constant (V8 / R8), so feeding
    `learned_scores + prior_logprobs` produces the same hard path as feeding the
    log-softmax-normalized form. Do not "normalize" the DP scores in a future refactor.
    """
    device = scores.device
    dtype = scores.dtype
    B, F, P = scores.shape
    NEG_INF = float('-inf')

    dp = torch.full((B, F, P), NEG_INF, device=device, dtype=dtype)
    # Bool path tensor: True = move-by-1, False = stay (8x smaller than int64).
    path = torch.zeros((B, F, P), dtype=torch.bool, device=device)

    dp[:, 0, 0] = torch.where(
        attn_mask[:, 0, 0],
        scores[:, 0, 0],
        scores.new_full((), NEG_INF),
    )

    # Reused shifted buffer — avoids re-allocating [B, P] every frame.
    dp_move = torch.empty((B, P), device=device, dtype=dtype)

    for f in range(1, F):
        prev = dp[:, f - 1, :]                  # [B, P]

        # Shifted predecessor (+1 in phoneme direction). First column is -inf
        # because there is no phoneme to "move from" at p = 0.
        dp_move[:, 0] = NEG_INF
        dp_move[:, 1:] = prev[:, :-1]
        dp_stay = prev

        move_better = dp_move > dp_stay         # [B, P] bool
        path[:, f, :] = move_better
        dp_max = torch.where(move_better, dp_move, dp_stay)

        dp[:, f, :] = torch.where(
            attn_mask[:, f, :],
            scores[:, f, :] + dp_max,
            scores.new_full((), NEG_INF),
        )

    # --- Backtrack into compact [B, F] path indices ---
    # path[:, 0, :] is intentionally never written; at f=0 the optimal path is
    # always at p=0 and decision=False (we never move out of -inf at the boundary),
    # so backtrack has no decrement to apply at f=0.
    frame_lengths_idx = frame_lengths.long() - 1
    p_cur = phoneme_encodings_lengths.long() - 1   # [B], always >= 0
    batch_indices = torch.arange(B, device=device)

    path_indices = torch.zeros((B, F), dtype=torch.long, device=device)

    for f in reversed(range(F)):
        active = (f <= frame_lengths_idx)                  # [B] bool — within valid frame range
        # Padded frames: record 0; their duration contribution is zero because frame_valid masks them.
        idx_f = torch.where(active, p_cur, torch.zeros_like(p_cur))
        path_indices[:, f] = idx_f

        decision = path[batch_indices, f, p_cur]                    # [B] bool
        p_cur = p_cur - (decision & active).long()

    return path_indices



class ForwardSumLoss(nn.Module):
    """
    Paper: RAD-TTS (Appendix A.6). Forward-sum CTC over per-frame label posteriors.

    Targets are alignment positions 1..P, not phoneme vocabulary ids (Decision 5):
    repeated phonemes in the transcript remain distinct alignment targets, and the
    aligner only learns frame-to-position affinity.
    """

    def __init__(self, blank_logit: float = -1.0):
        super().__init__()
        self.blank_logit = blank_logit
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(
        self,
        posterior_label_logits: torch.Tensor,    # [B, F, P] FP32
        frame_lengths: torch.Tensor,             # [B] long
        phoneme_encodings_lengths: torch.Tensor, # [B] long
    ) -> torch.Tensor:
        B, _, P = posterior_label_logits.shape
        device = posterior_label_logits.device

        # Pre-softmax blank logit, calibrated against the learned-score range.
        logits_with_blank = torch.nn.functional.pad(
            posterior_label_logits,
            pad=(1, 0),
            value=self.blank_logit,
        )  # [B, F, P+1]

        log_probs = logits_with_blank.log_softmax(dim=-1)        # [B, F, P+1]
        log_probs = rearrange(log_probs, 'b f c -> f b c')

        targets = repeat(
            torch.arange(1, P + 1, device=device, dtype=torch.long),
            'p -> b p',
            b=B,
        )

        return self.ctc_loss(
            log_probs,
            targets,
            frame_lengths.long(),
            phoneme_encodings_lengths.long(),
        )



class BinLoss(nn.Module):
    """
    Per-step NLL of the posterior-label distribution along the Viterbi hard path.
    Consumes posterior log-probs (Decision 3), not prior-free learned log-probs.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        posterior_label_logprobs: torch.Tensor,  # [B, F, P] FP32
        path_indices: torch.Tensor,              # [B, F] long
        frame_mask: torch.Tensor,                # [B, F, 1] bool
    ) -> torch.Tensor:
        selected = torch.gather(
            posterior_label_logprobs,
            dim=2,
            index=rearrange(path_indices, 'b f -> b f 1'),
        ).squeeze(-1)                                            # [B, F]

        mask = rearrange(frame_mask, 'b f 1 -> b f').to(selected.dtype)
        nll = -(selected * mask).sum(dim=1)                      # [B]
        denom = mask.sum(dim=1).clamp_min(1.0)                   # [B]
        return (nll / denom).mean()
