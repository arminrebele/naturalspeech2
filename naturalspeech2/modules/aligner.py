import torch
from torch import nn
from einops import rearrange, repeat

from naturalspeech2.modules.layers import Conv1D, RMSNorm
from naturalspeech2.ops.monotonic_align import maximum_path



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

        learned_scores = self.aligner_net(  # [B, F, P]
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

        with torch.no_grad():
            path_indices = maximum_path_indices(  # [B, F] long, padded frames clamped to 0
                posterior_label_logits,
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
        frame_mask,              # [B, F, 1]
        phoneme_encodings,       # [B, P, hidden_dim]
        phoneme_encodings_mask,  # [B, P, 1]
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
        # coarse for the downstream log_softmax + log_prior arithmetic. The .float()
        # casts alone are not enough: torch.bmm is an autocast op and would still run
        # in BF16 inside the training autocast region, so we explicitly disable autocast.
        with torch.autocast(device_type=audio_features.device.type, enabled=False):
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



@torch.compiler.disable
def maximum_path_indices(
    scores: torch.Tensor,                       # [B, F, P] FP32
    frame_lengths: torch.Tensor,                # [B] long — valid frames per item
    phoneme_encodings_lengths: torch.Tensor,    # [B] long — valid phonemes (acoustic axis)
) -> torch.Tensor:
    """
    Monotonic Viterbi (stay or move-by-1). Returns compact path indices [B, F] long.
    Padded frame indices are 0 (the kernel only writes 1s in the valid F-range, so
    `argmax` over an all-zero padded row returns index 0).

    Backed by the Glow-TTS `monotonic_align` Cython kernel (Kim et al., NeurIPS 2020;
    full citation chain in CLAUDE.md "Aligner Viterbi"). The Python `for f` loop this
    replaces was hostile to `torch.compile` — Inductor would either compile-time-blow-up
    unrolling the 2249-iter FX graph or recompile per bucket length, which is the
    actual load-bearing motivation. Secondary effect: per-batch CUDA launch overhead
    drops from a per-frame storm (~18k tiny launches, analytically ~145 ms at
    F≈2250) to a single CPU op plus one `.cpu()` move; exact wall-clock improvement
    to be measured on first GPU run.

    `@torch.compiler.disable` because the kernel runs CPU-side after a `.cpu()` move;
    Dynamo treats this as a graph break and compiles around it.

    Note: the DP is invariant to a per-frame additive constant (V8 / R8), so feeding
    `learned_scores + prior_logprobs` produces the same hard path as feeding the
    log-softmax-normalized form. Do not "normalize" the DP scores in a future refactor.

    Lengths are passed explicitly (rather than recovered from the mask) so the kernel
    contract doesn't silently assume an outer-product attention mask — see review
    in `.codex/plans/aligner_followups.md` Phase 6 post-review fixes.
    """
    # Kernel convention is [B, P, F]; ours is [B, F, P]. Transpose to match.
    value = rearrange(scores, 'b f p -> b p f').contiguous()
    return maximum_path(  # [B, F] long — argmax done CPU-side inside maximum_path
        value,
        phoneme_encodings_lengths.to(torch.int32),
        frame_lengths.to(torch.int32),
    )



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
