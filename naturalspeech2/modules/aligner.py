import torch
from torch import nn
from einops import rearrange, repeat

from naturalspeech2.modules.layers import Conv1D, RMSNorm
from naturalspeech2.ops.monotonic_align import maximum_path
from naturalspeech2.utils.initialization import standard_init



class Aligner(nn.Module):
    def __init__(
        self,
        audio_dim: int = 80,
        hidden_dim: int = 512,
        attn_channels: int = 80,
        temperature: float = 5.0,
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
        frame_mask,                 # [B, F, 1]
        frame_lengths,              # [B] 
        phoneme_encodings,          # [B, P, hidden_dim]
        phoneme_encodings_mask,     # [B, P, 1]
        phoneme_encodings_lengths,  # [B]
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

        # Mask invalid phoneme cols pre-softmax with large-negative (not -inf) →
        # FP32 log_softmax stays defined on rows with ≥1 valid column.
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

        # Per-frame log_softmax of learned scores BEFORE adding the prior. ForwardSumLoss
        # softmaxes over [blank, P labels]; without this, label logits sit at an uncalibrated
        # scale (feature norm × temperature) → shifts blank-vs-label calibration. Viterbi/bin-loss
        # unaffected: log_softmax adds a per-frame constant, invariant under the DP.
        learned_label_logprobs = learned_scores.log_softmax(dim=-1)             # [B, F, P] FP32
        posterior_label_logits = learned_label_logprobs + prior_logprobs        # [B, F, P] FP32
        posterior_label_logprobs = posterior_label_logits.log_softmax(dim=-1)   # [B, F, P] FP32

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
        temperature: float = 5.0,
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

    def _init_weights(self) -> None:
        standard_init(self)

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

        # Conv1D pre-masks; re-mask after each activation (conv bias produces nonzero at padding).
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

        # L2-normalize features → cosine-similarity attention: dist_sq = 2·(1−cos) ∈ [0,4],
        # so learned_scores = −temperature·dist_sq ∈ [−2·temperature, 0] — range INDEPENDENT of
        # feature magnitude. Without it, softmax sharpness is hostage to feature norm (tiny under
        # N(0,0.02) init → near-uniform softmax → alignment never sharpens). temperature sets the
        # cosine scale directly (2·temperature on cos; ~10 here, cf. CLIP's learned ~14) and also
        # scales the gradient reaching the features.
        #
        # FP32 GEMM (autocast disabled): torch.bmm is an autocast op, so .float() alone wouldn't keep
        # the cross-term FP32 in the autocast region — cheap insurance for the log_softmax+prior math.
        with torch.autocast(device_type=audio_features.device.type, enabled=False):
            a = torch.nn.functional.normalize(audio_features.float(), dim=-1)   # unit ‖·‖ over channels
            p = torch.nn.functional.normalize(phoneme_features.float(), dim=-1)
            a_sq = a.square().sum(dim=-1, keepdim=True)                      # [B, F, 1] ≈ 1
            p_sq = rearrange(p.square().sum(dim=-1), 'b p -> b 1 p')         # [B, 1, P] ≈ 1
            cross = torch.bmm(a, rearrange(p, 'b p c -> b c p'))             # [B, F, P] = cos sim
            dist_sq = (a_sq + p_sq - 2.0 * cross).clamp_min(0.0)             # = 2·(1 − cos) ∈ [0, 4]
            learned_scores = -self.temperature * dist_sq                     # [B, F, P] FP32
        return learned_scores



_PRIOR_PAD = -1e9   # finite stand-in for log 0 (exp underflows to 0 in fp32) that stays
                    # finite under batched CTC backward — `-inf` would give 0*NaN=NaN grads.


def compute_beta_binomial_prior(
    frame_lengths: torch.Tensor,              # [B] long
    phoneme_encodings_lengths: torch.Tensor,  # [B] long
    frames_max: int,
    phoneme_encodings_max: int,
    w: float = 1.0,
) -> torch.Tensor:
    """Beta-binomial alignment log-prior (paper: One TTS Alignment To Rule Them All, Eq 12-13).
    Returns FP32 [B, F, P] regardless of autocast — lgamma diffs lose precision fast in BF16."""
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

    alpha = w * t_grid                                                              # [1, F, 1]
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
    """Monotonic Viterbi (stay or move-by-1) → compact path indices [B, F] long.
    Padded frame indices are 0 (kernel writes 1s only in the valid F-range, so argmax
    over an all-zero padded row returns 0).

    Backed by the Glow-TTS monotonic_align Cython kernel (Kim et al., NeurIPS 2020). The
    pure-Python `for f` loop it replaces was hostile to torch.compile (Inductor blow-up
    unrolling the ~2249-iter graph, or per-bucket recompiles) — the load-bearing motivation.
    Kernel cost: 0.55 ms typical (B=8,F=375,P=60), 5.4 ms worst-case bucket (B=8,F=2250,P=120).

    @torch.compiler.disable: kernel runs CPU-side after .cpu(); Dynamo graph-breaks around it.

    DP is invariant to a per-frame additive constant, so learned_scores + prior_logprobs gives
    the same hard path as the log-softmax-normalized form. Do NOT "normalize" the DP scores.

    Lengths passed explicitly (not recovered from a mask) so the kernel contract doesn't assume
    an outer-product attention mask.
    """
    # Kernel convention is [B, P, F]; ours is [B, F, P]. Transpose to match.
    value = rearrange(scores, 'b f p -> b p f').contiguous()
    return maximum_path(  # [B, F] long — argmax done CPU-side inside maximum_path
        value,
        phoneme_encodings_lengths.to(torch.int32),
        frame_lengths.to(torch.int32),
    )



class ForwardSumLoss(nn.Module):
    """Forward-sum CTC over per-frame label posteriors (paper: RAD-TTS, Appendix A.6).

    Targets are alignment positions 1..P, not phoneme vocab ids: repeated phonemes stay
    distinct targets; the aligner only learns frame-to-position affinity.
    """

    def __init__(self, blank_logit: float = -1.0):
        super().__init__()
        self.blank_logit = blank_logit
        # reduction='sum' (not 'mean'): 'mean' divides by phoneme count P → nats/phoneme, but
        # bin_loss and the rest are frame-normalized. Same units → loss_weights stay comparable.
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

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

        total_nll = self.ctc_loss(
            log_probs,
            targets,
            frame_lengths.long(),
            phoneme_encodings_lengths.long(),
        )
        return total_nll



class BinLoss(nn.Module):
    """Per-step NLL of the posterior-label distribution along the Viterbi hard path.
    Consumes posterior log-probs, not prior-free learned log-probs."""

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
        nll_sum = -(selected * mask).sum()                       # sum over whole batch
        return nll_sum
