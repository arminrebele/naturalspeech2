import torch
from torch import nn
from einops import rearrange



class Aligner(nn.Module):
    def __init__(
        self,
        audio_dim=80,
        hidden_dim=512,
        attn_channels=80,
        temperature=0.0005,
        prior_w=1.0,
    ):
        super().__init__()
        self.prior_w = prior_w

        self.aligner_net = AlignerNet(
            audio_dim = audio_dim,
            hidden_dim = hidden_dim,
            attn_channels = attn_channels,
            temperature = temperature,
        )

    def forward(
        self,
        audio_encodings,            # [B, F, audio_dim=80]
        frame_mask,                 # [B, F, 1]
        frame_lengths,              # [B]
        phoneme_encodings,          # [B, P, hidden_dim=512]
        phoneme_encodings_mask,     # [B, P, 1]
        phoneme_encodings_lengths,  # [B]
    ) -> dict[str, torch.Tensor]:
        
        alignment_soft, alignment_logits = self.aligner_net(audio_encodings, phoneme_encodings, phoneme_encodings_mask)  # [B, F, P]
        
        attn_mask = frame_mask & rearrange(phoneme_encodings_mask, 'b t 1 -> b 1 t')  # [B, F, P]

        B, F, P = alignment_soft.shape
        alignment_logprobs = torch.log(alignment_soft + 1e-9) # [B,F,P]
        
        prior_logprobs = compute_beta_binomial_prior(  # [B,F,P]
            frame_lengths,
            phoneme_encodings_lengths, 
            frames_max=F,
            phoneme_encodings_max=P, 
            w=self.prior_w 
        )

        alignment_logits_with_prior = alignment_logits + prior_logprobs # [B, F, P]
        alignment_logprobs_for_viterbi = alignment_logprobs + prior_logprobs # [B,F,P]

        with torch.no_grad():
            alignment_hard = maximum_path(          # [B,F,P]
                alignment_logprobs_for_viterbi,
                attn_mask, 
                frame_lengths, 
                phoneme_encodings_lengths
            )  
            durations = alignment_hard.sum(dim=1).int()   # [B,P]

        return (
            durations,                     # [B, P]
            alignment_hard,                # [B, F, P]
            alignment_soft,                # [B, F, P]
            alignment_logprobs,            # [B, F, P]
            attn_mask,                     # [B, F, P]
            alignment_logits_with_prior,   # [B, F, P]
        )



class AlignerNet(nn.Module):
    def __init__(
            self, 
            audio_dim=80, 
            hidden_dim=512, 
            attn_channels=80, 
            temperature=0.0005
    ):
        super().__init__()
        self.temperature = temperature

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(audio_dim*2, audio_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(audio_dim, attn_channels, kernel_size=1)
        )

        self.phoneme_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, attn_channels, kernel_size=1)
        )

    def forward(
            self,
            audio_encodings,         # [B, F, 80]
            phoneme_encodings,       # [B, P, 512]
            phoneme_encodings_mask   # [B, P, 1]
    ):
        # Transpose for Conv1d
        audio_encodings = rearrange(audio_encodings, 'b t d -> b d t')  # [B, 80, F]
        phoneme_encodings = rearrange(phoneme_encodings, 'b t d -> b d t')  # [B, 512, P]

        # Apply conv encoders
        audio_features = self.audio_encoder(audio_encodings)   # [B, 80, F]
        phoneme_features = self.phoneme_encoder(phoneme_encodings)   # [B, 80, P]

        # Transpose for cdist
        audio_features = rearrange(audio_features, "b d t -> b t d")        # [B, F, 80]
        phoneme_features = rearrange(phoneme_features, "b d t -> b t d")    # [B, P, 80]
        

        # L2 distances between frames and phonemes
        alignment_logits = torch.cdist(audio_features, phoneme_features)  # [B, F, P]
        alignment_logits = -alignment_logits / self.temperature # [B, F, P]

        mask_value = -torch.finfo(alignment_logits.dtype).max

        mask = rearrange(phoneme_encodings_mask.bool(), "b t 1 -> b 1 t")

        alignment_logits.masked_fill_(~mask, mask_value)
        
        alignment_soft = alignment_logits.softmax(dim=-1)

        return alignment_soft, alignment_logits  # [B, F, P]



def maximum_path(
        logprobs: torch.Tensor,               # [B, F, P]
        attn_mask: torch.Tensor,              # [B, F, P]
        frame_lengths: torch.Tensor,          # [B]
        phoneme_encodings_lengths: torch.Tensor  # [B]
) -> torch.Tensor:
    device = logprobs.device
    dtype = logprobs.dtype

    B, F, P = logprobs.shape

    ### Initialize dp table and path tracker ###

    # dp saves the cumulative log-probability of the best path to cell (f, p)
    dp = torch.full((B, F, P), float("-inf"), device=device, dtype=dtype)

    # path saves the decision made (0=stay, 1=move)
    path = torch.zeros((B, F, P), dtype=torch.long, device=device)

    # initialize
    dp[:, 0, 0] = torch.where(attn_mask[:, 0, 0], logprobs[:, 0, 0], float("-inf"))

    ### Dynamic Programming ###
    for f in range(1, F):
        dp_stay = dp[:, f-1, :]
        dp_move = torch.nn.functional.pad(dp[:, f-1, :], (1, 0), value=float("-inf"))[:, :-1]
        dp_max, indices = torch.max(torch.stack([dp_stay, dp_move]), dim=0)
        path[:, f, :] = indices  # save decision (0=stay, 1=move)

        dp[:, f, :] = torch.where(attn_mask[:, f, :], logprobs[:, f, :] + dp_max, float("-inf")) #update dp table


    ### Backtracking ###
    frame_lengths_idx = frame_lengths.long() - 1
    phoneme_encodings_lengths_idx = phoneme_encodings_lengths.long() - 1
    batch_indices = torch.arange(B, device=device)
    alignment_hard = torch.zeros_like(path, dtype=torch.bool, device=device) # initialize hard alignment
    p = phoneme_encodings_lengths_idx # start from the last phoneme token

    for f in reversed(range(F)):
        active = (f <= frame_lengths_idx) # check if within valid frame length
        alignment_hard[batch_indices, f, p] = alignment_hard[batch_indices, f, p] | active # mark the path position if valid
        decision = path[batch_indices, f, p]  # get the decision made at (f, p) (0=stay, 1=move)
        p = p - (decision & active).long() # if decision = 1 and active, move to previous phoneme token
        p = torch.clamp(p, min=0) # ensure p does not go negative
    
    alignment_hard = alignment_hard & attn_mask # final masking to ensure path only in valid positions

    return alignment_hard.float()



def compute_beta_binomial_prior(
    frame_lengths: torch.Tensor,  # [B]
    phoneme_encodings_lengths: torch.Tensor,  # [B]
    frames_max: int,
    phoneme_encodings_max: int, 
    w: float = 1.0
) -> torch.Tensor:
    """
    Paper: One TTS Alignment To Rule Them All
    Equations (12), (13)
    """
    device = frame_lengths.device
    B = len(frame_lengths) # batch size

    T = rearrange(frame_lengths, 'b -> b 1 1').float()          # [B, 1, 1]
    N = rearrange(phoneme_encodings_lengths, 'b -> b 1 1').float() - 1.0 # [B, 1, 1]

    t_grid = torch.arange(1, frames_max + 1, device=device).view(1, -1, 1)  # [1, T, 1]
    k_grid = torch.arange(0, phoneme_encodings_max, device=device).view(1, 1, -1)  # [1, 1, N]

    alpha = w * t_grid                     # [1, T, 1] | w * t
    alpha = torch.clamp(alpha, min=1e-5)

    beta = w * (T - t_grid + 1.0)          # [B, T, 1] | w * (T - t + 1)
    beta = torch.clamp(beta, min=1e-5)


    # Log Binomial Coefficient: log( N choose k ) = log(N!) - log(k!) - log((N-k)!)
    N_minus_k = torch.clamp(N - k_grid, min=1.0)  

    log_binom_coeff = (
        torch.lgamma(N + 1)            # log(N!)
        - torch.lgamma(k_grid + 1)     # log(k!)
        - torch.lgamma(N_minus_k + 1)  # log((N-k)!)
    )

    # Numerator: log B(k+α, N-k+β) = log Γ(k+α) + log Γ(N-k+β) - log Γ(N + α + β)
    log_beta_numerator = (
        torch.lgamma(k_grid + alpha)
        + torch.lgamma(torch.clamp(N - k_grid + beta, min=1e-5))
        - torch.lgamma(k_grid + alpha + torch.clamp(N - k_grid + beta, min=1e-5))
    )

    # Denominator: log B(α, β) = log Γ(α) + log Γ(β) - log Γ(α + β)
    log_beta_denominator = (
        torch.lgamma(alpha)
        + torch.lgamma(beta)
        - torch.lgamma(alpha + beta)
    )
    log_beta_denominator = log_beta_denominator.expand(B, frames_max, phoneme_encodings_max) # broadcast

    log_prior = log_binom_coeff + log_beta_numerator - log_beta_denominator  # [B, T, N] | [B, F, P]
    
    t_mask = (t_grid <= T)  # [B, T, 1]  | True for all valid frames
    k_mask = (k_grid < N + 1.0)  # [1, 1, N]  | True for all valid phoneme_tokens
    mask = t_mask & k_mask     # [B, T, N]

    # Where the mask is True -> use the calculated log_prior
    # Everywhere else (in padding) -> set to -inf (forbidden path)
    return torch.where(mask, log_prior, torch.tensor(float("-inf"), device=device))



class ForwardSumLoss(nn.Module):
    """
    Paper: RAD-TTS: Parallel Flow-Based TTS with Robust Alignment Learning and Diverse Synthesis | Appendix A.6
    """
    def __init__(self, blank_logprob: float = -1e4):
        super().__init__()
        self.blank_logprob = blank_logprob
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        alignment_logits: torch.Tensor,           # [B, F, P] # must be alignment_logits_with_prior
        frame_lengths: torch.Tensor,              # [B]
        phoneme_encodings_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        device = alignment_logits.device
        B, F, P = alignment_logits.shape

        alignment_logits_padded = torch.nn.functional.pad(   #  [B, F, P+1]
            alignment_logits,
            pad=(1, 0), 
            value=self.blank_logprob,
        )

        losses = []
        for b in range(B):
            T = int(frame_lengths[b].item())
            N = int(phoneme_encodings_lengths[b].item())

            curr = alignment_logits_padded[b, :T, :N + 1]    # [T, N+1]
            curr = rearrange(curr, "t c -> t 1 c")           # [T, 1, N+1]
            log_probs = self.log_softmax(curr)               # [T, 1, N+1]

            targets = torch.arange(1, N + 1, device=device, dtype=torch.long)  # [N]
            input_len = torch.tensor([T], device=device, dtype=torch.long)
            target_len = torch.tensor([N], device=device, dtype=torch.long)

            loss_b = self.ctc_loss(log_probs, targets, input_len, target_len)
            losses.append(loss_b)

        return torch.stack(losses).mean()



class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        alignment_logprobs: torch.Tensor,  # [B, F, P]
        alignment_hard: torch.Tensor,      # [B, F, P]  {0.0, 1.0}
    ) -> torch.Tensor:
        
        nll_per_batch = -(alignment_hard * alignment_logprobs).sum(dim=(1, 2))  # [B] | negative log likelihood
        denom = alignment_hard.sum(dim=(1, 2)).clamp_min(1.0)  # [B] | how many steps does the path have?
        loss_per_batch = nll_per_batch / denom  # [B] | average negative log likelihood per step

        return loss_per_batch.mean()


if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    F = 15
    P = 5
    dim_audio = 80
    dim_hidden = 512

    device = "cpu"

    # Inputs [B, T, D]
    audio_encodings = torch.randn(B, F, dim_audio, device=device)   # [B, F, 80]
    phoneme_encodings = torch.randn(B, P, dim_hidden, device=device)  # [B, P, 512]

    frame_lengths = torch.randint(low=10, high=F + 1, size=(B,), device=device)          # [B]
    phoneme_encodings_lengths = torch.randint(low=3, high=P + 1, size=(B,), device=device)  # [B]

    # Masks erstellen [B, T]
    frame_idx = torch.arange(F, device=device).unsqueeze(0)          # [1, F]
    frame_mask = (frame_idx < frame_lengths.unsqueeze(1)).unsqueeze(-1)                  # [B, F, 1]
    
    phoneme_idx = torch.arange(P, device=device).unsqueeze(0)        # [1, P]
    phoneme_encodings_mask = (phoneme_idx < phoneme_encodings_lengths.unsqueeze(1)).unsqueeze(-1) # [B, P, 1]

    aligner = Aligner(
        dim_audio,
        dim_hidden,
        attn_channels=80,
        temperature=5e-4,
    ).to(device)

    forward_sum_loss_fn = ForwardSumLoss()
    bin_loss_fn = BinLoss()

    # Forward Pass mit [B, T, 1] Masken
    durations, alignment_hard, alignment_soft, alignment_logprobs, attn_mask, alignment_logits_with_prior = aligner(
            audio_encodings=audio_encodings,
            frame_mask=frame_mask,
            frame_lengths=frame_lengths,
            phoneme_encodings=phoneme_encodings,
            phoneme_encodings_mask=phoneme_encodings_mask,
            phoneme_encodings_lengths=phoneme_encodings_lengths,
        )
    
    print("durations shape:", durations.shape)
    print("alignment_hard shape:", alignment_hard.shape)
    print("alignment_soft shape:", alignment_soft.shape)
    print("alignment_logprobs shape:", alignment_logprobs.shape)
    print("attn_mask shape:", attn_mask.shape)
    print("alignment_logits_with_prior shape:", alignment_logits_with_prior.shape)

    print("durations:", durations)
    
    # Loss Calculation
    loss_forward_sum = forward_sum_loss_fn(
        alignment_logits_with_prior,
        frame_lengths,
        phoneme_encodings_lengths
    )

    print("ForwardSumLoss:", loss_forward_sum.item())