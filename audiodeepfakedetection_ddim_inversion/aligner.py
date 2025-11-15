import torch
from torch import nn
from einops import rearrange, repeat



class Aligner(nn.Module):
    def __init__(
        self,
        dim_audio=80,
        dim_hidden=512,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()

        self.aligner_net = AlignerNet(
            dim_audio = dim_audio,
            dim_hidden = dim_hidden,
            attn_channels = attn_channels,
            temperature = temperature,
        )

    def forward(
        self,
        audio_encodings,          # [B, dim_audio=80, F]
        frame_mask,               # [B, 1, F]
        frame_lengths,            # [B]
        phoneme_encodings,        # [B, dim_hidden=512, P]
        phoneme_tokens_mask,      # [B, 1, P]
        phoneme_tokens_lengths,   # [B]
    ) -> dict[str, torch.Tensor]:
        
        alignment_soft, alignment_logits = self.aligner_net(audio_encodings, phoneme_encodings, phoneme_tokens_mask)  # [B, 1, F, P]

        with torch.no_grad():
            # combine masks [B,1,F] & [B,1,P] -> [B,F,P]
            frame_mask_2d = frame_mask.squeeze(1).bool()   # [B,F]
            phoneme_tokens_mask_2d = phoneme_tokens_mask.squeeze(1).bool()   # [B,P]
            attn_mask = frame_mask_2d.unsqueeze(2) & phoneme_tokens_mask_2d.unsqueeze(1)  # [B,F,P]

            alignment_soft_2d = alignment_soft.squeeze(1)           # [B,F,P]
            B, F, P = alignment_soft_2d.shape
            alignment_logprobs = torch.log(alignment_soft_2d + 1e-9) # [B,F,P]
            prior_logprobs = compute_beta_binomial_prior(
                frame_lengths,
                phoneme_tokens_lengths, 
                frames_max=F,
                phoneme_tokens_max=P, 
                w=1.0 
            )

            alignment_logprobs_for_viterbi = alignment_logprobs + prior_logprobs # [B,F,P]
            alignment_hard = maximum_path(alignment_logprobs_for_viterbi, attn_mask, frame_lengths, phoneme_tokens_lengths)  # [B,F,P]
            durations = alignment_hard.sum(dim=1).int()   # [B,P]

            return {
                "durations": durations, 
                "alignment_hard": alignment_hard,
                "alignment_soft": alignment_soft,
                "alignment_logprobs": alignment_logprobs
            }



class AlignerNet(nn.Module):
    def __init__(self, dim_audio=80, dim_hidden=512, attn_channels=80, temperature=0.0005):
        super().__init__()
        self.temperature = temperature

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(dim_audio, dim_audio*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim_audio*2, dim_audio, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim_audio, attn_channels, kernel_size=1)
        )

        self.phoneme_encoder = nn.Sequential(
            nn.Conv1d(dim_hidden, dim_hidden*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim_hidden*2, attn_channels, kernel_size=1)
        )

    def forward(self, audio_encodings, phoneme_encodings, phoneme_token_mask):
        """
        audio_encodings:     [B, 80, F]
        phoneme_encodings:   [B, 512, P]
        phoneme_token_mask:  [B, 1, P]
        """

        audio_features = self.audio_encoder(audio_encodings)   # [B, 80, F]

        phoneme_features = self.phoneme_encoder(phoneme_encodings)   # [B, 80, P]

        # Transpose for cdist
        audio_features = rearrange(audio_features, "b c f -> b f c")  # [B, F, 80]
        phoneme_features = rearrange(phoneme_features, "b c p -> b p c")  # [B, P, 80]
        

        # L2 distances between frames and phonemes
        alignment_logits = torch.cdist(audio_features, phoneme_features)  # [B, F, P]
        alignment_logits = rearrange(alignment_logits, "b f p -> b 1 f p")
        alignment_logits = -alignment_logits / self.temperature # [B, 1, F, P]
        mask_value = -torch.finfo(alignment_logits.dtype).max
        mask = rearrange(phoneme_token_mask.bool(), "b 1 p -> b 1 1 p")
        alignment_logits.masked_fill_(~mask, mask_value)
        
        alignment_soft = alignment_logits.softmax(dim=-1)

        return alignment_soft, alignment_logits  # [B, 1, F, P]



def maximum_path(
        logprobs: torch.Tensor,      # [B, F, P]
        attn_mask: torch.Tensor,     # [B, F, P]
        frame_lengths: torch.Tensor,    # [B]
        phoneme_tokens_lengths: torch.Tensor  # [B]
) -> torch.Tensor:
    """

    """
    
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
    phoneme_tokens_lengths_idx = phoneme_tokens_lengths.long() - 1
    batch_indices = torch.arange(B, device=device)
    alignment_hard = torch.zeros_like(path, dtype=torch.bool, device=device) # initialize hard alignment
    p = phoneme_tokens_lengths_idx # start from the last phoneme token

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
    phoneme_tokens_lengths: torch.Tensor,  # [B]
    frames_max: int,
    phoneme_tokens_max: int, 
    w: float = 1.0
) -> torch.Tensor:
    """

    """
    device = frame_lengths.device
    B = len(frame_lengths) # batch size

    F_grid = torch.arange(1, frames_max + 1, device=device, dtype=torch.float).view(1, -1, 1)      # [1, F, 1]   | [1, 2, 3, ..., F]
    P_grid = torch.arange(0, phoneme_tokens_max, device=device, dtype=torch.float).view(1, 1, -1)  # [1, 1, P]   | [0, 1, 2, ..., P-1]

    frame_lengths = frame_lengths.view(B, 1, 1).float()        # [B, 1, 1]
    phoneme_tokens_lengths = phoneme_tokens_lengths.view(B, 1, 1).float()        # [B, 1, 1]

    alpha = w * F_grid                                   # [1, F, 1]
    beta = w * (frame_lengths - F_grid + 1)              # [B, F, 1]

    alpha = torch.clamp(alpha, min=1e-5)
    beta = torch.clamp(beta, min=1e-5)

    
    ### Equation 12 from "One TTS Alignment To Rule Them All" ###
    # log space for numerical stability(lgamma instead faculty, lbeta instead beta)
    
    # Log Binomial Coefficient: log( "P_len" choose "P_grid" )
    # Formula: log(N!) - log(k!) - log((N-k)!)
    # N = token_lengths, k = P_grid

    #phoneme_tokens_lengths - P_grid + 1 kann bei P_grid >= phoneme_tokens_lengths <= 0
    safe_phoneme_tokens_lengths_minus_P_grid_plus1 = torch.clamp(phoneme_tokens_lengths - P_grid + 1.0, min=1.0)
    
    log_binom_coeff = (
        torch.lgamma(phoneme_tokens_lengths + 1)
        - torch.lgamma(P_grid + 1)
        - torch.lgamma(safe_phoneme_tokens_lengths_minus_P_grid_plus1)
    )

    # Log Beta Functions (Numerator and Denominator)
    safe_phoneme_tokens_lengths_minus_P_grid_plus_beta = torch.clamp(phoneme_tokens_lengths - P_grid + beta, min=1e-5)
    log_beta_numerator = torch.lbeta(P_grid + alpha, safe_phoneme_tokens_lengths_minus_P_grid_plus_beta)
    log_beta_denominator = torch.lbeta(alpha, beta)

    log_prior = log_binom_coeff + log_beta_numerator - log_beta_denominator  # [B, F, P]
    ###

    mask_F = (F_grid <= frame_lengths) # [B, F, 1]  | True for all valid frames
    mask_P = (P_grid < phoneme_tokens_lengths)  # [B, 1, P]  | True for all valid phoneme_tokens
    mask = mask_F & mask_P     # [B, F, P]

    # Where the mask is True -> use the calculated log_prior
    # Everywhere else (in padding) -> set to -inf (forbidden path)
    return torch.where(mask, log_prior, torch.tensor(float("-inf"), device=device))




if __name__ == "__main__":
    pass