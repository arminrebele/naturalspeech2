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
        
        attn_soft, attn_logits = self.aligner_net(audio_encodings, phoneme_encodings, phoneme_tokens_mask)  # [B, 1, F, P]

        with torch.no_grad():
            # Masken kombinieren: [B,1,F] & [B,1,P] -> [B,F,P]
            frame_mask_2d = frame_mask.squeeze(1).bool()   # [B,F]
            phoneme_tokens_mask_2d = phoneme_tokens_mask.squeeze(1).bool()   # [B,P]
            attn_mask = frame_mask_2d.unsqueeze(2) & phoneme_tokens_mask_2d.unsqueeze(1)  # [B,F,P]

            attn_soft_2d = attn_soft.squeeze(1)           # [B,F,P]
            B, F, P = attn_soft_2d.shape
            attn_logprob = torch.log(attn_soft_2d + 1e-9) # [B,F,P]
            prior_logprob = compute_beta_binomial_prior(
                frame_lengths,
                phoneme_tokens_lengths, 
                frames_max=F,
                phoneme_tokens_max=P, 
                w=1.0 
            )

            logprob_for_viterbi = attn_logprob + prior_logprob

            alignment_mask = maximum_path(logprob_for_viterbi, attn_mask, frame_lengths, phoneme_tokens_lengths)  # [B,F,P] Hard alignment
            durations = alignment_mask.sum(dim=1).int()   # [B,P]

            return {
                "durations": durations, 
                "alignment_mask": alignment_mask,
                "attn_soft": attn_soft,
                "attn_logprob": attn_logprob
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
        attn_logits = torch.cdist(audio_features, phoneme_features)  # [B, F, P]
        attn_logits = rearrange(attn_logits, "b f p -> b 1 f p")

        mask_value = -torch.finfo(attn_logits.dtype).max
        mask = rearrange(phoneme_token_mask.bool(), "b 1 p -> b 1 1 p")
        attn_logits.masked_fill_(~mask, mask_value)

        attn_logits = -attn_logits / self.temperature # [B, 1, F, P]
        attn_soft = attn_logits.softmax(dim=-1)

        return attn_soft, attn_logits  # [B, 1, F, P]


def maximum_path(value: torch.Tensor, mask: torch.Tensor, frame_lengths: torch.Tensor, token_lengths: torch.Tensor) -> torch.Tensor:
    """
    Findet den wahrscheinlichsten monotonen Pfad mittels Viterbi-Algorithmus.
    Implementiert die DP-Logik: dp[i, j] = value[i, j] + max(dp[i-1, j], dp[i-1, j-1])
    
    value: [B, F, P] (Log-Wahrscheinlichkeiten)
    mask:  [B, F, P] (Boolesche Maske für gültige Positionen)
    """
    device = value.device
    dtype = value.dtype
    B, T_x, T_y = value.shape  # B=Batch, T_x=Frames (Audio), T_y=Phoneme (Text)
    
    # 1. DP-Tabelle und Pfad-Tracker initialisieren
    # dp speichert die kumulative Log-Wahrscheinlichkeit des besten Pfades
    dp = torch.full((B, T_x, T_y), float("-inf"), device=device, dtype=dtype)
    
    # path speichert die "Entscheidung" (0=bleiben, 1=wechseln)
    path = torch.zeros((B, T_x, T_y), dtype=torch.long, device=device)

    # 2. Initialisierung (Frame i=0)
    # Ein Pfad kann nur bei Frame 0, Phonem 0 beginnen.
    # Wir nutzen `mask` um sicherzustellen, dass dies eine gültige Position ist.
    dp[:, 0, 0] = torch.where(mask[:, 0, 0], value[:, 0, 0], float("-inf"))

    # 3. Dynamic Programming (DP)
    for i in range(1, T_x):  # Äußere Schleife über die Audio-Frames
        
        # Option 1: Beim gleichen Phonem j bleiben
        # Der Pfad kommt von dp[i-1, j]
        dp_stay = dp[:, i-1, :]
        
        # Option 2: Vom vorherigen Phonem j-1 wechseln
        # Der Pfad kommt von dp[i-1, j-1]
        # Wir paddern links mit -inf, da man bei j=0 nicht von j-1 kommen kann
        dp_move = torch.nn.functional.pad(dp[:, i-1, :], (1, 0), value=float("-inf"))[:, :-1]
        
        # Wähle die beste der beiden Optionen
        dp_max, indices = torch.max(torch.stack([dp_stay, dp_move]), dim=0)
        
        # Speichere die Entscheidung (0=bleiben, 1=wechseln)
        path[:, i, :] = indices
        
        # Aktualisiere die DP-Tabelle für den aktuellen Frame i
        # dp[i, j] = value[i, j] + max(dp[i-1, j], dp[i-1, j-1])
        # Maskierte Positionen werden auf -inf gesetzt
        dp[:, i, :] = torch.where(
            mask[:, i, :],
            value[:, i, :] + dp_max,
            float("-inf")
        )

    # 4. Backtracking
    # Finde die tatsächlichen Längen aus der Maske
    frame_lengths_idx = frame_lengths.long() - 1
    token_lengths_idx = token_lengths.long() - 1
    
    batch_indices = torch.arange(B, device=device)
    
    # Initialisiere die binäre Ausgabemaske
    alignment_mask = torch.zeros_like(path, dtype=torch.bool, device=device)
    
    # Starte das Backtracking beim letzten gültigen Phonem
    j = token_lengths_idx  # [B]

    # Gehe rückwärts durch die Frames
    for i in reversed(range(T_x)):
        # Prüfe für jeden Batch-Eintrag, ob wir noch in der gültigen Frame-Länge sind
        active = (i <= frame_lengths_idx)
        
        # Markiere die Position (i, j) als Teil des Pfades, falls aktiv
        alignment_mask[batch_indices, i, j] = alignment_mask[batch_indices, i, j] | active
        
        # Finde die Entscheidung, die an (i, j) getroffen wurde
        decision = path[batch_indices, i, j]  # 0 = bleiben, 1 = wechseln
        
        # Aktualisiere j für den nächsten Schritt (i-1)
        # j = j - (decision * active) # (Subtrahiere 1, wenn 'decision' 1 war UND wir aktiv sind)
        j = j - (decision & active).long()
        
        # Stelle sicher, dass j nicht negativ wird
        j = torch.clamp(j, min=0)

    # 5. Finale Maskierung
    # Stelle sicher, dass der Pfad nur dort existiert, wo die Maske es erlaubt
    alignment_mask = alignment_mask & mask
    
    return alignment_mask.float()


def compute_beta_binomial_prior(
    frame_lengths: torch.Tensor,  # [B]
    token_lengths: torch.Tensor,  # [B]
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
    token_lengths = token_lengths.view(B, 1, 1).float()        # [B, 1, 1]

    alpha = w * F_grid                                   # [1, F, 1]
    beta = w * (frame_lengths - F_grid + 1)              # [B, F, 1]

    alpha = torch.clamp(alpha, min=1e-5)
    beta = torch.clamp(beta, min=1e-5)

    
    ### Equation 12 from "One TTS Alignment To Rule Them All" ###
    # log space for numerical stability(lgamma instead faculty, lbeta instead beta)
    
    # Log Binomial Coefficient: log( "P_len" choose "P_grid" )
    # Formula: log(N!) - log(k!) - log((N-k)!)
    # N = token_lengths, k = P_grid

    #token_lengths - P_grid + 1 kann bei P_grid >= token_lengths <= 0
    safe_token_lengths_minus_P_grid_plus1 = torch.clamp(token_lengths - P_grid + 1.0, min=1.0)
    
    log_binom_coeff = (
        torch.lgamma(token_lengths + 1)
        - torch.lgamma(P_grid + 1)
        - torch.lgamma(safe_token_lengths_minus_P_grid_plus1)
    )

    # Log Beta Functions (Numerator and Denominator)
    safe_token_lengths_minus_P_grid_plus_beta = torch.clamp(token_lengths - P_grid + beta, min=1e-5)
    log_beta_numerator = torch.lbeta(P_grid + alpha, safe_token_lengths_minus_P_grid_plus_beta)
    log_beta_denominator = torch.lbeta(alpha, beta)

    log_prior = log_binom_coeff + log_beta_numerator - log_beta_denominator  # [B, F, P]
    ###

    mask_F = (F_grid <= frame_lengths) # [B, F, 1]  | True for all valid frames
    mask_P = (P_grid < token_lengths)  # [B, 1, P]  | True for all valid phoneme_tokens
    mask = mask_F & mask_P     # [B, F, P]

    # Where the mask is True -> use the calculated log_prior
    # Everywhere else (in padding) -> set to -inf (forbidden path)
    return torch.where(mask, log_prior, torch.tensor(float("-inf"), device=device))




if __name__ == "__main__":
    pass