import torch
from torch import nn
from einops import rearrange, repeat

class AlignerNet(nn.Module):
    def __init__(self, dim_mel=80, dim_hidden=512, attn_channels=80, temperature=0.0005):
        super().__init__()
        self.temperature = temperature

        self.log_mel_encoder = nn.Sequential(
            nn.Conv1d(dim_mel, dim_mel*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim_mel*2, dim_mel, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(dim_mel, attn_channels, kernel_size=1)
        )

        self.phoneme_encoder = nn.Sequential(
            nn.Conv1d(dim_hidden, dim_hidden*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim_hidden*2, attn_channels, kernel_size=1)
        )

    def forward(self, log_mel_spectrogram, phoneme_encodings, phoneme_token_mask):
        """
        log_mel_spectrogram: [B, 80, F]
        phoneme_encodings:   [B, 512, P]
        phoneme_token_mask:  [B, 1, P]
        """

        log_mel_features = self.log_mel_encoder(log_mel_spectrogram)   # [B, 80, F]

        phoneme_features = self.phoneme_encoder(phoneme_encodings)   # [B, 80, P]

        # Transpose for cdist
        log_mel_features = rearrange(log_mel_features, "b c f -> b f c")
        phoneme_features = rearrange(phoneme_features, "b c p -> b p c")
        

        # L2 distances between frames and phonemes
        attn_logits = torch.cdist(log_mel_features, phoneme_features)  # [B, F, P]
        attn_logits = rearrange(attn_logits, "b f p -> b 1 f p")

        mask_value = -torch.finfo(attn_logits.dtype).max
        mask = rearrange(phoneme_token_mask.bool(), "b 1 p -> b 1 1 p")
        attn_logits.masked_fill_(~mask, mask_value)

        attn_logits = -attn_logits / self.temperature
        attn_soft = attn_logits.softmax(dim=-1)

        return attn_soft, attn_logits  # [B, 1, F, P]


if __name__ == "__main__":
    pass