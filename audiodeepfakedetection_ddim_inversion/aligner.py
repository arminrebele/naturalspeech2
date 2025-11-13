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
        phoneme_encodings,      # [B, dim_hidden=512, P]
        frame_mask,             # [B, 1, F]
        token_mask,             # [B, 1, P]
        frame_lengths,          # [B]
        token_lengths,          # [B]
    ) -> dict[str, torch.Tensor]:
        
        attn_soft, attn_logits = self.aligner_net(audio_encodings, phoneme_encodings, token_mask)  # [B, 1, F, P]

        with torch.no_grad():
            # Masken kombinieren: [B,1,F] & [B,1,P] -> [B,F,P]
            frame_mask_2d = frame_mask.squeeze(1).bool()   # [B,F]
            token_mask_2d = token_mask.squeeze(1).bool()   # [B,P]
            attn_mask = frame_mask_2d.unsqueeze(2) & token_mask_2d.unsqueeze(1)  # [B,F,P]

            attn_soft_2d = attn_soft.squeeze(1)           # [B,F,P]
            alignment_mask = maximum_path(attn_soft_2d, attn_mask)  # [B,F,P]
            durations = alignment_mask.sum(dim=1).int()   # [B,P]



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

        attn_logits = -attn_logits / self.temperature
        attn_soft = attn_logits.softmax(dim=-1)

        return attn_soft, attn_logits  # [B, 1, F, P]


def maximum_path():
    pass






if __name__ == "__main__":
    pass