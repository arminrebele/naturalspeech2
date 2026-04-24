import torch
from torch import nn
from transformers import EncodecModel

from einops import rearrange

from naturalspeech2.paths import ENCODEC_24KHZ_DIR

ENCODER_HOP_LENGTH = 320


class EncodecWrapper(nn.Module):
    def __init__(self, bandwidth=24, auto_load=True):
        super().__init__()
        self.bandwidth = bandwidth          # 24.0 kbps -> 32 codebooks
        self.sampling_rate = 24000
        self.model_dir = ENCODEC_24KHZ_DIR
        self.model = None
        if auto_load:
            self.load_model()

    def load_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(self.model_dir)

        self.model = EncodecModel.from_pretrained(
            "facebook/encodec_24khz",
            cache_dir=cache_dir
        )

        self.model.eval()

        codebook_embeddings = torch.stack(
            [layer.codebook.embed for layer in self.model.quantizer.layers],
            dim=0,
        )  # [Q, K=1024, D=128] float32
        self.register_buffer('codebook_embeddings', codebook_embeddings, persistent=False)

    @torch.no_grad()
    def encode(
        self,
        audio   # [B, T]
    ):
        input_values = rearrange(audio, 'b t -> b 1 t')     # [B, C=1, T]   C=1 for mono audio

        output = self.model.encode(
            input_values,
            padding_mask=None,
            bandwidth=self.bandwidth,
        )

        # output.audio_codes: [C=1, B, Q=32, F]      e.g. 1 second => F=75 frames
        # Q = Quantizer/Codebook
        # each codebook [codebook_index=0-1023, latent_dim=128]
        return output.audio_codes, output.audio_scales

    @torch.no_grad()
    def get_latents(
        self, 
        audio,          # [B, T]
        audio_lengths   # [B]
    ):
        audio_codes, _ = self.encode(audio) # [C=1, B, Q=32, F]
        audio_codes = rearrange(audio_codes, '1 b q f -> q b f').contiguous() # [Q, B, F]

        audio_latents = self.model.quantizer.decode(audio_codes)          # [B, D=128, F] | sum of the 32 codebook vectors per frame
        audio_latents = rearrange(audio_latents, "b d f -> b f d").contiguous() # [B, F, D]

        F = audio_latents.shape[1]
        audio_latents_lengths = (audio_lengths + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
        audio_latents_lengths = audio_latents_lengths.clamp(min=1, max=F)

        codes = rearrange(audio_codes, 'q b f -> b q f').contiguous()  # [B, Q, F] long

        return audio_latents, audio_latents_lengths, codes

    def decode_from_codes(self, audio_codes, audio_scales):
        return self.model.decode(audio_codes, audio_scales)

    def decode_from_latents(self, latents): # latents: [B, F, D]
        latents = rearrange(latents, "b f d -> b d f").contiguous()  # [B, D, F]
        return self.model.decoder(latents)
