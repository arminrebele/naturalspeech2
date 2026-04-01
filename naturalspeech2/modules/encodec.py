import torch
from torch import nn
from transformers import EncodecModel

from einops import rearrange

from naturalspeech2.paths import ENCODEC_24KHZ_DIR


class EncodecWrapper(nn.Module):
    def __init__(self, bandwidth=24, auto_load=True):
        super().__init__()
        self.bandwidth = bandwidth
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
        
        # Freeze the pre-trained model parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, audio):
        # audio is perfectly padded [B, T] from dataloader buckets.
        # Encodec expects [Batch, Channels, Time] -> add the Mono channel.
        input_values = rearrange(audio, 'b t -> b 1 t')

        output = self.model.encode(
            input_values,
            padding_mask=None,
            bandwidth=self.bandwidth,
        )

        return output.audio_codes, output.audio_scales     # output.audio_codes: (C=1, B, Q, F)
         # output.audio_codes => discrete codebook indices (0-1023), Shape: (Channel(Mono), Batch, Quantizer/Codebook, Frames/Time)

    @torch.no_grad()
    def get_latents(self, audio):
        # audio: [B, T]
        audio_codes, _ = self.encode(audio) # [C=1, B, Q, F]

        # The quantizer's decode method expects the codebooks (quantizers) as the first dimension.
        audio_codes = rearrange(audio_codes, '1 b q f -> q b f').contiguous() # [Q, B, F]

        # De-quantize: Use codebook indices to look up the continuous latent vectors.
        latents = self.model.quantizer.decode(audio_codes)          # [B, D=128, F]
        latents = rearrange(latents, "b d f -> b f d").contiguous() # [B, F, D]
        return latents

    def decode_from_codes(self, audio_codes, audio_scales):
        return self.model.decode(audio_codes, audio_scales)

    def decode_from_latents(self, latents): # latents: [B, F, D]
        latents = rearrange(latents, "b f d -> b d f").contiguous()  # [B, D, F]
        return self.model.decoder(latents)
