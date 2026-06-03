from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import EncodecModel

from einops import rearrange

from naturalspeech2.paths import ENCODEC_24KHZ_DIR, PROJECT_ROOT

ENCODER_HOP_LENGTH = 320
LATENT_DIM = 128  # Encodec 24kHz quantizer output dimension
SAMPLING_RATE = 24000  # facebook/encodec_24khz rate (paper used 16 kHz; 24 kHz forced by codec)


class EncodecWrapper(nn.Module):
    def __init__(
        self,
        bandwidth: int = 24,
        auto_load: bool = True,
        latent_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.bandwidth = bandwidth          # 24.0 kbps -> 32 codebooks
        self.sampling_rate = SAMPLING_RATE
        self.model_dir = ENCODEC_24KHZ_DIR
        self.model = None

        # Per-channel latent norm buffers. Persistent → checkpoint self-contained (stats
        # file not needed post-train). Default identity (0/1); overwritten by stats file if given.
        self.register_buffer('latent_mean', torch.zeros(LATENT_DIM), persistent=True)
        self.register_buffer('latent_std', torch.ones(LATENT_DIM), persistent=True)
        if latent_stats_path is not None:
            self._load_latent_stats(latent_stats_path)

        if auto_load:
            self.load_model()

    def _load_latent_stats(self, path: str) -> None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(
                f"Encodec latent stats file not found at {p}. "
                f"Run `python scripts/compute_encodec_latent_stats.py` to generate it, "
                f"or set encodec.latent_stats_path to null to disable normalization."
            )
        stats = torch.load(p, map_location='cpu', weights_only=True)
        mean = stats['mean'].float()
        std = stats['std'].float()
        assert mean.shape == (LATENT_DIM,), f"Expected mean shape ({LATENT_DIM},), got {mean.shape}"
        assert std.shape == (LATENT_DIM,), f"Expected std shape ({LATENT_DIM},), got {std.shape}"
        assert std.min() > 0, f"Encodec latent_std must be strictly positive; got min={std.min()}"
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def load_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(self.model_dir)

        self.model = EncodecModel.from_pretrained(
            "facebook/encodec_24khz",
            cache_dir=cache_dir
        )

        self.model.eval()
        self.model.requires_grad_(False)

        codebook_embeddings = torch.stack(
            [layer.codebook.embed for layer in self.model.quantizer.layers],
            dim=0,
        )  # [Q, K=1024, latent_dim=128] float32
        self.register_buffer('codebook_embeddings', codebook_embeddings, persistent=False)

    @torch.compiler.disable
    @torch.no_grad()
    def encode(
        self,
        audio   # [B, T]
    ):
        # Force FP32: under BF16 the quantizer's nearest-codebook L2 can flip assignments
        # at tight ties → silently corrupts the GT codes diffusion trains against.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            audio = audio.float()
            input_values = rearrange(audio, 'b t -> b 1 t')     # [B, C=1, T]   C=1 for mono audio

            output = self.model.encode(
                input_values,
                padding_mask=None,
                bandwidth=self.bandwidth,
            )

        # output.audio_codes: [C=1, B, Q=32, F] (1 s → F=75). Q = quantizer/codebook,
        # each codebook [index 0-1023, latent_dim 128]
        return output.audio_codes, output.audio_scales

    @torch.compiler.disable
    @torch.no_grad()
    def get_latents(
        self,
        audio,          # [B, T]
        audio_lengths   # [B]
    ):
        with torch.autocast(device_type=audio.device.type, enabled=False):
            codebook_indices, _ = self.encode(audio)                                           # [C=1, B, Q=32, F]
            codebook_indices = rearrange(codebook_indices, '1 b q f -> q b f').contiguous()    # [Q, B, F]

            audio_latents = self.model.quantizer.decode(codebook_indices)      # [B, D=128, F] | sum of the 32 codebook vectors per frame
            audio_latents = rearrange(audio_latents, "b d f -> b f d").contiguous() # [B, F, D]

            # Per-channel norm (z-μ)/σ; identity at defaults. Broadcasts [128]→[B,F,128].
            audio_latents = (audio_latents - self.latent_mean) / self.latent_std

            F = audio_latents.shape[1]
            audio_latents_lengths = (audio_lengths + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
            audio_latents_lengths = audio_latents_lengths.clamp(min=1, max=F)

            codebook_indices = rearrange(codebook_indices, 'q b f -> b f q').contiguous()      # [B, F, Q] long

        return audio_latents, audio_latents_lengths, codebook_indices

    def unnormalize_latents(self, latents):  # [B, F, D] normalized -> [B, F, D] raw
        # Inverse of get_latents norm. Used by CE-RVQ (raw codebook embeds) + decode_from_latents.
        return latents * self.latent_std + self.latent_mean

    @torch.compiler.disable
    @torch.no_grad()
    def decode_from_codes(self, codebook_indices, audio_scales):
        return self.model.decode(codebook_indices, audio_scales)

    @torch.compiler.disable
    @torch.no_grad()
    def decode_from_latents(self, latents): # latents: [B, F, D] in normalized space
        # Un-normalize before the Encodec decoder (trained on raw quantizer output).
        latents = self.unnormalize_latents(latents)
        latents = rearrange(latents, "b f d -> b d f").contiguous()  # [B, D, F]
        return self.model.decoder(latents)
