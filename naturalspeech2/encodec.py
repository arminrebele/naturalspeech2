import torch
from transformers import EncodecModel, AutoProcessor

from einops import rearrange

from naturalspeech2.paths import ENCODEC_24KHZ_DIR


class EncodecWrapper:
    def __init__(self, device, bandwidth=24, auto_load=True):
        self.device = torch.device(device)
        self.bandwidth = bandwidth
        self.sampling_rate = 24000
        self.model_dir = ENCODEC_24KHZ_DIR
        self.model = None
        self.processor = None
        if auto_load:
            self.load_model()

    def load_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(self.model_dir)

        self.processor = AutoProcessor.from_pretrained(
            "facebook/encodec_24khz",
            cache_dir=cache_dir,
            use_fast=False
        )

        self.model = EncodecModel.from_pretrained(
            "facebook/encodec_24khz",
            cache_dir=cache_dir
        ).to(self.device).eval()

    @torch.no_grad()
    def encode(self, audio):
        inputs = self.processor(
            raw_audio=audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        output = self.model.encode(
            inputs["input_values"],
            inputs.get("padding_mask"),
            bandwidth=self.bandwidth,
        )

        return output.audio_codes, output.audio_scales     # output.audio_codes: (C=1, B, Q, F)
         # output.audio_codes => discrete codebook indices (0-1023), Shape: (Channel(Mono), Batch, Quantizer/Codebook, Frames/Time)

    @torch.no_grad()
    def get_latents(self, audio):
        # if batch tensor (B, T)
        if isinstance(audio, torch.Tensor) and audio.dim() == 2:
            audio = [a.cpu().numpy() for a in audio]
        # if single audio (T,)
        elif isinstance(audio, torch.Tensor):
            audio = [audio.cpu().numpy()]

        audio_codes, _ = self.encode(audio) # (C=1, B, Q, F)
        audio_codes_qbt = audio_codes.squeeze(0).permute(1, 0, 2).contiguous() # (Q, B, F)
        latents = self.model.quantizer.decode(audio_codes_qbt)  # (B, D=128, F)
        latents = rearrange(latents, "b d f -> b f d")          # [B, F, D]
        return latents

    @torch.no_grad()
    def decode_from_codes(self, audio_codes, audio_scales):
        if audio_scales is not None and len(audio_scales) > 0 and audio_scales[0] is not None:
            audio_scales = [audio_scales[0].to(self.device)]
        return self.model.decode(audio_codes.to(self.device), audio_scales)

    @torch.no_grad()
    def decode_from_latents(self, latents): # latents: [B, F, D]
        latents = rearrange(latents, "b f d -> b d f")  # [B, D, F]
        return self.model.decoder(latents.to(self.device))
