import torch
from torch import nn

from audiodeepfakedetection_ddim_inversion.encodec import EncodecWrapper
from audiodeepfakedetection_ddim_inversion.log_mel_spectrogram import LogMelSpectrogramGenerator
from audiodeepfakedetection_ddim_inversion.phoneme_encoder import PhonemeEncoder
from audiodeepfakedetection_ddim_inversion.aligner import Aligner, ForwardSumLoss, BinLoss



class NaturalSpeech2Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.encodec = EncodecWrapper(config["device"])

        self.log_mel_spectrogram_generator = LogMelSpectrogramGenerator(
            sampling_rate=config["sampling_rate"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            f_min=config["f_min"],
            f_max=config["f_max"],
        )

        self.phoneme_encoder = PhonemeEncoder() # TODO: implement PhonemeEncoder

        self.aligner = Aligner(
            dim_audio=config.get("dim_audio", 80),
            dim_hidden=config.get("dim_hidden", 512),
            attn_channels=config.get("attn_channels", 80),
            temperature=config.get("temperature", 5e-4),
        )

        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor, 
        audio_lengths: torch.Tensor,

        phoneme_tokens: torch.Tensor,
        phoneme_tokens_mask: torch.Tensor,
        phoneme_tokens_lengths: torch.Tensor,
    ):

        audio_encodings, frame_mask, frame_lengths = self.log_mel_spectrogram_generator(audio, audio_lengths)
        
        phoneme_encodings = self.phoneme_encoder(phoneme_tokens)

        durations, alignment_hard, alignment_soft, alignment_logprobs, attn_mask, alignment_logits_with_prior = self.aligner(
            audio_encodings,
            frame_mask,
            frame_lengths,
            phoneme_encodings,
            phoneme_tokens_mask,
            phoneme_tokens_lengths,
        )

        audio_latents = self.encodec.get_latents(audio)


        #### Compute Losses ####
        
        forward_sum_loss = self.forward_sum_loss(
            alignment_logits_with_prior,
            frame_lengths,
            phoneme_tokens_lengths
        )

        bin_loss = self.bin_loss(
            alignment_logprobs,
            alignment_hard
        )

        loss = forward_sum_loss + bin_loss

        return {
            "forward_sum_loss": forward_sum_loss,
            "bin_loss": bin_loss,
            "loss": loss,
        }
    





