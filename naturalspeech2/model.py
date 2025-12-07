import torch
from torch import device, nn

from naturalspeech2.encodec import EncodecWrapper
from naturalspeech2.log_mel_spectrogram import LogMelSpectrogramGenerator
from naturalspeech2.phoneme_encoder import PhonemeEncoder
from naturalspeech2.aligner import Aligner, ForwardSumLoss, BinLoss
from naturalspeech2.utils.utils import expand_phoneme_encodings


class NaturalSpeech2Model(nn.Module):
    def __init__(self,
                 # global parameters
                 device: str = "cpu",
                 token_vocabulary_size: int = None,
                 dim_hidden: int = 512,
                 sampling_rate: int = 24000,

                 # Log Mel Spectrogram parameters
                 n_fft: int = 1024,
                 hop_length: int = 320,
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: float = None,

                 # Phoneme Encoder parameters
                 phoneme_encoder_layers: int = 6,
                 phoneme_encoder_heads: int = 8,
                 phoneme_encoder_filter_size: int = 2048,
                 phoneme_encoder_kernel_size: int = 9,
                 phoneme_encoder_dropout: float = 0.2,

                 # Aligner parameters
                 aligner_attn_channels: int = 80,
                 aligner_temperature: float = 0.0005,
                 prior_w: float = 1.0,

                 **kwargs
    ):
        super().__init__()

        self.encodec = EncodecWrapper(device)

        self.log_mel_spectrogram_generator = LogMelSpectrogramGenerator(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        self.phoneme_encoder = PhonemeEncoder(
            token_vocabulary_size=token_vocabulary_size,
            dim_hidden=dim_hidden,
            transformer_layers=phoneme_encoder_layers,
            attention_heads=phoneme_encoder_heads,
            conv1d_filter_size=phoneme_encoder_filter_size,
            conv1d_kernel_size=phoneme_encoder_kernel_size,
            dropout=phoneme_encoder_dropout,
        )

        self.aligner = Aligner(
            dim_audio=n_mels,
            dim_hidden=dim_hidden,
            attn_channels=aligner_attn_channels,
            temperature=aligner_temperature,
            prior_w=prior_w,
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
        
        phoneme_encodings = self.phoneme_encoder(phoneme_tokens, phoneme_tokens_mask, phoneme_tokens_lengths)

        durations, alignment_hard, alignment_soft, alignment_logprobs, attn_mask, alignment_logits_with_prior = self.aligner(
            audio_encodings,
            frame_mask,
            frame_lengths,
            phoneme_encodings,
            phoneme_tokens_mask,
            phoneme_tokens_lengths,
        )

        expanded_phoneme_encodings, frame_mask_expanded, frame_lengths_expanded = expand_phoneme_encodings(
            phoneme_encodings,
            durations,
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
    





