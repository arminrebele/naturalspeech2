import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from naturalspeech2.encodec import EncodecWrapper
from naturalspeech2.log_mel_spectrogram import LogMelSpectrogramGenerator
from naturalspeech2.phoneme_encoder import PhonemeEncoder
from naturalspeech2.aligner import Aligner, ForwardSumLoss, BinLoss
from naturalspeech2.speech_prompt_encoder import SpeechPromptEncoder


class NaturalSpeech2Model(nn.Module):
    def __init__(self,
                 # global parameters
                 device: str = "cpu",
                 token_vocabulary_size: int = None,
                 dim_hidden: int = 512,
                 sampling_rate: int = 24000,
                 rope_base: float = 10000.0,
                 rope_max_seq_len: int = 3000,
                 min_prompt_pct: float = 0.2,
                 max_prompt_pct: float = 0.5,

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
    ):
        super().__init__()
        self.min_prompt_pct = min_prompt_pct
        self.max_prompt_pct = max_prompt_pct
        self.hop_length = hop_length

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
            rope_base=rope_base,
            rope_max_seq_len=rope_max_seq_len,
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

    @staticmethod
    def _expand_phoneme_encodings(
        phoneme_encodings: torch.Tensor,  # [B, H, P]
        durations: torch.Tensor,          # [B, P]
    ):
        """
        
        """
        phoneme_encodings = phoneme_encodings.transpose(1, 2)  # [B, P, H]
        B, P, H = phoneme_encodings.shape
        device = phoneme_encodings.device
        durations = durations.to(torch.long)
        
        expanded_phoneme_encodings_list = []
        frame_lengths = []

        for b in range(B):
            durations_b = durations[b]          # [P]
            phoneme_encodings_b = phoneme_encodings[b]    # [P, H]

            expanded_phoneme_encodings_b = torch.repeat_interleave(phoneme_encodings_b, durations_b, dim=0)  # [T_b, H]

            expanded_phoneme_encodings_list.append(expanded_phoneme_encodings_b)
            frame_lengths.append(expanded_phoneme_encodings_b.shape[0])

        frame_lengths = torch.tensor(frame_lengths, device=device, dtype=torch.long)  # [B]

        expanded_phoneme_encodings = pad_sequence(expanded_phoneme_encodings_list, batch_first=True)  # [B, T_max, H]
        F_max = expanded_phoneme_encodings.shape[1]

        frame_idx = torch.arange(F_max, device=device).unsqueeze(0)     # [1, F_max]
        frame_mask = (frame_idx < frame_lengths.unsqueeze(1)).unsqueeze(1)  # [B, 1, F_max]

        expanded_phoneme_encodings = expanded_phoneme_encodings.transpose(1, 2)  # [B, H, F_max]

        return expanded_phoneme_encodings, frame_mask, frame_lengths

    @staticmethod
    def _generate_prompt_and_targets(audio_latents, audio_lengths, min_prompt_pct, max_prompt_pct, hop_length):
        """
        audio_latents: [B, D, F]
        """
        device = audio_latents.device
        B, D, F = audio_latents.shape
        
        downsample_factor = hop_length
        audio_latents_lengths = (audio_lengths / downsample_factor).ceil().long()

        audio_prompt_latents = []
        audio_target_latents = []

        for i in range(B):
            audio_latents_length = audio_latents_lengths[i].item()
            audio_latents_without_padding = audio_latents[i, :, :audio_latents_length] # [D, F] | F = audio latents length without padding
            
            min_len = int(audio_latents_length * min_prompt_pct) # minimum number of frames for the speech prompt
            max_len = int(audio_latents_length * max_prompt_pct) # maximum number of frames for the speech prompt

            prompt_len = torch.randint(low=min_len, high=max_len + 1, size=(1,)).item() # number of frames for the speech prompt

            # Prompt length must fit within the audio latents: prompt_start + prompt_len <= audio_latents_length
            max_start = audio_latents_length - prompt_len
            prompt_start = torch.randint(low=0, high=max_start + 1, size=(1,)).item()
            prompt_end = prompt_start + prompt_len

            prompt = audio_latents_without_padding[:, prompt_start:prompt_end] # [D, F] | F = prompt_len

            # target: everything before and after the prompt concatenated
            target = torch.cat([
                audio_latents_without_padding[:, :prompt_start], 
                audio_latents_without_padding[:, prompt_end:]
            ], dim=-1)

            audio_prompt_latents.append(prompt)
            audio_target_latents.append(target)
        
        # [D, F] -> .t() -> [F, D] -> pad_sequence -> [B, F, D] -> transpose -> [B, D, F]
        audio_prompt_latents_padded = pad_sequence([p.t() for p in audio_prompt_latents], batch_first=True).transpose(1, 2)
        audio_target_latents_padded = pad_sequence([t.t() for t in audio_target_latents], batch_first=True).transpose(1, 2)
        
        audio_prompt_latents_lengths = torch.tensor([p.shape[-1] for p in audio_prompt_latents], device=device)
        audio_target_latents_lengths = torch.tensor([t.shape[-1] for t in audio_target_latents], device=device)

        return audio_prompt_latents_padded, audio_prompt_latents_lengths, audio_target_latents_padded, audio_target_latents_lengths


    def forward(
        self,
        audio: torch.Tensor,                  # [B, T]    | float
        audio_mask: torch.Tensor,             # [B, 1, T] | True/False
        audio_lengths: torch.Tensor,          # [B]       | int

        phoneme_tokens: torch.Tensor,         # [B, P]    | int
        phoneme_tokens_mask: torch.Tensor,    # [B, 1, P] | True/False
        phoneme_tokens_lengths: torch.Tensor, # [B]       | int
    ):

        audio_encodings, frame_mask, frame_lengths = self.log_mel_spectrogram_generator(audio, audio_lengths)
        # audio_encodings: [B, n_mels, F]
        # frame_mask: [B, 1, F]
        # frame_lengths: [B]

        phoneme_encodings = self.phoneme_encoder(   # [B, dim_hidden, P]
            phoneme_tokens,
            phoneme_tokens_mask,
            phoneme_tokens_lengths
        )

        durations, alignment_hard, alignment_soft, alignment_logprobs, attn_mask, alignment_logits_with_prior = self.aligner(
            audio_encodings,
            frame_mask,
            frame_lengths,
            phoneme_encodings,
            phoneme_tokens_mask,
            phoneme_tokens_lengths,
        )

        expanded_phoneme_encodings, frame_mask_expanded, frame_lengths_expanded = self._expand_phoneme_encodings(
            phoneme_encodings,
            durations,
        )

        audio_latents = self.encodec.get_latents(audio) # (B, D=128, F)

        audio_prompt_latents, audio_prompt_latents_lengths, audio_target_latents, audio_target_latents_lengths = self._generate_prompt_and_targets(
            audio_latents,
            audio_lengths,
            self.min_prompt_pct,
            self.max_prompt_pct,
            self.hop_length
        )


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
    





