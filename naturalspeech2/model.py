import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange

from naturalspeech2.encodec import EncodecWrapper
from naturalspeech2.log_mel_spectrogram import LogMelSpectrogramGenerator
from naturalspeech2.phoneme_encoder import PhonemeEncoder
from naturalspeech2.aligner import Aligner, ForwardSumLoss, BinLoss
from naturalspeech2.speech_prompt_encoder import SpeechPromptEncoder
from naturalspeech2.duration_predictor import DurationPredictor
from naturalspeech2.utils.utils import create_mask_from_lengths


class NaturalSpeech2Model(nn.Module):
    def __init__(self,
                 # global parameters
                 device: str = "cpu",
                 token_vocabulary_size: int = None,
                 hidden_dim: int = 512,
                 latent_dim: int = 128,
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

                 # Speech Prompt Encoder parameters
                 speech_prompt_encoder_layers: int = 6,
                 speech_prompt_encoder_heads: int = 8,
                 speech_prompt_encoder_filter_size: int = 2048,
                 speech_prompt_encoder_kernel_size: int = 9,
                 speech_prompt_encoder_dropout: float = 0.2,

                 # Duration Predictor parameters
                 duration_predictor_conv1d_layers: int = 30,
                 duration_predictor_conv1d_kernel_size: int = 3,
                 duration_predictor_attention_layers: int = 10,
                 duration_predictor_attention_heads: int = 8,
                 duration_predictor_dropout: float = 0.5,
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
            hidden_dim=hidden_dim,
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
            hidden_dim=hidden_dim,
            attn_channels=aligner_attn_channels,
            temperature=aligner_temperature,
            prior_w=prior_w,
        )

        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

        self.speech_prompt_encoder = SpeechPromptEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            transformer_layers=speech_prompt_encoder_layers,
            attention_heads=speech_prompt_encoder_heads,
            conv1d_filter_size=speech_prompt_encoder_filter_size,
            conv1d_kernel_size=speech_prompt_encoder_kernel_size,
            dropout=speech_prompt_encoder_dropout,
            rope_base=rope_base,
            rope_max_seq_len=rope_max_seq_len,
        )

        self.duration_predictor = DurationPredictor(
            hidden_dim=hidden_dim,
            conv1d_layers=duration_predictor_conv1d_layers,
            conv1d_kernel_size=duration_predictor_conv1d_kernel_size,
            attention_layers=duration_predictor_attention_layers,
            attention_heads=duration_predictor_attention_heads,
            dropout=duration_predictor_dropout,
        )

    @staticmethod
    def _expand_phoneme_encodings(
        phoneme_encodings: torch.Tensor,  # [B, P, dim_hidden]
        durations: torch.Tensor,          # [B, P]
    ):
        B, P, H = phoneme_encodings.shape
        device = phoneme_encodings.device
        durations = durations.to(torch.long)
        
        expanded_phoneme_encodings_list = []
        frame_lengths = []

        for b in range(B):
            durations_b = durations[b]          # [P]
            phoneme_encodings_b = phoneme_encodings[b]    # [P, hidden_dim]

            expanded_phoneme_encodings_b = torch.repeat_interleave(phoneme_encodings_b, durations_b, dim=0)  # [P_b, hidden_dim]

            expanded_phoneme_encodings_list.append(expanded_phoneme_encodings_b)
            frame_lengths.append(expanded_phoneme_encodings_b.shape[0])

        frame_lengths = torch.tensor(frame_lengths, device=device, dtype=torch.long)  # [B]

        expanded_phoneme_encodings = pad_sequence(expanded_phoneme_encodings_list, batch_first=True)  # [B, F, hidden_dim]
        F_max = expanded_phoneme_encodings.shape[1]

        frame_mask = create_mask_from_lengths(frame_lengths, max_len=F_max)  # [B, F, 1]

        return expanded_phoneme_encodings, frame_mask, frame_lengths

    @staticmethod
    def _generate_prompts_and_targets(
        audio_latents: torch.Tensor,     # [B, F, hidden_dim]
        audio_lengths: torch.Tensor,     # [B]
        min_prompt_pct: float,
        max_prompt_pct: float,
        hop_length: int
    ):
        device = audio_latents.device
        B, F, D = audio_latents.shape
        
        # hop_length in this case the downsample factor from audio samples to audio latents
        audio_latents_lengths = (audio_lengths / hop_length).ceil().long()

        prompt_latents_list = []
        target_latents_list = []

        for i in range(B):
            audio_latents_length = audio_latents_lengths[i].item()
            audio_latents_without_padding = audio_latents[i, :audio_latents_length, :] # [F, D] | F = audio latents length without padding
            
            min_len = int(audio_latents_length * min_prompt_pct) # minimum number of frames for the speech prompt
            max_len = int(audio_latents_length * max_prompt_pct) # maximum number of frames for the speech prompt
            prompt_len = torch.randint(low=min_len, high=max_len + 1, size=(1,)).item() # number of frames for the speech prompt

            # Prompt length must fit within the audio latents: prompt_start + prompt_len <= audio_latents_length
            # Random starting position for the prompt
            max_start = audio_latents_length - prompt_len
            prompt_start = torch.randint(low=0, high=max_start + 1, size=(1,)).item()
            prompt_end = prompt_start + prompt_len

            prompt = audio_latents_without_padding[prompt_start:prompt_end, :] # [F, D] | F = prompt_len

            # target: everything before and after the prompt concatenated
            target = torch.cat([
                audio_latents_without_padding[:prompt_start, :], 
                audio_latents_without_padding[prompt_end:, :]
            ], dim=0)

            prompt_latents_list.append(prompt)
            target_latents_list.append(target)

        prompt_latents_padded = pad_sequence(prompt_latents_list, batch_first=True)  # [B, F, D]
        target_latents_padded = pad_sequence(target_latents_list, batch_first=True)  # [B, F, D]


        prompt_latents_lengths = torch.tensor([p.shape[0] for p in prompt_latents_list], device=device)  # [B]
        prompt_max_len = prompt_latents_padded.shape[1]
        prompt_latents_mask = create_mask_from_lengths(
            prompt_latents_lengths,
            max_len=prompt_max_len
        )

        target_latents_lengths = torch.tensor([t.shape[0] for t in target_latents_list], device=device)  # [B]
        target_max_len = target_latents_padded.shape[1]
        target_latents_mask = create_mask_from_lengths(
            target_latents_lengths,
            max_len=target_max_len
        )

        return (prompt_latents_padded, prompt_latents_mask, prompt_latents_lengths,
                target_latents_padded, target_latents_mask, target_latents_lengths)


    def forward(
        self,
        audio: torch.Tensor,                  # [B, T]    | float
        audio_mask: torch.Tensor,             # [B, T, 1] | True/False
        audio_lengths: torch.Tensor,          # [B]       | int

        phoneme_tokens: torch.Tensor,         # [B, P]    | int
        phoneme_tokens_mask: torch.Tensor,    # [B, P, 1] | True/False
        phoneme_tokens_lengths: torch.Tensor, # [B]       | int
    ):
        """
        B: batch size
        T: number of audio samples
        P: seq_len of phonemes
        F: seq_len of frames
        """

        audio_encodings, frame_mask, frame_lengths = self.log_mel_spectrogram_generator(audio, audio_lengths)
        # audio_encodings: [B, F, n_mels]
        # frame_mask: [B, F, 1]
        # frame_lengths: [B]

        phoneme_encodings = self.phoneme_encoder(           # [B, P, hidden_dim]
            phoneme_tokens,
            phoneme_tokens_mask,
            phoneme_tokens_lengths
        )
        phoneme_encodings_mask = phoneme_tokens_mask        # [B, P, 1]
        phoneme_encodings_lengths = phoneme_tokens_lengths  # [B]

        durations, alignment_hard, alignment_soft, alignment_logprobs, attn_mask, alignment_logits_with_prior = self.aligner(
            audio_encodings,
            frame_mask,
            frame_lengths,
            phoneme_encodings,
            phoneme_encodings_mask,
            phoneme_encodings_lengths,
        )

        expanded_phoneme_encodings, frame_mask_expanded, frame_lengths_expanded = self._expand_phoneme_encodings(
            phoneme_encodings,
            durations,
        )
        # expanded_phoneme_encodings: [B, F, dim_hidden]
        # frame_mask_expanded: [B, F, 1]
        # frame_lengths_expanded: [B]

        audio_latents = self.encodec.get_latents(audio) # (B, F, D=128)

        prompt_latents, prompt_latents_mask, prompt_latents_lengths, target_latents, target_latents_mask, target_latents_lengths = self._generate_prompts_and_targets(
            audio_latents,
            audio_lengths,
            self.min_prompt_pct,
            self.max_prompt_pct,
            self.hop_length
        )
        # prompt_latents: [B, F, hidden_dim]             # target_latents: [B, F, hidden_dim]
        # prompt_latents_mask: [B, F, 1]                 # target_latents_mask: [B, F, 1]
        # prompt_latents_lengths: [B]                    # target_latents_lengths: [B]

        prompt_encodings = self.speech_prompt_encoder(      # [B, F, hidden_dim]
            prompt_latents,
            prompt_latents_mask,
            prompt_latents_lengths
        )
        prompt_encodings_mask = prompt_latents_mask         # [B, F, 1]
        prompt_encodings_lengths = prompt_latents_lengths   # [B]

        duration_predictor_durations = self.duration_predictor(
            phoneme_encodings,
            phoneme_encodings_mask,
            prompt_encodings,
            prompt_encodings_mask
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
    





