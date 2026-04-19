import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from naturalspeech2.modules.encodec import EncodecWrapper
from naturalspeech2.modules.log_mel_spectrogram import LogMelSpectrogramGenerator
from naturalspeech2.modules.phoneme_encoder import PhonemeEncoder
from naturalspeech2.modules.aligner import Aligner, ForwardSumLoss, BinLoss
from naturalspeech2.modules.speech_prompt_encoder import SpeechPromptEncoder
from naturalspeech2.modules.duration_predictor import DurationPredictor
from naturalspeech2.modules.pitch_predictor import PitchPredictor
from naturalspeech2.modules.diffusion_model import DiffusionModel
from naturalspeech2.modules.layers import Conv1D
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
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: float = None,

                 # Phoneme Encoder parameters
                 phoneme_encoder_layers: int = 6,
                 phoneme_encoder_heads: int = 8,
                 phoneme_encoder_filter_size: int = 2048,
                 phoneme_encoder_kernel_size: int = 9,
                 phoneme_encoder_conv_dropout: float = 0.2,
                 phoneme_encoder_attn_weights_dropout: float = 0.2,
                 phoneme_encoder_attn_out_dropout: float = 0.2,

                 # Aligner parameters
                 aligner_attn_channels: int = 80,
                 aligner_temperature: float = 0.0005,
                 prior_w: float = 1.0,

                 # Speech Prompt Encoder parameters
                 speech_prompt_encoder_layers: int = 6,
                 speech_prompt_encoder_heads: int = 8,
                 speech_prompt_encoder_filter_size: int = 2048,
                 speech_prompt_encoder_kernel_size: int = 9,
                 speech_prompt_encoder_conv_dropout: float = 0.2,
                 speech_prompt_encoder_attn_weights_dropout: float = 0.2,
                 speech_prompt_encoder_attn_out_dropout: float = 0.2,

                 # Duration Predictor parameters
                 duration_predictor_conv1d_layers: int = 30,
                 duration_predictor_conv1d_kernel_size: int = 3,
                 duration_predictor_attention_layers: int = 10,
                 duration_predictor_attention_heads: int = 8,
                 duration_predictor_conv_dropout: float = 0.5,
                 duration_predictor_attn_weights_dropout: float = 0.5,
                 duration_predictor_attn_out_dropout: float = 0.5,

                 # Pitch Predictor parameters
                 pitch_predictor_conv1d_layers: int = 30,
                 pitch_predictor_conv1d_kernel_size: int = 5,
                 pitch_predictor_attention_layers: int = 10,
                 pitch_predictor_attention_heads: int = 8,
                 pitch_predictor_conv_dropout: float = 0.5,
                 pitch_predictor_attn_weights_dropout: float = 0.5,
                 pitch_predictor_attn_out_dropout: float = 0.5,

                 # Diffusion Model parameters
                 diffusion_model_wavenet_layers: int = 40,
                 diffusion_model_wavenet_kernel_size: int = 3,
                 diffusion_model_wavenet_dilation: int = 2,
                 diffusion_model_wavenet_filter_size: int = 1024,
                 diffusion_model_attention_heads: int = 8,
                 diffusion_model_query_tokens: int = 32,
                 diffusion_model_attn_weights_dropout: float = 0.2,
                 diffusion_model_attn_out_dropout: float = 0.2,
                 diffusion_model_wavenet_attn_weights_dropout: float = 0.2,
                 diffusion_model_wavenet_attn_out_dropout: float = 0.2,
                 diffusion_model_wavenet_gate_dropout: float = 0.2,
                 diffusion_model_beta_min: float = 0.05,
                 diffusion_model_beta_max: float = 20.0,
                 diffusion_model_sampling_steps: int = 150,
                 diffusion_model_sampling_temperature: float = 1.44,
    ):
        super().__init__()
        self.min_prompt_pct = min_prompt_pct
        self.max_prompt_pct = max_prompt_pct

        self.encodec = EncodecWrapper()

        self.log_mel_spectrogram_generator = LogMelSpectrogramGenerator(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
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
            conv_dropout=phoneme_encoder_conv_dropout,
            attn_weights_dropout=phoneme_encoder_attn_weights_dropout,
            attn_out_dropout=phoneme_encoder_attn_out_dropout,
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
            conv_dropout=speech_prompt_encoder_conv_dropout,
            attn_weights_dropout=speech_prompt_encoder_attn_weights_dropout,
            attn_out_dropout=speech_prompt_encoder_attn_out_dropout,
            rope_base=rope_base,
            rope_max_seq_len=rope_max_seq_len,
        )

        self.duration_predictor = DurationPredictor(
            hidden_dim=hidden_dim,
            conv1d_layers=duration_predictor_conv1d_layers,
            conv1d_kernel_size=duration_predictor_conv1d_kernel_size,
            attention_layers=duration_predictor_attention_layers,
            attention_heads=duration_predictor_attention_heads,
            conv_dropout=duration_predictor_conv_dropout,
            attn_weights_dropout=duration_predictor_attn_weights_dropout,
            attn_out_dropout=duration_predictor_attn_out_dropout,
        )

        self.pitch_predictor = PitchPredictor(
            hidden_dim=hidden_dim,
            conv1d_layers=pitch_predictor_conv1d_layers,
            conv1d_kernel_size=pitch_predictor_conv1d_kernel_size,
            attention_layers=pitch_predictor_attention_layers,
            attention_heads=pitch_predictor_attention_heads,
            conv_dropout=pitch_predictor_conv_dropout,
            attn_weights_dropout=pitch_predictor_attn_weights_dropout,
            attn_out_dropout=pitch_predictor_attn_out_dropout,
        )

        # Projects per-frame pitch (1 channel) up to hidden_dim so it can be
        # added to expanded_phoneme_encodings to form the diffusion condition c.
        self.pitch_projection = Conv1D(1, hidden_dim, 1)

        self.diffusion_model = DiffusionModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            wavenet_layers=diffusion_model_wavenet_layers,
            wavenet_kernel_size=diffusion_model_wavenet_kernel_size,
            wavenet_dilation=diffusion_model_wavenet_dilation,
            wavenet_filter_size=diffusion_model_wavenet_filter_size,
            attention_heads=diffusion_model_attention_heads,
            query_tokens=diffusion_model_query_tokens,
            attn_weights_dropout=diffusion_model_attn_weights_dropout,
            attn_out_dropout=diffusion_model_attn_out_dropout,
            wavenet_attn_weights_dropout=diffusion_model_wavenet_attn_weights_dropout,
            wavenet_attn_out_dropout=diffusion_model_wavenet_attn_out_dropout,
            wavenet_gate_dropout=diffusion_model_wavenet_gate_dropout,
            beta_min=diffusion_model_beta_min,
            beta_max=diffusion_model_beta_max,
            sampling_steps=diffusion_model_sampling_steps,
            sampling_temperature=diffusion_model_sampling_temperature,
        )

    @staticmethod
    def _expand_phoneme_encodings(
        phoneme_encodings: torch.Tensor,  # [B, P, D]
        durations: torch.Tensor,          # [B, P]
        max_frames: int,                  # F | target frame count aligned to the mel grid
    ):
        _, P, D = phoneme_encodings.shape
        durations = durations.to(torch.long)

        frame_lengths = durations.sum(dim=1)  # [B]

        # Cumulative ends: duration_ends[b, p] = first frame index NOT belonging to phoneme p.
        # Non-decreasing (durations >= 0), so valid input for searchsorted.
        duration_ends = durations.cumsum(dim=1)  # [B, P]

        # For each frame f, the phoneme it belongs to is the smallest p with f < duration_ends[b, p].
        frame_positions = repeat(torch.arange(max_frames, device=durations.device), 'f -> b f', b=durations.shape[0])  # [B, F]
        phoneme_idx = torch.searchsorted(duration_ends, frame_positions, right=True).clamp(max=P - 1)  # [B, F]

        expanded_phoneme_encodings = torch.gather(phoneme_encodings, 1, repeat(phoneme_idx, 'b f -> b f d', d=D))  # [B, F, D]

        frame_mask = create_mask_from_lengths(frame_lengths, max_len=max_frames)  # [B, F, 1]
        expanded_phoneme_encodings = expanded_phoneme_encodings * frame_mask.to(expanded_phoneme_encodings.dtype)

        return expanded_phoneme_encodings, frame_mask, frame_lengths

    @staticmethod
    def _generate_prompts_and_targets(
        audio_latents: torch.Tensor,            # [B, F, D]
        audio_latents_lengths: torch.Tensor,    # [B]
        min_prompt_pct: float,
        max_prompt_pct: float,
    ):
        device = audio_latents.device
        B, F, D = audio_latents.shape

        # Compute per-sample prompt length bounds
        min_lens = (audio_latents_lengths.float() * min_prompt_pct).long().clamp(min=1)  # [B] | minimum number of frames for the speech prompt
        max_lens = (audio_latents_lengths.float() * max_prompt_pct).long().clamp(min=1)  # [B] | maximum number of frames for the speech prompt

        # Sample prompt_lengths in [min_lens, max_lens] — one rand call covers both samples (one kernel launch)
        rand = torch.rand(B, 2, device=device)                                                      # [B, 2]
        range_lens = (max_lens - min_lens + 1).float()                                              # [B]
        prompt_latents_lengths = min_lens + (rand[:, 0] * range_lens).floor().long()                # [B] | number of frames for the speech prompt

        # Sample prompt_starts in [0, lengths - prompt_lengths]
        max_starts = audio_latents_lengths - prompt_latents_lengths                                 # [B] | maximum starting index for the speech prompt to ensure it fits within the audio latents
        prompt_starts = (rand[:, 1] * (max_starts + 1).float()).floor().long()                      # [B] | frame index where the prompt starts
        prompt_ends = prompt_starts + prompt_latents_lengths                                        # [B] | frame index where the prompt ends (exclusive)

        # Extract prompt
        max_prompt_len = math.ceil(max_prompt_pct * F)
        j_p = rearrange(torch.arange(max_prompt_len, device=device), 'fp -> 1 fp')                              # [1, Fp] | [0, 1, 2, ..., Fp-1] -> relative offset
        prompt_idx = (rearrange(prompt_starts, 'b -> b 1') + j_p).clamp(max=F - 1)                              # [B, Fp] | frame indices for the prompt in the audio latents
        prompt_latents = torch.gather(audio_latents, 1, repeat(prompt_idx, 'b fp -> b fp d', d=D))              # [B, Fp, D]
        prompt_latents_mask = rearrange(j_p < rearrange(prompt_latents_lengths, 'b -> b 1'), 'b fp -> b fp 1')  # [B, Fp, 1]
        prompt_latents = prompt_latents * prompt_latents_mask.to(prompt_latents.dtype)                          # mask out padding frames in the prompt latents

        # Extract target
        max_target_len = F - int(math.floor(min_prompt_pct * F))
        j_t = rearrange(torch.arange(max_target_len, device=device), 'ft -> 1 ft')                  # [1, Ft] | [0, 1, 2, ..., Ft-1] 
        target_idx = (                                                                              # [B, Ft]
            j_t
            + (j_t >= rearrange(prompt_starts, 'b -> b 1')).long()
            * rearrange(prompt_latents_lengths, 'b -> b 1')
        ).clamp(max=F - 1)
                                                                                   
        target_latents = torch.gather(audio_latents, 1, repeat(target_idx, 'b ft -> b ft d', d=D))              # [B, Ft, D]
        target_latents_lengths = audio_latents_lengths - prompt_latents_lengths                                 # [B]
        target_latents_mask = rearrange(j_t < rearrange(target_latents_lengths, 'b -> b 1'), 'b ft -> b ft 1')  # [B, Ft, 1]
        target_latents = target_latents * target_latents_mask.to(target_latents.dtype)

        return (
            prompt_latents,          # [B, Fp, D]
            prompt_latents_mask,     # [B, Fp, 1]
            prompt_latents_lengths,  # [B]
            target_latents,          # [B, Ft, D]
            target_latents_mask,     # [B, Ft, 1]
            target_latents_lengths,  # [B]
            prompt_starts,           # [B]
            prompt_ends,             # [B]
            target_idx,              # [B, Ft]
        )

    def _generate_condition(
        self,
        expanded_phoneme_encodings,  # [B, F, D]
        pitch,                       # [B, F]          | GT F0 in Hz during training
        frame_mask,                  # [B, F, 1]       | bool
    ):
        pitch = rearrange(pitch, 'b f -> b f 1')
        pitch_projection = self.pitch_projection(pitch, frame_mask)  # [B, F, D]
        condition = expanded_phoneme_encodings + pitch_projection
        condition = condition * frame_mask.to(condition.dtype)
        return condition


    def forward(
        self,
        audio: torch.Tensor,                  # [B, T]    | float
        audio_mask: torch.Tensor,             # [B, T, 1] | True/False
        audio_lengths: torch.Tensor,          # [B]       | int

        phoneme_tokens: torch.Tensor,         # [B, P]    | int
        phoneme_tokens_mask: torch.Tensor,    # [B, P, 1] | True/False
        phoneme_tokens_lengths: torch.Tensor, # [B]       | int

        pitch: torch.Tensor,                  # [B, F]    | float | GT F0 in Hz (0.0 = unvoiced / padding)
    ):
        """
        B: batch size
        T: number of audio samples
        P: seq_len of phonemes
        F: seq_len of frames
        D: hidden_dim
        """

        (audio_encodings,                                           # audio_encodings: [B, F, n_mels]
         frame_mask,                                                # frame_mask: [B, F, 1]
         frame_lengths) = self.log_mel_spectrogram_generator(       # frame_lengths: [B]
             audio,
             audio_lengths
        )
        
        phoneme_encodings = self.phoneme_encoder(           # [B, P, D]
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

        (expanded_phoneme_encodings,                                    # expanded_phoneme_encodings: [B, F, D]
         frame_mask_expanded,                                           # frame_mask_expanded: [B, F, 1]
         frame_lengths_expanded) = self._expand_phoneme_encodings(      # frame_lengths_expanded: [B]
            phoneme_encodings,
            durations,
            max_frames=audio_encodings.shape[1],
        )
        
        audio_latents, audio_latents_lengths = self.encodec.get_latents(audio, audio_lengths) # (B, F, D=128)

        (prompt_latents, prompt_latents_mask, prompt_latents_lengths,           # prompt_latents: [B, Fp, D]   prompt_latents_mask: [B, Fp, 1]   prompt_latents_lengths: [B]
         target_latents, target_latents_mask, target_latents_lengths,           # target_latents: [B, Ft, D]   target_latents_mask: [B, Ft, 1]   target_latents_lengths: [B]
         prompt_starts, prompt_ends,                                            # prompt_starts: [B]           prompt_ends: [B]
         target_idx) = self._generate_prompts_and_targets(                      # target_idx: [B, Ft]          frame indices of the target in the full F axis
            audio_latents,
            audio_latents_lengths,
            self.min_prompt_pct,
            self.max_prompt_pct,
        )

        prompt_encodings = self.speech_prompt_encoder(      # [B, Fp, D]
            prompt_latents,
            prompt_latents_mask,
            prompt_latents_lengths
        )
        prompt_encodings_mask = prompt_latents_mask         # [B, Fp, 1]
        prompt_encodings_lengths = prompt_latents_lengths   # [B]

        predicted_log_durations = self.duration_predictor(  # [B, P]
            phoneme_encodings,
            phoneme_encodings_mask,
            prompt_encodings,
            prompt_encodings_mask
        )

        predicted_log_pitch = self.pitch_predictor(         # [B, F]
            expanded_phoneme_encodings,
            frame_mask_expanded,
            prompt_encodings,
            prompt_encodings_mask,
        )

        condition = self._generate_condition(       # [B, F, D]
            expanded_phoneme_encodings,
            pitch,
            frame_mask_expanded,
        )

        # Slice condition down to the same Ft target frames the diffusion model will denoise
        D = condition.shape[-1]
        condition_target = torch.gather(                                      # [B, Ft, D]
            condition,
            1,
            repeat(target_idx, 'b ft -> b ft d', d=D),
        )
        condition_target = condition_target * target_latents_mask.to(condition_target.dtype)

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

        # Loss is in log-space so errors are scale-symmetric: a 2x overshoot on a 2-frame
        # consonant (perceptually catastrophic) weighs more than a 2x overshoot on a 50-frame
        # vowel (barely audible), whereas linear-space MSE would treat them as equal and let
        # long vowels dominate the gradient. log1p (not log) keeps log(0) → 0 for padded/1-frame
        # phonemes. The network predicts log-duration directly; exp(y)-1 at inference gives frames.
        gt_log_durations = torch.log1p(durations.to(predicted_log_durations.dtype))  # [B, P]
        duration_loss_per_phoneme = F.mse_loss(
            predicted_log_durations,
            gt_log_durations,
            reduction='none',
        )  # [B, P]
        phoneme_mask_flat = rearrange(phoneme_encodings_mask, 'b p 1 -> b p').to(predicted_log_durations.dtype)
        duration_predictor_loss = (duration_loss_per_phoneme * phoneme_mask_flat).sum() / phoneme_mask_flat.sum()

        # Pitch loss
        voiced_mask = (pitch > 0).to(predicted_log_pitch.dtype)                                # [B, F]
        frame_mask_flat = rearrange(frame_mask_expanded, 'b f 1 -> b f').to(predicted_log_pitch.dtype)
        pitch_loss_mask = voiced_mask * frame_mask_flat                                        # [B, F]
        gt_log_pitch = torch.log(pitch.clamp(min=1e-5)).to(predicted_log_pitch.dtype)          # [B, F]
        pitch_loss_per_frame = F.mse_loss(
            predicted_log_pitch,
            gt_log_pitch,
            reduction='none',
        )  # [B, F]
        pitch_predictor_loss = (pitch_loss_per_frame * pitch_loss_mask).sum() / pitch_loss_mask.sum().clamp(min=1.0)

        diffusion_loss = self.diffusion_model(
            target_latents,           # [B, Ft, latent_dim]
            condition_target,         # [B, Ft, D]
            prompt_encodings,         # [B, Fp, D]
            target_latents_mask,      # [B, Ft, 1]
            prompt_encodings_mask,    # [B, Fp, 1]
        )

        return {
            "diffusion_loss": diffusion_loss,
            "duration_predictor_loss": duration_predictor_loss,
            "pitch_predictor_loss": pitch_predictor_loss,
            "aligner_loss":{
                "forward_sum_loss": forward_sum_loss,
                "bin_loss": bin_loss,
            }
        }

    @torch.no_grad()
    def generate(
        self,
        reference_audio: torch.Tensor,           # [B, T_ref]    | reference utterance for speaker/style prompt
        reference_audio_lengths: torch.Tensor,   # [B]

        phoneme_tokens: torch.Tensor,            # [B, P]        | text to synthesize, phonemized + tokenized
        phoneme_tokens_mask: torch.Tensor,       # [B, P, 1]     | True/False
        phoneme_tokens_lengths: torch.Tensor,    # [B]

        num_diffusion_steps: int = 150,
        cfg_scale: float = 1.0,
    ):
        # 1. Build speech prompt from reference audio.
        reference_latents, reference_latents_lengths = self.encodec.get_latents(  # [B, Fp, D], [B]
            reference_audio,
            reference_audio_lengths,
        )
        prompt_latents_mask = create_mask_from_lengths(                           # [B, Fp, 1]
            reference_latents_lengths,
            max_len=reference_latents.shape[1],
        )
        prompt_encodings = self.speech_prompt_encoder(                            # [B, Fp, D]
            reference_latents,
            prompt_latents_mask,
            reference_latents_lengths,
        )
        prompt_encodings_mask = prompt_latents_mask

        # 2. Encode phonemes.
        phoneme_encodings = self.phoneme_encoder(                                 # [B, P, D]
            phoneme_tokens,
            phoneme_tokens_mask,
            phoneme_tokens_lengths,
        )

        # 3. Predict durations, expand phonemes to frames.
        predicted_log_durations = self.duration_predictor(                        # [B, P]
            phoneme_encodings,
            phoneme_tokens_mask,
            prompt_encodings,
            prompt_encodings_mask,
        )
        # Inverse of training's torch.log1p: expm1 -> round -> clamp -> mask to 0 on padding.
        phoneme_mask_flat = rearrange(phoneme_tokens_mask, 'b p 1 -> b p').long()
        predicted_durations = torch.expm1(predicted_log_durations).round().long().clamp(min=0)
        predicted_durations = predicted_durations * phoneme_mask_flat             # [B, P]

        # max_frames as Python int forces one CPU<->GPU sync — acceptable inside generate()
        # (not inside forward()). Required because _expand_phoneme_encodings needs a
        # Python int for torch.arange.
        frame_lengths = predicted_durations.sum(dim=1)                            # [B]
        max_frames = int(frame_lengths.max().item())

        (expanded_phoneme_encodings,                                              # [B, F', D]
         frame_mask,                                                              # [B, F', 1]
         frame_lengths) = self._expand_phoneme_encodings(                         # [B]
            phoneme_encodings,
            predicted_durations,
            max_frames=max_frames,
        )

        # 4. Predict pitch, build condition.
        predicted_log_pitch = self.pitch_predictor(                               # [B, F']
            expanded_phoneme_encodings,
            frame_mask,
            prompt_encodings,
            prompt_encodings_mask,
        )
        # Inverse of training's torch.log(pitch.clamp(min=1e-5)). Unvoiced frames
        # produce very low predicted log-pitch, so exp(.) recovers ~0 Hz naturally.
        predicted_pitch = torch.exp(predicted_log_pitch)                          # [B, F']

        condition = self._generate_condition(                                     # [B, F', D]
            expanded_phoneme_encodings,
            predicted_pitch,
            frame_mask,
        )

        # 5. Diffusion sampling — pending diffusion model implementation.
        # Planned call signature:
        #   generated_latents = self.diffusion_model.sample(
        #       condition=condition,                    # [B, F', D]
        #       condition_mask=frame_mask,              # [B, F', 1]
        #       prompt=prompt_encodings,                # [B, Fp, D]
        #       prompt_mask=prompt_encodings_mask,      # [B, Fp, 1]
        #       num_steps=num_diffusion_steps,
        #       cfg_scale=cfg_scale,
        #   )  # [B, F', D]
        # 6. Decode via self.encodec.decode_from_latents(generated_latents).

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # Start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create optim groups. Tensors that are 2D or higher (Matmuls + Embeddings) decay.
        # 1D tensors (Biases and LayerNorms) do not decay.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Hardcode fused=True
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

    

class LossWrapper(torch.nn.Module):
    def __init__(self, loss_weights: dict, loss_warmup_steps: dict):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss_warmup_steps = loss_warmup_steps
        self.current_weights = {}
        self._update_weights(0)  # Initialize weights for step 0

    def _update_weights(self, step: int):
        for key, target_weight in self.loss_weights.items():
            warmup_steps = self.loss_warmup_steps[key]
            if warmup_steps > 0:
                progress = min(1.0, step / warmup_steps)
                # Linear warm-up from 1/10th of target weight up to the target weight
                self.current_weights[key] = target_weight * (0.1 + 0.9 * progress)
            else:
                self.current_weights[key] = target_weight

    def forward(self, loss_dict: dict, step: int = None):
        total_loss = 0.0
        log_dict = {}
        weighted_tensors = {}
        
        # Update current weights only if step is explicitly passed (train loop)
        if step is not None:
            self._update_weights(step)

        for key, value in loss_dict.items():

            if not isinstance(value, dict):
                weight = self.current_weights[key]
                weighted_loss = value * weight
                total_loss += weighted_loss
                
                log_dict[key] = value.detach().item()
                log_dict[f"{key}_weighted"] = weighted_loss.detach().item()
                weighted_tensors[key] = weighted_loss
            else:
                group_weight = self.current_weights[key]
                group_loss = 0.0
                
                for sub_key, sub_value in value.items():
                    sub_weight = self.current_weights[sub_key]
                    weighted_sub = sub_value * sub_weight
                    group_loss += weighted_sub
                    
                    log_dict[sub_key] = sub_value.detach().item()
                    log_dict[f"{sub_key}_weighted"] = weighted_sub.detach().item()
                    weighted_tensors[sub_key] = weighted_sub * group_weight

                weighted_group = group_loss * group_weight
                total_loss += weighted_group
                log_dict[f"{key}_total_weighted"] = weighted_group.detach().item()

        return total_loss, log_dict, weighted_tensors


class GradientAnalyzer:
    @staticmethod
    def analyze_gradients(model, optimizer, weighted_loss_tensors):
        grad_norms = {}
        grad_vectors = {}
        
        # 1. Isolate and capture gradients for each individual loss term
        for name, loss_tensor in weighted_loss_tensors.items():
            if loss_tensor.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                loss_tensor.backward(retain_graph=True)
                
                norm_sq = 0.0
                vec_dict = {}
                for p_name, p in model.named_parameters():
                    if p.grad is not None:
                        # Calculate norm on GPU for maximum speed
                        norm_sq += torch.sum(p.grad.detach() ** 2).item()
                        # Store copy in system RAM to prevent massive VRAM accumulation
                        vec_dict[p_name] = p.grad.detach().cpu().clone()
                
                grad_norms[name] = math.sqrt(norm_sq)
                grad_vectors[name] = vec_dict
                
        # Clear the grads so the main backward pass can run cleanly
        optimizer.zero_grad(set_to_none=True)
        
        # 2. Compute Cosine Similarities only over dynamically identified shared parameters
        cos_sims = {}
        names = list(grad_vectors.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i = names[i]
                name_j = names[j]
                
                shared_params = set(grad_vectors[name_i].keys()).intersection(set(grad_vectors[name_j].keys()))
                
                if shared_params:
                    dot_product = 0.0
                    norm_i_sq = 0.0
                    norm_j_sq = 0.0
                    
                    for p in shared_params:
                        vi = grad_vectors[name_i][p]
                        vj = grad_vectors[name_j][p]
                        dot_product += torch.sum(vi * vj).item()
                        norm_i_sq += torch.sum(vi ** 2).item()
                        norm_j_sq += torch.sum(vj ** 2).item()
                        
                    if norm_i_sq > 0 and norm_j_sq > 0:
                        sim = dot_product / (math.sqrt(norm_i_sq) * math.sqrt(norm_j_sq))
                        cos_sims[f"{name_i}_vs_{name_j}"] = sim
                    else:
                        cos_sims[f"{name_i}_vs_{name_j}"] = 0.0
                        
        return grad_norms, cos_sims