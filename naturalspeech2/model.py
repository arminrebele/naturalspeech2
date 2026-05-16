from dataclasses import asdict

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from naturalspeech2.config.schema import ModelConfig
from naturalspeech2.modules.encodec import EncodecWrapper, ENCODER_HOP_LENGTH
from naturalspeech2.modules.log_mel_spectrogram import LogMelSpectrogramGenerator
from naturalspeech2.modules.phoneme_encoder import PhonemeEncoder
from naturalspeech2.modules.aligner import Aligner, ForwardSumLoss, BinLoss
from naturalspeech2.modules.speech_prompt_encoder import SpeechPromptEncoder
from naturalspeech2.modules.duration_predictor import DurationPredictor
from naturalspeech2.modules.pitch_predictor import PitchPredictor
from naturalspeech2.modules.diffusion_model import DiffusionModel
from naturalspeech2.modules.layers import Conv1D
from naturalspeech2.utils.utils import create_mask_from_lengths
from naturalspeech2.utils.init import standard_init


class NaturalSpeech2Model(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        token_vocabulary_size: int,
        sampling_rate: int,
    ):
        super().__init__()
        self.prompt_frames = int(cfg.prompt_seconds * sampling_rate / ENCODER_HOP_LENGTH)
        self.min_target_frames = int(cfg.min_target_seconds * sampling_rate / ENCODER_HOP_LENGTH)

        self.encodec = EncodecWrapper(latent_stats_path=cfg.encodec.latent_stats_path)

        self.log_mel_spectrogram_generator = LogMelSpectrogramGenerator(
            sampling_rate=sampling_rate,
            **asdict(cfg.mel),
        )

        self.phoneme_encoder = PhonemeEncoder(
            token_vocabulary_size=token_vocabulary_size,
            hidden_dim=cfg.hidden_dim,
            rope_base=cfg.rope_base,
            rope_max_seq_len=cfg.rope_max_seq_len,
            **asdict(cfg.phoneme_encoder),
        )

        self.aligner = Aligner(
            audio_dim=cfg.mel.n_mels,
            hidden_dim=cfg.hidden_dim,
            **asdict(cfg.aligner),
        )

        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

        self.speech_prompt_encoder = SpeechPromptEncoder(
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            rope_base=cfg.rope_base,
            rope_max_seq_len=cfg.rope_max_seq_len,
            **asdict(cfg.speech_prompt_encoder),
        )

        self.duration_predictor = DurationPredictor(
            hidden_dim=cfg.hidden_dim,
            **asdict(cfg.duration_predictor),
        )

        self.pitch_predictor = PitchPredictor(
            hidden_dim=cfg.hidden_dim,
            **asdict(cfg.pitch_predictor),
        )

        # Projects per-frame pitch (1 channel) up to hidden_dim so it can be
        # added to expanded_phoneme_encodings to form the diffusion condition c.
        self.pitch_projection = Conv1D(1, cfg.hidden_dim, 1)

        self.diffusion_model = DiffusionModel(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            **asdict(cfg.diffusion_model),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Submodules self-init; pitch_projection is the only top-level learnable layer owned here.
        standard_init(self.pitch_projection)

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
        codebook_indices: torch.Tensor,         # [B, F, Q]  | codebook indices per quantizer
        audio_latents_lengths: torch.Tensor,    # [B]
        prompt_frames: int,                     # fixed prompt length in frames (e.g. 225 = 3s at 75 Hz)
        min_target_frames: int,                 # safety net — min target frames preserved on short clips
    ):
        device = audio_latents.device
        B, F, D = audio_latents.shape

        # Per-sample prompt length: fixed prompt_frames frames, capped so the target retains
        # min_target_frames. clamp(min=1) handles degenerate clips shorter than min_target_frames+1.
        max_allowed_prompt = (audio_latents_lengths - min_target_frames).clamp(min=1)               # [B]
        prompt_latents_lengths = torch.minimum(                                                     # [B] | number of frames for the speech prompt
            max_allowed_prompt,
            audio_latents_lengths.new_full((B,), prompt_frames),
        )

        # Sample prompt_starts in [0, lengths - prompt_lengths]. One rand call per sample — length is deterministic.
        rand = torch.rand(B, device=device)                                                         # [B]
        max_starts = audio_latents_lengths - prompt_latents_lengths                                 # [B] | maximum starting index for the speech prompt to ensure it fits within the audio latents
        prompt_starts = (rand * (max_starts + 1).float()).floor().long()                            # [B] | frame index where the prompt starts
        prompt_ends = prompt_starts + prompt_latents_lengths                                        # [B] | frame index where the prompt ends (exclusive)

        # Extract prompt latents. Buffer width is a Python-int constant — bucket-stable for torch.compile.
        max_prompt_len = min(prompt_frames, F)
        j_p = rearrange(torch.arange(max_prompt_len, device=device), 'fp -> 1 fp')                              # [1, Fp] | [0, 1, 2, ..., Fp-1] -> relative offset
        prompt_idx = (rearrange(prompt_starts, 'b -> b 1') + j_p).clamp(max=F - 1)                              # [B, Fp] | frame indices for the prompt in the audio latents
        prompt_latents = torch.gather(audio_latents, 1, repeat(prompt_idx, 'b fp -> b fp d', d=D))              # [B, Fp, D]
        prompt_latents_mask = rearrange(j_p < rearrange(prompt_latents_lengths, 'b -> b 1'), 'b fp -> b fp 1')  # [B, Fp, 1]
        prompt_latents = prompt_latents * prompt_latents_mask.to(prompt_latents.dtype)                          # mask out padding frames in the prompt latents

        # Extract target latents. Buffer width is a Python-int constant — bucket-stable for torch.compile.
        max_target_len = F - max(1, min(prompt_frames, F - min_target_frames))
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

        # Extract per-quantizer GT codebook indices at target frames
        # target_idx was clamped on padded slots -> zero those with the mask; index 0 is valid but CE skips them.
        Q = codebook_indices.shape[2]
        target_codebook_indices = torch.gather(      # [B, Ft, Q]
            codebook_indices,
            1,
            repeat(target_idx, 'b ft -> b ft q', q=Q),
        )
        target_codebook_indices = target_codebook_indices * target_latents_mask.long()

        return (
            prompt_latents,               # [B, Fp, D]
            prompt_latents_mask,          # [B, Fp, 1]
            prompt_latents_lengths,       # [B]
            target_latents,               # [B, Ft, D]
            target_latents_mask,          # [B, Ft, 1]
            target_latents_lengths,       # [B]
            prompt_starts,                # [B]
            prompt_ends,                  # [B]
            target_idx,                   # [B, Ft]
            target_codebook_indices,      # [B, Ft, Q]
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
            phoneme_tokens_lengths,
        )

        durations, path_indices, posterior_label_logprobs, posterior_label_logits = self.aligner(
            audio_encodings,
            frame_mask,
            frame_lengths,
            phoneme_encodings,
            phoneme_tokens_mask,
            phoneme_tokens_lengths,
        )

        (expanded_phoneme_encodings,                                    # expanded_phoneme_encodings: [B, F, D]
         frame_mask_expanded,                                           # frame_mask_expanded: [B, F, 1]
         frame_lengths_expanded) = self._expand_phoneme_encodings(      # frame_lengths_expanded: [B]
            phoneme_encodings,
            durations,
            max_frames=audio_encodings.shape[1],
        )
        
        audio_latents, audio_latents_lengths, codebook_indices = self.encodec.get_latents(audio, audio_lengths) # [B, F, latent_dim], [B], [B, F, Q]

        (prompt_latents, prompt_latents_mask, prompt_latents_lengths,                   # prompt_latents: [B, Fp, latent_dim]   prompt_latents_mask: [B, Fp, 1]   prompt_latents_lengths: [B]
         target_latents, target_latents_mask, target_latents_lengths,                   # target_latents: [B, Ft, latent_dim]   target_latents_mask: [B, Ft, 1]   target_latents_lengths: [B]
         prompt_starts, prompt_ends,                                                    # prompt_starts: [B]                    prompt_ends: [B]
         target_idx,                                                                    # target_idx: [B, Ft]                   frame indices of the target in the full F axis
         target_codebook_indices) = self._generate_prompts_and_targets(                 # target_codebook_indices: [B, Ft, Q]   per-quantizer GT codebook indices at target frames
            audio_latents,
            codebook_indices,
            audio_latents_lengths,
            self.prompt_frames,
            self.min_target_frames,
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
            phoneme_tokens_mask,
            prompt_encodings,
            prompt_encodings_mask,
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
            posterior_label_logits,
            frame_lengths,
            phoneme_tokens_lengths,
        )

        bin_loss = self.bin_loss(
            posterior_label_logprobs,
            path_indices,
            frame_mask,
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
        phoneme_mask_flat = rearrange(phoneme_tokens_mask, 'b p 1 -> b p').to(predicted_log_durations.dtype)
        duration_predictor_loss = (
            (duration_loss_per_phoneme * phoneme_mask_flat).sum()
            / phoneme_mask_flat.sum().clamp_min(1.0)
        )

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

        diffusion_losses = self.diffusion_model(
            target_latents,           # [B, Ft, latent_dim]   in normalized space
            target_latents_mask,      # [B, Ft, 1]
            prompt_encodings,         # [B, Fp, D]
            prompt_encodings_mask,    # [B, Fp, 1]
            condition_target,         # [B, Ft, D]
            target_codebook_indices=target_codebook_indices,           # [B, Ft, Q]    | GT codebook indices per quantizer
            codebook_embeddings=self.encodec.codebook_embeddings,       # [Q, K, latent_dim] | raw codebook vectors
            latent_mean=self.encodec.latent_mean,                       # [latent_dim] | unnormalize ẑ₀ → raw before CE-RVQ
            latent_std=self.encodec.latent_std,                         # [latent_dim]
        )

        loss_dict = {
            "diffusion_loss": diffusion_losses,
            "duration_predictor_loss": duration_predictor_loss,
            "pitch_predictor_loss": pitch_predictor_loss,
            "aligner_loss":{
                "forward_sum_loss": forward_sum_loss,
                "bin_loss": bin_loss,
            }
        }
        return loss_dict

    @torch.no_grad()
    def generate(
        self,
        reference_audio: torch.Tensor,           # [B, T_ref]    | reference utterance for speaker/style prompt
        reference_audio_lengths: torch.Tensor,   # [B]

        phoneme_tokens: torch.Tensor,            # [B, P]        | text to synthesize, phonemized + tokenized
        phoneme_tokens_mask: torch.Tensor,       # [B, P, 1]     | True/False
        phoneme_tokens_lengths: torch.Tensor,    # [B]

        sampling_steps: int | None = None,       # override the model's configured default for this call
    ):
        # 1. Build speech prompt from reference audio.
        reference_latents, reference_latents_lengths, _ = self.encodec.get_latents(  # [B, Fp, D], [B]
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
        # Inverse of training's torch.log1p: expm1 -> round -> clamp valid phonemes to >=1 frame -> mask padding to 0.
        # min=1 (not 0) prevents an untrained / early-checkpoint model from collapsing valid phonemes
        # to zero frames, which would produce max_frames=0 and crash the downstream pitch/diffusion/decode path.
        phoneme_mask_flat = rearrange(phoneme_tokens_mask, 'b p 1 -> b p').long()
        predicted_durations = torch.expm1(predicted_log_durations).round().long().clamp(min=1)
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

        # 5. Diffusion sampling.
        generated_latents = self.diffusion_model.sample(                          # [B, F', latent_dim]
            condition=condition,
            condition_mask=frame_mask,
            prompt_encodings=prompt_encodings,
            prompt_encodings_mask=prompt_encodings_mask,
            sampling_steps=sampling_steps,
        )

        # 6. Decode latents to waveform.
        generated_audio = self.encodec.decode_from_latents(generated_latents)     # [B, 1, T]
        generated_audio = rearrange(generated_audio, 'b 1 t -> b t')              # [B, T]
        # T is the max-padded length; per-sample valid lengths let callers trim
        # away decoder output beyond each item's frame_lengths (matters for B>1).
        audio_lengths = frame_lengths * ENCODER_HOP_LENGTH                        # [B]
        return generated_audio, audio_lengths

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
        self.loss_weights = self._flatten_config(loss_weights, "group_weight")
        self.loss_warmup_steps = self._flatten_config(loss_warmup_steps, "group_warmup")
        self.current_weights = {}
        self._update_weights(0)  # Initialize weights for step 0

    def _flatten_config(self, config: dict, group_key_name: str) -> dict:
        """Flattens a nested config dict so group keys and sub-keys share a flat namespace."""
        flat = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key == group_key_name:
                        flat[key] = sub_value
                    else:
                        flat[sub_key] = sub_value
            else:
                flat[key] = value
        return flat

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
        # log_dict values are detached tensors, NOT Python floats — materialising
        # to floats here would force a CPU↔GPU sync at training-step frequency
        # even when the consumer is only logging every log_interval steps.
        # Callers .item() at log/eval time (see scripts/train.py).
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

                log_dict[key] = value.detach()
                log_dict[f"{key}_weighted"] = weighted_loss.detach()
                weighted_tensors[key] = weighted_loss
            else:
                group_weight = self.current_weights[key]
                group_loss = 0.0

                for sub_key, sub_value in value.items():
                    sub_weight = self.current_weights[sub_key]
                    weighted_sub = sub_value * sub_weight
                    group_loss += weighted_sub

                    log_dict[sub_key] = sub_value.detach()
                    log_dict[f"{sub_key}_weighted"] = weighted_sub.detach()
                    weighted_tensors[sub_key] = weighted_sub * group_weight

                weighted_group = group_loss * group_weight
                total_loss += weighted_group
                log_dict[f"{key}_total_weighted"] = weighted_group.detach()

        return total_loss, log_dict, weighted_tensors


class GradientAnalyzer:
    def __init__(self):
        self.grad_vectors = {}
        
    def extract_gradients(self, model, name):
        self.grad_vectors[name] = {}
            
        for p_name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # Extract standard .grad to CPU RAM 
                self.grad_vectors[name][p_name] = p.grad.detach().cpu()
        
    def compute_metrics(self):
        grad_norms = {}
        cos_sims = {}
        
        # Compute Total Norms
        for name, vec_dict in self.grad_vectors.items():
            if vec_dict:
                v_flat = torch.cat([v.flatten() for v in vec_dict.values()])
                grad_norms[f"{name}_total"] = torch.norm(v_flat).item()
                del v_flat
            else:
                grad_norms[f"{name}_total"] = 0.0
            
        # Compute Cosine Similarities only over dynamically identified shared parameters
        names = list(self.grad_vectors.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i = names[i]
                name_j = names[j]
                
                shared_params = sorted(list(set(self.grad_vectors[name_i].keys()).intersection(set(self.grad_vectors[name_j].keys()))))
                
                if shared_params:
                    # Vectorize parameter arithmetic by concatenating all shared parameters
                    # This is vastly faster on CPU caches than iterating over dictionaries
                    vi_flat = torch.cat([self.grad_vectors[name_i][p].flatten() for p in shared_params])
                    vj_flat = torch.cat([self.grad_vectors[name_j][p].flatten() for p in shared_params])
                    
                    norm_i = torch.norm(vi_flat).item()
                    norm_j = torch.norm(vj_flat).item()
                    dot_product = torch.dot(vi_flat, vj_flat).item()
                    
                    del vi_flat, vj_flat
                    
                    # Log norms computed STRICTLY over the shared backbone
                    grad_norms[f"{name_i}_shared_with_{name_j}"] = norm_i
                    grad_norms[f"{name_j}_shared_with_{name_i}"] = norm_j

                    if norm_i > 0 and norm_j > 0:
                        sim = dot_product / (norm_i * norm_j)
                        cos_sims[f"{name_i}_vs_{name_j}"] = sim
                    else:
                        cos_sims[f"{name_i}_vs_{name_j}"] = 0.0
                        
        return grad_norms, cos_sims