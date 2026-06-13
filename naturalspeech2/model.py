import logging
import math
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
from naturalspeech2.utils.initialization import standard_init

logger = logging.getLogger(__name__)


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
        self.rope_max_seq_len = cfg.rope_max_seq_len   # phoneme/prompt seq ceiling (RoPE cache) — inference-boundary guard

        # Stop-gradient toggles (plain bools → torch.compile specializes the branch, no graph break).
        self.detach_aligner_input = cfg.detach_aligner_input
        self.detach_duration_predictor_input = cfg.detach_duration_predictor_input

        self.encodec = EncodecWrapper(
            bandwidth=cfg.encodec.bandwidth,
            latent_stats_path=cfg.encodec.latent_stats_path,
        )

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

        # blank_logit belongs to ForwardSumLoss (sibling), not AlignerNet — route it out of the spread.
        aligner_kwargs = asdict(cfg.aligner)
        blank_logit = aligner_kwargs.pop("blank_logit")
        self.aligner = Aligner(
            audio_dim=cfg.mel.n_mels,
            hidden_dim=cfg.hidden_dim,
            **aligner_kwargs,
        )

        self.forward_sum_loss = ForwardSumLoss(blank_logit=blank_logit)
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

        # Project per-frame pitch (1 ch) → hidden_dim, added to expanded_phoneme_encodings for condition c.
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

        # duration_ends[b,p] = first frame NOT in phoneme p; non-decreasing → valid for searchsorted.
        duration_ends = durations.cumsum(dim=1)  # [B, P]

        # Frame f belongs to the smallest p with f < duration_ends[b, p]. .contiguous(): einops
        # repeat returns an expanded (broadcast) view → searchsorted warns + copies internally on a
        # non-contiguous value tensor; materialize once here instead.
        frame_positions = repeat(torch.arange(max_frames, device=durations.device), 'f -> b f', b=durations.shape[0]).contiguous()  # [B, F]
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

        # Prompt length: prompt_frames, capped so target keeps min_target_frames.
        # clamp(min=1) for degenerate clips shorter than min_target_frames+1.
        max_allowed_prompt = (audio_latents_lengths - min_target_frames).clamp(min=1)               # [B]
        prompt_latents_lengths = torch.minimum(                                                     # [B] | speech-prompt frame count
            max_allowed_prompt,
            audio_latents_lengths.new_full((B,), prompt_frames),
        )

        # Sample prompt_starts in [0, lengths - prompt_lengths]; one rand per sample (length deterministic).
        rand = torch.rand(B, device=device)                                                         # [B]
        max_starts = audio_latents_lengths - prompt_latents_lengths                                 # [B] | max start so the prompt fits
        prompt_starts = (rand * (max_starts + 1).float()).floor().long()                            # [B] | prompt start frame

        # Extract prompt latents. Buffer width = Python-int constant → bucket-stable for torch.compile.
        max_prompt_len = min(prompt_frames, F)
        j_p = rearrange(torch.arange(max_prompt_len, device=device), 'fp -> 1 fp')                              # [1, Fp] | offsets 0..Fp-1
        prompt_idx = (rearrange(prompt_starts, 'b -> b 1') + j_p).clamp(max=F - 1)                              # [B, Fp] | prompt frame indices
        prompt_latents = torch.gather(audio_latents, 1, repeat(prompt_idx, 'b fp -> b fp d', d=D))              # [B, Fp, D]
        prompt_latents_mask = rearrange(j_p < rearrange(prompt_latents_lengths, 'b -> b 1'), 'b fp -> b fp 1')  # [B, Fp, 1]
        prompt_latents = prompt_latents * prompt_latents_mask.to(prompt_latents.dtype)                          # mask padding frames

        # Extract target latents. Buffer width = Python-int constant → bucket-stable for torch.compile.
        max_target_len = F - max(1, min(prompt_frames, F - min_target_frames))
        j_t = rearrange(torch.arange(max_target_len, device=device), 'ft -> 1 ft')                  # [1, Ft] | offsets 0..Ft-1
        target_idx = (                                                                              # [B, Ft]
            j_t
            + (j_t >= rearrange(prompt_starts, 'b -> b 1')).long()
            * rearrange(prompt_latents_lengths, 'b -> b 1')
        ).clamp(max=F - 1)
                                                                                   
        target_latents = torch.gather(audio_latents, 1, repeat(target_idx, 'b ft -> b ft d', d=D))              # [B, Ft, D]
        target_latents_lengths = audio_latents_lengths - prompt_latents_lengths                                 # [B]
        target_latents_mask = rearrange(j_t < rearrange(target_latents_lengths, 'b -> b 1'), 'b ft -> b ft 1')  # [B, Ft, 1]
        target_latents = target_latents * target_latents_mask.to(target_latents.dtype)

        # Per-quantizer GT codebook indices at target frames. target_idx clamped on padded
        # slots → zero them via mask (index 0 is valid but CE skips padded).
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
            target_latents,               # [B, Ft, D]
            target_latents_mask,          # [B, Ft, 1]
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
        audio_lengths: torch.Tensor,          # [B]       | int

        phoneme_tokens: torch.Tensor,         # [B, P]    | int
        phoneme_tokens_mask: torch.Tensor,    # [B, P, 1] | True/False
        phoneme_tokens_lengths: torch.Tensor, # [B]       | int

        pitch: torch.Tensor,                  # [B, F]    | float | GT F0 in Hz (0.0 = unvoiced / padding)

        return_diffusion_inputs: bool = False, # also return diffusion inputs so a caller can
                                               # re-run sample() with the same condition (overfit diagnostic).
    ):
        """Shapes — B: batch, T: audio samples, P: phonemes, F: frames, D: hidden_dim."""

        (audio_encodings,                                           # audio_encodings: [B, F, n_mels]
         frame_mask,                                                # frame_mask: [B, F, 1]
         frame_lengths) = self.log_mel_spectrogram_generator(       # frame_lengths: [B]
             audio,
             audio_lengths
        )
        
        phoneme_encodings = self.phoneme_encoder(           # [B, P, D]
            phoneme_tokens,
            phoneme_tokens_mask,
        )

        # Stop-gradient (parallel-TTS decoupling): the aligner losses (forward_sum/bin) train the
        # aligner net but not the shared phoneme encoder. The encoder still learns via the diffusion
        # condition path below — _expand_phoneme_encodings consumes the un-detached phoneme_encodings.
        aligner_phoneme_encodings = (
            phoneme_encodings.detach() if self.detach_aligner_input else phoneme_encodings
        )
        durations, path_indices, posterior_label_logprobs, posterior_label_logits = self.aligner(
            audio_encodings,
            frame_mask,
            frame_lengths,
            aligner_phoneme_encodings,
            phoneme_tokens_mask,
            phoneme_tokens_lengths,
        )

        (expanded_phoneme_encodings,                                    # expanded_phoneme_encodings: [B, F, D]
         frame_mask_expanded,                                           # frame_mask_expanded: [B, F, 1]
         _) = self._expand_phoneme_encodings(                           # frame_lengths (3rd return) unused here; generate() consumes it
            phoneme_encodings,
            durations,
            max_frames=audio_encodings.shape[1],
        )
        
        audio_latents, audio_latents_lengths, codebook_indices = self.encodec.get_latents(audio, audio_lengths) # [B, F, latent_dim], [B], [B, F, Q]

        (prompt_latents, prompt_latents_mask,                                           # prompt_latents: [B, Fp, latent_dim]   prompt_latents_mask: [B, Fp, 1]
         target_latents, target_latents_mask,                                           # target_latents: [B, Ft, latent_dim]   target_latents_mask: [B, Ft, 1]
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
        )
        prompt_encodings_mask = prompt_latents_mask         # [B, Fp, 1]

        # Stop-gradient (Glow-TTS sg[·] on the duration input): the duration loss trains the predictor,
        # not the phoneme encoder. prompt_encodings stay attached (speaker-conditional durations).
        duration_phoneme_encodings = (
            phoneme_encodings.detach() if self.detach_duration_predictor_input else phoneme_encodings
        )
        predicted_log_durations = self.duration_predictor(  # [B, P]
            duration_phoneme_encodings,
            phoneme_tokens_mask,
            prompt_encodings,
            prompt_encodings_mask,
        )

        predicted_log_pitch, predicted_voicing_logit = self.pitch_predictor(   # [B, F], [B, F]
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

        # Log-space loss → scale-symmetric: a 2× error on a 2-frame consonant (catastrophic) outweighs
        # a 2× error on a 50-frame vowel (inaudible); linear MSE would let long vowels dominate. log1p
        # keeps log(0)→0 for padded/1-frame phonemes. Net predicts log-duration; exp(y)-1 → frames.
        gt_log_durations = torch.log1p(durations.to(predicted_log_durations.dtype))  # [B, P]
        duration_loss_per_phoneme = F.mse_loss(
            predicted_log_durations,
            gt_log_durations,
            reduction='none',
        )  # [B, P]
        phoneme_mask_flat = rearrange(phoneme_tokens_mask, 'b p 1 -> b p').to(predicted_log_durations.dtype)
        duration_predictor_loss = (duration_loss_per_phoneme * phoneme_mask_flat).sum()

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
        pitch_predictor_loss = (pitch_loss_per_frame * pitch_loss_mask).sum()

        # Voiced/unvoiced BCE over ALL valid frames (not voiced-only). Pitch head is trained
        # voiced-only on F0 value → emits ~speaker-mean F0 on unvoiced frames at inference; this
        # head lets generate() gate those to 0, matching the GT condition (pitch=0 on unvoiced).
        voicing_bce_per_frame = F.binary_cross_entropy_with_logits(
            predicted_voicing_logit, voiced_mask, reduction='none',
        )  # [B, F]
        pitch_voicing_loss = (voicing_bce_per_frame * frame_mask_flat).sum()

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
            "pitch_voicing_loss": pitch_voicing_loss,
            "aligner_loss":{
                "forward_sum_loss": forward_sum_loss,
                "bin_loss": bin_loss,
            }
        }
        if return_diffusion_inputs:
            diffusion_inputs = {
                "target_latents": target_latents,                # [B, Ft, latent_dim] normalized
                "target_latents_mask": target_latents_mask,      # [B, Ft, 1] bool
                "condition_target": condition_target,            # [B, Ft, D]
                "prompt_encodings": prompt_encodings,            # [B, Fp, D]
                "prompt_encodings_mask": prompt_encodings_mask,  # [B, Fp, 1] bool
                "target_codebook_indices": target_codebook_indices,  # [B, Ft, Q] — not re-derivable (random prompt/target split)
            }
            return loss_dict, diffusion_inputs
        return loss_dict

    @staticmethod
    def _cap_durations(durations: torch.Tensor, max_frames_per_phoneme: int | None,
                       on_overflow: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sync-free per-phoneme duration clamp → guards duration-predictor blow-ups (expm1 → runaway
        frames → OOM). Returns (clamped, worst_pre_clamp | None). Clamp is unconditional (no-op below
        cap) so it needs NO sync; the caller reads `worst` folded into the existing max_frames sync and
        then raises/warns. Per-phoneme cap → total bounded by P·cap (no magic number). None = off."""
        if max_frames_per_phoneme is None:
            return durations, None
        assert on_overflow in ("raise", "warn"), f"on_overflow must be 'raise'|'warn', got {on_overflow!r}"
        return durations.clamp(max=max_frames_per_phoneme), durations.max()

    @torch.no_grad()
    def generate(
        self,
        reference_audio: torch.Tensor,           # [B, T_ref]    | reference utterance for speaker/style prompt
        reference_audio_lengths: torch.Tensor,   # [B]

        phoneme_tokens: torch.Tensor,            # [B, P]        | text to synthesize, phonemized + tokenized
        phoneme_tokens_mask: torch.Tensor,       # [B, P, 1]     | True/False

        sampling_steps: int | None = None,       # override the model's configured default for this call

        durations: torch.Tensor | None = None,   # [B, P] teacher-forced GT durations — skips the duration predictor
        pitch: torch.Tensor | None = None,        # [B, F'] teacher-forced GT pitch in Hz — skips the pitch predictor

        max_frames_per_phoneme: int | None = None,  # cap each predicted duration (frames); None = no cap
        on_overflow: str = "raise",                 # cap exceeded → "raise" (abort) | "warn" (clamp + log)
    ):
        # Teacher-forcing contract: pitch lives on the duration-set frame grid (F' = Σ durations),
        # so a caller may supply pitch only alongside durations.
        if pitch is not None and durations is None:
            raise ValueError(
                "generate(pitch=...) requires durations=... — pitch is defined on the "
                "duration-determined frame grid, so it can't match a predicted F'."
            )

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
        )
        prompt_encodings_mask = prompt_latents_mask

        # 2. Encode phonemes.
        phoneme_encodings = self.phoneme_encoder(                                 # [B, P, D]
            phoneme_tokens,
            phoneme_tokens_mask,
        )

        # 3. Durations: predicted, unless teacher-forced GT durations are supplied
        #    (diagnostic / controllable generation — skips the duration predictor).
        if durations is None:
            predicted_log_durations = self.duration_predictor(                    # [B, P]
                phoneme_encodings,
                phoneme_tokens_mask,
                prompt_encodings,
                prompt_encodings_mask,
            )
            # Inverse of training's log1p: expm1 → round → clamp ≥1 frame → mask padding to 0.
            # min=1 (not 0) stops an untrained model collapsing valid phonemes to 0 frames →
            # max_frames=0 → crash downstream.
            phoneme_mask_flat = rearrange(phoneme_tokens_mask, 'b p 1 -> b p').long()
            durations = torch.expm1(predicted_log_durations).round().long().clamp(min=1)
            durations = durations * phoneme_mask_flat                             # [B, P]
            durations, worst_frames = self._cap_durations(durations, max_frames_per_phoneme, on_overflow)
        else:
            durations = durations.long()
            worst_frames = None

        # One CPU↔GPU sync — fine in generate() (eager, not the compiled forward). max_frames as a
        # Python int feeds torch.arange in _expand_phoneme_encodings; the duration-cap check reads
        # `worst` from the SAME sync (one fused .tolist()), so the guard adds no extra drain.
        frame_lengths = durations.sum(dim=1)                                      # [B]
        if worst_frames is None:
            max_frames = int(frame_lengths.max().item())
        else:
            max_frames, worst = torch.stack([frame_lengths.max(), worst_frames]).tolist()
            if worst > max_frames_per_phoneme:
                msg = (f"Duration predictor emitted {worst} frames for a single phoneme (cap "
                       f"{max_frames_per_phoneme}) — under-trained predictor or out-of-distribution text.")
                if on_overflow == "raise":
                    raise RuntimeError(msg + " Aborting generation.")
                logger.warning(msg + " Clamping to cap and continuing.")

        (expanded_phoneme_encodings,                                              # [B, F', D]
         frame_mask,                                                              # [B, F', 1]
         frame_lengths) = self._expand_phoneme_encodings(                         # [B]
            phoneme_encodings,
            durations,
            max_frames=max_frames,
        )

        # 4. Pitch: predicted, unless teacher-forced GT pitch ([B, F'] in Hz) is supplied.
        if pitch is None:
            predicted_log_pitch, predicted_voicing_logit = self.pitch_predictor(  # [B, F'], [B, F']
                expanded_phoneme_encodings,
                frame_mask,
                prompt_encodings,
                prompt_encodings_mask,
            )
            # Gate by predicted voicing: unvoiced → 0 Hz, matching the GT condition (pitch=0 on
            # unvoiced). Pitch head is voiced-only trained → ~speaker-mean F0 on unvoiced = OOD → noise.
            voiced = (torch.sigmoid(predicted_voicing_logit) > 0.5).to(predicted_log_pitch.dtype)
            pitch = torch.exp(predicted_log_pitch) * voiced                       # [B, F']

        condition = self._generate_condition(                                     # [B, F', D]
            expanded_phoneme_encodings,
            pitch,
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
        # T is max-padded; per-sample lengths let callers trim decoder output beyond each item (B>1).
        audio_lengths = frame_lengths * ENCODER_HOP_LENGTH                        # [B]
        return generated_audio, audio_lengths

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Total parameter count."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Optim groups: ≥2D (matmuls, embeddings) decay; 1D (biases, norms) don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

    

class LossWrapper(torch.nn.Module):
    def __init__(self, loss_weights: dict, loss_warmup_steps: dict, loss_warmup_hold_steps: dict):
        super().__init__()
        self.loss_weights = self._flatten_config(loss_weights, "group_weight")
        self.loss_warmup_steps = self._flatten_config(loss_warmup_steps, "group_warmup")
        self.loss_warmup_hold_steps = self._flatten_config(loss_warmup_hold_steps, "group_hold")
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
        # Hold at 0 for `hold` steps, then linear ramp 0→target over `ramp` steps (true-zero hard
        # onset). hold + ramp = step at full weight; both 0 → full weight from step 0.
        for key, target_weight in self.loss_weights.items():
            hold = self.loss_warmup_hold_steps[key]
            ramp = self.loss_warmup_steps[key]
            if step < hold:
                self.current_weights[key] = 0.0
            elif ramp > 0:
                progress = min(1.0, (step - hold) / ramp)
                self.current_weights[key] = target_weight * progress
            else:
                self.current_weights[key] = target_weight

    def forward(self, loss_dict: dict, step: int = None, denominators: dict = None):
        # log_dict values are detached tensors, NOT floats — materializing here would force a
        # CPU↔GPU sync every step. Callers .item() at log/eval time.
        total_loss = 0.0
        log_dict = {}
        weighted_tensors = {}

        # Update weights only if step passed (train loop)
        if step is not None:
            self._update_weights(step)

        for key, value in loss_dict.items():

            if not isinstance(value, dict):
                value = value / max(1, denominators[key])

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
                    sub_value = sub_value / max(1, denominators[sub_key])

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
    # Cross-head contested representation: the modules >1 functional head writes to (the loss-balance metric).
    ENCODER_MODULES = ("phoneme_encoder", "speech_prompt_encoder")

    def __init__(self):
        self.grad_vectors = {}
        
    def extract_gradients(self, model, name):
        self.grad_vectors[name] = {}
            
        for p_name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # .grad → CPU RAM
                self.grad_vectors[name][p_name] = p.grad.detach().cpu()
        
    def compute_metrics(self):
        # Norms decomposed by top-level module (param-name prefix). L2 norms over disjoint param groups
        # compose by root-sum-of-squares, so Total + Encoders are exact roll-ups of the per-module norms.
        # Keys: "Module/{module}/{term}", "Total/{term}", "Encoders/{term}". Cosines stay pairwise over
        # the params two terms actually share (each per-term backward yields .grad only on that term's path).
        grad_norms = {}
        cos_sims = {}

        for name, vec_dict in self.grad_vectors.items():
            module_sq = {}  # module → Σ‖grad‖² over its params
            for p_name, v in vec_dict.items():
                module = p_name.split(".", 1)[0]
                module_sq[module] = module_sq.get(module, 0.0) + float(v.square().sum())
            for module, sq in module_sq.items():
                grad_norms[f"Module/{module}/{name}"] = math.sqrt(sq)
            grad_norms[f"Total/{name}"] = math.sqrt(sum(module_sq.values()))
            grad_norms[f"Encoders/{name}"] = math.sqrt(sum(module_sq.get(m, 0.0) for m in self.ENCODER_MODULES))

        # Cosine sim over the params two terms actually share (intersection of their grad supports).
        names = list(self.grad_vectors.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i, name_j = names[i], names[j]
                shared = sorted(set(self.grad_vectors[name_i]).intersection(self.grad_vectors[name_j]))
                if not shared:
                    continue
                vi = torch.cat([self.grad_vectors[name_i][p].flatten() for p in shared])
                vj = torch.cat([self.grad_vectors[name_j][p].flatten() for p in shared])
                norm_i, norm_j = torch.norm(vi).item(), torch.norm(vj).item()
                dot = torch.dot(vi, vj).item()
                del vi, vj
                cos_sims[f"{name_i}_vs_{name_j}"] = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0.0

        return grad_norms, cos_sims