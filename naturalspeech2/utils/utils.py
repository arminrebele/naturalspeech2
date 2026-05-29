import logging
from pathlib import Path
from typing import Union
import torch
from omegaconf import DictConfig
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH

from einops import rearrange

ENCODEC_Q = 32 # Encodec 24kHz utilizes 32 quantizers


def setup_file_logger(
    logger: logging.Logger,
    log_file: Union[str, Path],
    mode: str = "a",
    format_str: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> None:
    """Configures a file handler for the given logger."""
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def create_mask_from_lengths(
        lengths: torch.Tensor,   # [B]
        max_len: int
):
    device = lengths.device

    seq_range = torch.arange(max_len, device=device) # creates: [0, 1, 2, ..., max_len-1]
    mask = rearrange(seq_range, 't -> 1 t 1') < rearrange(lengths, 'b -> b 1 1')
    return mask  # [B, T, 1]


def compute_denominators(micro_batches: list[dict], cfg: DictConfig) -> dict:
    denominators = {
        "duration_predictor_loss": 0,
        "pitch_predictor_loss": 0,
        "pitch_voicing_loss": 0,
        "forward_sum_loss": 0,
        "bin_loss": 0,
        "data_loss": 0,
        "score_loss": 0,
        "ce_rvq_loss": 0,
    }
    
    hop = ENCODER_HOP_LENGTH
    sr = cfg.dataloader.sampling_rate
    prompt_frames = int(cfg.model.prompt_seconds * sr / hop)
    min_target_frames = int(cfg.model.min_target_seconds * sr / hop)
    latent_dim = cfg.model.latent_dim
    
    for batch in micro_batches:
        denominators["duration_predictor_loss"] += batch["phoneme_tokens_lengths"].sum().item()
        denominators["pitch_predictor_loss"] += (batch["pitch"] > 0).sum().item()
            
        audio_lengths = batch["audio_lengths"]
        frame_lengths = (audio_lengths + hop - 1) // hop
        total_frames = frame_lengths.sum().item()
        
        denominators["forward_sum_loss"] += total_frames
        denominators["bin_loss"] += total_frames
        denominators["pitch_voicing_loss"] += total_frames   # BCE over all valid frames
        
        max_allowed_prompt = (frame_lengths - min_target_frames).clamp(min=1)
        prompt_latents_lengths = torch.minimum(max_allowed_prompt, torch.full_like(frame_lengths, prompt_frames))
        target_latents_lengths = frame_lengths - prompt_latents_lengths
        target_frames = target_latents_lengths.sum().item()
        
        valid_scalars = target_frames * latent_dim
        denominators["data_loss"] += valid_scalars
        denominators["score_loss"] += valid_scalars
        denominators["ce_rvq_loss"] += target_frames * ENCODEC_Q
        
    return denominators


def generate_dummy_batch(
    batch_size: int,
    audio_samples: int,
    phoneme_samples: int,
    min_audio_samples: int,
    vocab_size: int,
    device: str
) -> dict[str, torch.Tensor]:

    """Generates dummy tensors representing perfectly bucketed sequences."""

    # Audio lengths are strictly bounded by the previous bucket's maximum
    audio_lengths = torch.randint(min_audio_samples, audio_samples + 1, (batch_size,), device=device)
    # Force at least one sequence to hit the max bucket boundary
    audio_lengths[0] = audio_samples
    idx_a = rearrange(torch.arange(audio_samples, device=device), 't -> 1 t')
    audio_mask_2d = idx_a < rearrange(audio_lengths, 'b -> b 1')

    # Generate audio and apply zero-padding outside valid lengths
    audio = torch.randn(batch_size, audio_samples, device=device)
    audio = audio.masked_fill(~audio_mask_2d, 0.0)

    audio_mask = rearrange(audio_mask_2d, 'b t -> b t 1')

    # Phoneme lengths are NOT strictly bounded by previous buckets (fast vs slow speakers)
    # So we maintain a generous variance down to half the bucket's max length
    phoneme_tokens_lengths = torch.randint(max(1, phoneme_samples // 2), phoneme_samples + 1, (batch_size,), device=device)
    phoneme_tokens_lengths[0] = phoneme_samples
    idx_p = rearrange(torch.arange(phoneme_samples, device=device), 't -> 1 t')
    phoneme_tokens_mask_2d = idx_p < rearrange(phoneme_tokens_lengths, 'b -> b 1')

    # Generate tokens and apply zero-padding outside valid lengths (matching pad_token_id=0)
    phoneme_tokens = torch.randint(0, vocab_size, (batch_size, phoneme_samples), device=device)
    phoneme_tokens = phoneme_tokens.masked_fill(~phoneme_tokens_mask_2d, 0)

    phoneme_tokens_mask = rearrange(phoneme_tokens_mask_2d, 'b t -> b t 1')

    # Dummy pitch in Hz, frame-aligned to the mel/encodec grid
    frame_count = (audio_samples + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
    frame_lengths = (audio_lengths + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
    idx_f = rearrange(torch.arange(frame_count, device=device), 't -> 1 t')
    pitch_mask = idx_f < rearrange(frame_lengths, 'b -> b 1')
    pitch = torch.rand(batch_size, frame_count, device=device) * 300.0 + 80.0  # ~80..380 Hz
    pitch = pitch.masked_fill(~pitch_mask, 0.0)

    return {
        "audio": audio,
        "audio_mask": audio_mask,
        "audio_lengths": audio_lengths,
        "phoneme_tokens": phoneme_tokens,
        "phoneme_tokens_mask": phoneme_tokens_mask,
        "phoneme_tokens_lengths": phoneme_tokens_lengths,
        "pitch": pitch,
    }