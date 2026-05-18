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
        
        max_allowed_prompt = (frame_lengths - min_target_frames).clamp(min=1)
        prompt_latents_lengths = torch.minimum(max_allowed_prompt, torch.full_like(frame_lengths, prompt_frames))
        target_latents_lengths = frame_lengths - prompt_latents_lengths
        target_frames = target_latents_lengths.sum().item()
        
        valid_scalars = target_frames * latent_dim
        denominators["data_loss"] += valid_scalars
        denominators["score_loss"] += valid_scalars
        denominators["ce_rvq_loss"] += target_frames * ENCODEC_Q
        
    return denominators
