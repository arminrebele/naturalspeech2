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
    format_str: str = "%(asctime)s - %(levelname)s - %(message)s",
    root: bool = False,
) -> None:
    """File handler for `logger`, or the root logger if root=True → captures submodule logs too
    (not just this module's). root=True for a complete run log; False for an isolated one."""
    target = logging.getLogger() if root else logger
    target.setLevel(logging.INFO)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format_str))
    target.addHandler(file_handler)


class _PreInitLogBuffer(logging.Handler):
    """Root handler retaining records (no target) for later replay → wandb Logs tab."""
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def install_prewandb_log_buffer() -> _PreInitLogBuffer:
    """Buffer all pre-wandb-init logs (this script + submodules) on the root logger for later replay
    into the wandb Logs tab. Pair with flush_prewandb_log_buffer after wandb.init()."""
    handler = _PreInitLogBuffer()
    logging.getLogger().addHandler(handler)
    return handler


def flush_prewandb_log_buffer(handler: _PreInitLogBuffer, stream=None) -> None:
    """Detach buffer from root. With `stream` (wandb-hooked stdout) replay records → wandb Logs tab,
    fenced by a header/footer: the replay also echoes to the terminal (stdout is the only channel into
    the Logs tab), so the fence marks it as a one-time recap, not new output. Without `stream`, drop
    them (wandb off). One-shot; format mirrors Hydra's console so the block matches the live stream."""
    logging.getLogger().removeHandler(handler)
    if stream is not None and handler.records:
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        stream.write(f"===== replaying {len(handler.records)} pre-wandb-init log line(s) for the wandb Logs tab =====\n")
        for record in handler.records:
            stream.write(formatter.format(record) + "\n")
        stream.write("===== end pre-wandb-init log replay =====\n")
        stream.flush()
    handler.records.clear()


def create_mask_from_lengths(
        lengths: torch.Tensor,   # [B]
        max_len: int
):
    device = lengths.device

    seq_range = torch.arange(max_len, device=device) # [0..max_len-1]
    mask = rearrange(seq_range, 't -> 1 t 1') < rearrange(lengths, 'b -> b 1 1')
    return mask  # [B, T, 1]


def pack_by_budget(sizes: list[int], budget: int) -> list[list[int]]:
    """Greedy length-sorted bin-packing. sizes[i] = a length proxy in any consistent unit (latent
    frames, audio samples); returns lists of original indices with batch·max_size ≤ budget. Sorting
    keeps each group's lengths close → minimal padding; a lone over-budget item still runs alone."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    groups, cur, cur_max = [], [], 0
    for i in order:
        nm = max(cur_max, sizes[i])
        if cur and (len(cur) + 1) * nm > budget:
            groups.append(cur)
            cur, cur_max = [i], sizes[i]
        else:
            cur.append(i)
            cur_max = nm
    if cur:
        groups.append(cur)
    return groups


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

    # Audio lengths bounded by prev bucket's max; force one to hit the boundary
    audio_lengths = torch.randint(min_audio_samples, audio_samples + 1, (batch_size,), device=device)
    audio_lengths[0] = audio_samples
    idx_a = rearrange(torch.arange(audio_samples, device=device), 't -> 1 t')
    audio_mask_2d = idx_a < rearrange(audio_lengths, 'b -> b 1')

    # Zero-pad outside valid lengths
    audio = torch.randn(batch_size, audio_samples, device=device)
    audio = audio.masked_fill(~audio_mask_2d, 0.0)

    # Phoneme lengths NOT bucket-bounded (fast vs slow speakers) → variance down to half max
    phoneme_tokens_lengths = torch.randint(max(1, phoneme_samples // 2), phoneme_samples + 1, (batch_size,), device=device)
    phoneme_tokens_lengths[0] = phoneme_samples
    idx_p = rearrange(torch.arange(phoneme_samples, device=device), 't -> 1 t')
    phoneme_tokens_mask_2d = idx_p < rearrange(phoneme_tokens_lengths, 'b -> b 1')

    # Zero-pad outside valid lengths (pad_token_id=0)
    phoneme_tokens = torch.randint(0, vocab_size, (batch_size, phoneme_samples), device=device)
    phoneme_tokens = phoneme_tokens.masked_fill(~phoneme_tokens_mask_2d, 0)

    phoneme_tokens_mask = rearrange(phoneme_tokens_mask_2d, 'b t -> b t 1')

    # Dummy pitch (Hz), frame-aligned to mel/encodec grid
    frame_count = (audio_samples + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
    frame_lengths = (audio_lengths + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
    idx_f = rearrange(torch.arange(frame_count, device=device), 't -> 1 t')
    pitch_mask = idx_f < rearrange(frame_lengths, 'b -> b 1')
    pitch = torch.rand(batch_size, frame_count, device=device) * 300.0 + 80.0  # ~80..380 Hz
    pitch = pitch.masked_fill(~pitch_mask, 0.0)

    return {
        "audio": audio,
        "audio_lengths": audio_lengths,
        "phoneme_tokens": phoneme_tokens,
        "phoneme_tokens_mask": phoneme_tokens_mask,
        "phoneme_tokens_lengths": phoneme_tokens_lengths,
        "pitch": pitch,
    }