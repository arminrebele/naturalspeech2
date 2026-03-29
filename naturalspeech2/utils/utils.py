import logging
from pathlib import Path
from typing import Union
import torch
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange


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
