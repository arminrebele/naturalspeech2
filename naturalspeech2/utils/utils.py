import torch
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange

def create_mask_from_lengths(
        lengths: torch.Tensor,   # [B]
        max_len: int
):
    device = lengths.device

    seq_range = torch.arange(max_len, device=device)
    mask = rearrange(seq_range, 't -> 1 1 t') < rearrange(lengths, 'b -> b 1 1')
    return mask  # [B, 1, T]
