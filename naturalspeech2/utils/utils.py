import torch
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange

def create_mask_from_lengths(
        lengths: torch.Tensor,   # [B]
        max_len: int
):
    device = lengths.device

    seq_range = torch.arange(max_len, device=device) # creates: [0, 1, 2, ..., max_len-1]
    mask = rearrange(seq_range, 't -> 1 t 1') < rearrange(lengths, 'b -> b 1 1')
    return mask  # [B, T, 1]

class LossWrapper(torch.nn.Module):     # will handle the loss balancing and scheduling later
    def __init__(self):
        super().__init__()

    def forward(self, loss_dict: dict[str, torch.Tensor]):
        # Sum all individual loss terms together
        total_loss = sum(loss_dict.values())
        # Detach and extract scalar items for logging (avoids memory leaks)
        log_dict = {k: v.detach().item() for k, v in loss_dict.items()}
        return total_loss, log_dict