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
            else:
                group_weight = self.current_weights[key]
                group_loss = 0.0
                
                for sub_key, sub_value in value.items():
                    sub_weight = self.current_weights[sub_key]
                    weighted_sub = sub_value * sub_weight
                    group_loss += weighted_sub
                    
                    log_dict[sub_key] = sub_value.detach().item()
                    log_dict[f"{sub_key}_weighted"] = weighted_sub.detach().item()

                weighted_group = group_loss * group_weight
                total_loss += weighted_group
                log_dict[f"{key}_total_weighted"] = weighted_group.detach().item()

        return total_loss, log_dict