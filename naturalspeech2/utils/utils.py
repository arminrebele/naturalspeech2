import torch
from torch.nn.utils.rnn import pad_sequence


def expand_phoneme_encodings(
    phoneme_encodings: torch.Tensor,  # [B, H, P]
    durations: torch.Tensor,          # [B, P]
):
    """
    
    """
    phoneme_encodings = phoneme_encodings.transpose(1, 2)  # [B, P, H]
    B, P, H = phoneme_encodings.shape
    device = phoneme_encodings.device
    durations = durations.to(torch.long)
    
    expanded_phoneme_encodings_list = []
    frame_lengths = []

    for b in range(B):
        durations_b = durations[b]          # [P]
        phoneme_encodings_b = phoneme_encodings[b]    # [P, H]

        expanded_phoneme_encodings_b = torch.repeat_interleave(phoneme_encodings_b, durations_b, dim=0)  # [T_b, H]

        expanded_phoneme_encodings_list.append(expanded_phoneme_encodings_b)
        frame_lengths.append(expanded_phoneme_encodings_b.shape[0])

    frame_lengths = torch.tensor(frame_lengths, device=device, dtype=torch.long)  # [B]

    expanded_phoneme_encodings = pad_sequence(expanded_phoneme_encodings_list, batch_first=True)  # [B, T_max, H]
    F_max = expanded_phoneme_encodings.shape[1]

    frame_idx = torch.arange(F_max, device=device).unsqueeze(0)     # [1, F_max]
    frame_mask = (frame_idx < frame_lengths.unsqueeze(1)).unsqueeze(1)  # [B, 1, F_max]

    expanded_phoneme_encodings = expanded_phoneme_encodings.transpose(1, 2)  # [B, H, F_max]

    return expanded_phoneme_encodings, frame_mask, frame_lengths

