from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from naturalspeech2.utils.utils import create_mask_from_lengths
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
from naturalspeech2.paths import PROJECT_ROOT

class LogMelSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        sampling_rate=24000,
        n_fft=1024,
        n_mels=80,
        f_min=0.0,
        f_max=None,
        stats_path: Optional[str] = None,
    ):
        super().__init__()
        if f_max is None:
            f_max = sampling_rate // 2

        self.log_mel_generator = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=ENCODER_HOP_LENGTH,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        # Per-channel log-mel normalization buffers. Persistent → a checkpoint is self-contained
        # (stats file only needed at first construction), like the Encodec latent stats. Defaults
        # identity (mean=0, std=1); overwritten from the stats file at construction if given.
        self.register_buffer('mel_mean', torch.zeros(n_mels), persistent=True)
        self.register_buffer('mel_std', torch.ones(n_mels), persistent=True)
        if stats_path is not None:
            self._load_mel_stats(stats_path)

    def _load_mel_stats(self, path: str) -> None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(
                f"Mel stats file not found at {p}. "
                f"Run `python scripts/compute_mel_stats.py` to generate it, "
                f"or set mel.stats_path to null to disable normalization."
            )
        stats = torch.load(p, map_location='cpu', weights_only=True)
        mean = stats['mean'].float()
        std = stats['std'].float()
        n = self.mel_mean.shape[0]
        assert mean.shape == (n,), f"Expected mel mean shape ({n},), got {mean.shape}"
        assert std.shape == (n,), f"Expected mel std shape ({n},), got {std.shape}"
        assert std.min() > 0, f"mel_std must be strictly positive; got min={std.min()}"
        self.mel_mean.copy_(mean)
        self.mel_std.copy_(std)


    def forward(
            self, 
            audio,         # [B, T]  | T = max audio length
            audio_lengths  # [B]
    ):
        audio_encodings= self.log_mel_generator(audio) # [B, n_mels, F+1]

        # MelSpectrogram(center=True) emits 1+T//hop frames, Encodec emits T//hop.
        # Drop trailing centered-padding frame → shared F grid.
        F = audio.shape[-1] // ENCODER_HOP_LENGTH
        audio_encodings = audio_encodings[:, :, :F]

        # Clamp ≥1e-5 → avoid log(0)=-inf on silence
        audio_encodings = torch.clamp(audio_encodings, min=1e-5)
        audio_encodings = self.to_db(audio_encodings)  # [B, n_mels, F]
        audio_encodings = rearrange(audio_encodings, 'b d t -> b t d')  # [B, F, n_mels]

        # Per-channel standardization (z − μ_c)/σ_c. Identity at default buffers. Centers the
        # dB-mel's per-channel envelope (frame-invariant common-mode the aligner's RMSNorm can't
        # remove — no mean subtraction) + equalizes per-channel std, so the squared-L2 attention
        # gets the O(1) zero-mean input its Xavier init assumes. Broadcasts [n_mels] → [B, F, n_mels].
        audio_encodings = (audio_encodings - self.mel_mean) / self.mel_std

        # Ceil division matches Encodec's ceil(T/hop) → mel/latent frames stay aligned.
        frame_lengths = (audio_lengths + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH  # [B]
        frame_lengths = frame_lengths.clamp(min=1, max=F) # number of valid frames

        frame_mask = create_mask_from_lengths(frame_lengths, max_len=F)  # [B, F, 1]

        return audio_encodings, frame_mask, frame_lengths