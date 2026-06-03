import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from naturalspeech2.utils.utils import create_mask_from_lengths
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH

class LogMelSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        sampling_rate=24000,
        n_fft=1024,
        n_mels=80,
        f_min=0.0,
        f_max=None
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

        # Ceil division matches Encodec's ceil(T/hop) → mel/latent frames stay aligned.
        frame_lengths = (audio_lengths + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH  # [B]
        frame_lengths = frame_lengths.clamp(min=1, max=F) # number of valid frames

        frame_mask = create_mask_from_lengths(frame_lengths, max_len=F)  # [B, F, 1]

        return audio_encodings, frame_mask, frame_lengths