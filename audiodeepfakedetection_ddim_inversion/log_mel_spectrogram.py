import torch
import torch.nn as nn
import torchaudio

class LogMelSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        sampling_rate=24000,
        n_fft=1024,
        hop_length=320,
        n_mels=80,
        f_min=0.0,
        f_max=None
    ):
        super().__init__()
        if f_max is None:
            f_max = sampling_rate // 2

        self.hop_length = hop_length
        self.log_mel_generator = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


    def forward(self, audio, audio_lengths):
        """
        audio: [B, T_max_audio]
        audio_lengths: [B]
        """
        audio_encodings= self.log_mel_generator(audio)
        audio_encodings = self.to_db(audio_encodings)  # [B, dim_audio, F]

        B, _, F = audio_encodings.shape

        device = audio_encodings.device

        frame_lengths = 1 + (audio_lengths // self.hop_length)    # [B]
        frame_lengths = frame_lengths.clamp(min=1, max=F) # number of valid frames

        frame_idx = torch.arange(F, device=device).unsqueeze(0)     # [1, F]
        frame_mask = (frame_idx < frame_lengths.unsqueeze(1)).unsqueeze(1)  # [B, 1, F]

        return audio_encodings, frame_mask, frame_lengths