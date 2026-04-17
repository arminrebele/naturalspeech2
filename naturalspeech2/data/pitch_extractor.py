import numpy as np
import pyworld

from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH


class PitchExtractor:
    """
    Extracts a frame-level F0 contour from a single raw audio array, aligned to
    the model's frame grid (same hop length as Encodec / the mel spectrogram).

    Intended to run inside the dataset preprocessing step (e.g. `datasets.map`),
    one example at a time, on CPU. The resulting array is cached on disk and
    collated into `[B, F]` alongside the other frame-level fields at batch time.

    Input:
        audio: np.ndarray, shape [T], float32 (or float64), mono.

    Output:
        f0: np.ndarray, shape [F], float32, where F = ceil(T / ENCODER_HOP_LENGTH)
        Unvoiced frames are 0.0; voiced frames are F0 in Hz.
    """

    def __init__(self, sampling_rate: int = 24000, hop_length: int = ENCODER_HOP_LENGTH):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        # Pyworld's frame_period is in milliseconds. Derive it from the hop length
        # so pyworld's frame grid lines up 1:1 with mel/encodec frames.
        self.frame_period_ms = hop_length / sampling_rate * 1000.0    # 13.33 ms for 24000 Hz / 320 hop

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        # Pyworld requires float64, C-contiguous.
        audio = np.ascontiguousarray(audio, dtype=np.float64)

        # DIO = coarse F0 candidate search; StoneMask = refinement step.
        f0_coarse, t = pyworld.dio(
            audio,
            fs=self.sampling_rate,
            frame_period=self.frame_period_ms,
        )
        f0 = pyworld.stonemask(audio, f0_coarse, t, self.sampling_rate)

        # Align to the model's frame grid: F = ceil(T / hop_length).
        # Pyworld's internal formula is floor(T/hop) + 1, so:
        #   - when T is a multiple of hop: pyworld emits ceil(T/hop) + 1  -> crop 1
        #   - when T is not a multiple of hop: pyworld emits ceil(T/hop)  -> no-op
        # The pad branch is a safety net for rare float-rounding edge cases
        # near exact boundaries; in practice we almost always crop by 1.
        target_frames = (audio.shape[0] + self.hop_length - 1) // self.hop_length  # frame count that the mel spectogram and Encodec produce for this audio
        if f0.shape[0] < target_frames:
            f0 = np.pad(f0, (0, target_frames - f0.shape[0]), mode="constant")
        else:
            f0 = f0[:target_frames]

        return f0.astype(np.float32)
