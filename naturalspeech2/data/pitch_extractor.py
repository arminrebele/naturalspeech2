import numpy as np
import pyworld

from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH


class PitchExtractor:
    """Frame-level F0 contour from raw audio, aligned to the model frame grid (Encodec/mel hop).

    Runs in dataset preprocessing (datasets.map), one example at a time, CPU; cached to disk,
    collated to [B, F] at batch time.

    Input:  audio [T] float32/64 mono.
    Output: f0 [F] float32, F = ceil(T / ENCODER_HOP_LENGTH); unvoiced=0.0, voiced=Hz.
    """

    def __init__(self, sampling_rate: int = 24000, hop_length: int = ENCODER_HOP_LENGTH):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        # frame_period (ms) from hop → pyworld grid lines up 1:1 with mel/encodec frames
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

        # Align to model frame grid F = ceil(T/hop). Pyworld emits floor(T/hop)+1:
        #   T multiple of hop → ceil(T/hop)+1 → crop 1
        #   else              → ceil(T/hop)   → no-op
        # Pad branch = safety net for rare float-rounding at exact boundaries.
        target_frames = (audio.shape[0] + self.hop_length - 1) // self.hop_length  # frame count mel/Encodec produce
        if f0.shape[0] < target_frames:
            f0 = np.pad(f0, (0, target_frames - f0.shape[0]), mode="constant")
        else:
            f0 = f0[:target_frames]

        return f0.astype(np.float32)
