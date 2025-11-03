import torch
from torch.utils.data import DataLoader

import numpy as np
import sounddevice as sd

from audiodeepfakedetection_ddim_inversion.data.vctk import VCTKDataset
from audiodeepfakedetection_ddim_inversion.encodec import EncodecWrapper

def vctk_collate_fn(batch):
    return {
        "audio": [b["audio"] for b in batch],
        "sampling_rate": batch[0]["sampling_rate"],
        "transcript": [b["transcript"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
    }

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = VCTKDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=vctk_collate_fn)

    encodec = EncodecWrapper(device)

    batch = next(iter(loader))

    audio = batch["audio"]
    transcripts = batch["transcript"]

    print(transcripts[0])
    sd.play(audio[0], samplerate=dataset.sampling_rate)
    sd.wait()

    latents = encodec.get_latents(audio)
    decoded_latents = encodec.decode_from_latents(latents)
    decoded_latents_np = decoded_latents.squeeze().detach().cpu().numpy()
    sd.play(decoded_latents_np[0], samplerate=dataset.sampling_rate)
    sd.wait()

    codes, scales = encodec.encode(audio)
    print(codes, "\n\n\n", scales)
    decoded_codes = encodec.decode_from_codes(codes, scales)
    print(decoded_codes)
    decoded_codes_np = decoded_codes.audio_values.squeeze().detach().cpu().numpy()
    sd.play(decoded_codes_np[0], samplerate=dataset.sampling_rate)
    sd.wait()
    
    


if __name__ == "__main__":
    main()