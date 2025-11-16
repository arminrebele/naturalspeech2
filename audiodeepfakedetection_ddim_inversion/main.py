import torch
from torch.utils.data import DataLoader

import numpy as np
import sounddevice as sd
import os

from audiodeepfakedetection_ddim_inversion.data.vctk import VCTKDataset, vctk_collate_fn
from audiodeepfakedetection_ddim_inversion.encodec import EncodecWrapper
from audiodeepfakedetection_ddim_inversion.data.phoneme_tokenizer import PhonemeTokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = VCTKDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=vctk_collate_fn)

    encodec = EncodecWrapper(device)

    batch = next(iter(loader))
    padded_audio = batch["padded_audio"]
    latents = encodec.get_latents(padded_audio)

    print(latents.shape)
    
    

if __name__ == "__main__":
    #main()
    tokenizer = PhonemeTokenizer()
    text = "Hello Mr., world! & 2025 This is a test."
    print(tokenizer(text))
    print(tokenizer.decode_tokens(tokenizer(text)))