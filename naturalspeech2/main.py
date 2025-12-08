import torch
from torch.utils.data import DataLoader

import numpy as np
import sounddevice as sd
import os

from naturalspeech2.data.vctk import VCTKDataset, vctk_collate_fn
from naturalspeech2.encodec import EncodecWrapper
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataset = VCTKDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=vctk_collate_fn)

    batch = next(iter(loader))

    print("audio: ", batch["audio"], "\n\n")
    print("audio_mask: ", batch["audio_mask"], "\n\n")
    print("audio_lengths: ", batch["audio_lengths"], "\n\n")

    print("phoneme_tokens: ", batch["phoneme_tokens"], "\n\n")
    print("phoneme_tokens_mask: ", batch["phoneme_tokens_mask"], "\n\n")
    print("phoneme_tokens_lengths: ", batch["phoneme_tokens_lengths"], "\n\n")
    
    
    

if __name__ == "__main__":
    main()
    # tokenizer = PhonemeTokenizer()
    # text = "Hello Mr., world! & 2025 This is a test."
    # print(tokenizer(text))
    # print(tokenizer.decode_tokens(tokenizer(text)))