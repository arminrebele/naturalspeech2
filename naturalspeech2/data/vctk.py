import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk, Audio
from einops import rearrange

from naturalspeech2.paths import VCTK_PROCESSED_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer, build_token_vocabulary
from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper
from naturalspeech2.utils.utils import create_mask_from_lengths

def phonemize_text(sample, phonemizer):
    sample["phonemes"] = phonemizer(sample["text"])
    return sample

def tokenize_text(sample, phoneme_tokenizer):
    sample["phoneme_tokens"] = phoneme_tokenizer(sample["phonemes"])
    return sample

class VCTKDataset(Dataset):
    def __init__(self, phonemizer: PhonemizerWrapper = None, sampling_rate=24000, num_proc=4):
        self.sampling_rate = sampling_rate
        self.num_proc = num_proc
        self.phonemizer = phonemizer or PhonemizerWrapper()
        self.dataset = self._process_dataset()

    def _process_dataset(self):
        try:
            return load_from_disk(str(VCTK_PROCESSED_DIR))
        except Exception:
            dataset = load_dataset("sanchit-gandhi/vctk", split="train")                    # !!! default cache_dir
            dataset = dataset.filter(lambda file: "_mic2" in file, input_columns=["file"])
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

            dataset = dataset.map(
                phonemize_text,
                fn_kwargs={"phonemizer": self.phonemizer},
                num_proc=self.num_proc,
                desc="Phonemizing transcripts",
            )

            build_token_vocabulary(dataset)
            phoneme_tokenizer = PhonemeTokenizer(self.phonemizer)
    
            dataset = dataset.map(
                tokenize_text,
                fn_kwargs={"phoneme_tokenizer": phoneme_tokenizer},
                num_proc=self.num_proc,
                desc="Tokenizing transcripts",
            )

            VCTK_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(VCTK_PROCESSED_DIR))
            return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "audio": item["audio"]["array"],
            "text": item["text"],
            "phonemes": item["phonemes"],
            "phoneme_tokens": item["phoneme_tokens"],
            "sampling_rate": item["audio"]["sampling_rate"],
            "speaker_id": item["speaker_id"],
            "age": int(item["age"]), 
            "gender": item["gender"],
            "accent": item["accent"],
            "region": item["region"],
        }

def vctk_collate_fn(batch, pad_token_id=0):

    audio_tensors = [torch.tensor(item["audio"]) for item in batch]
    audio_lengths = torch.tensor([len(tensor) for tensor in audio_tensors])
    audio_padded = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)  # [B, T]

    max_audio_len = audio_padded.shape[1]
    audio_mask = create_mask_from_lengths(audio_lengths, max_audio_len) # [B, T, 1]

    phoneme_tokens_tensors = [torch.tensor(item["phoneme_tokens"]) for item in batch]   # list of tensor with variable length
    phoneme_tokens_lengths = torch.tensor([len(tensor) for tensor in phoneme_tokens_tensors])
    phoneme_tokens_padded= pad_sequence(
        phoneme_tokens_tensors,
        batch_first=True, 
        padding_value=pad_token_id
    )

    max_tokens_len = phoneme_tokens_padded.shape[1]
    phoneme_tokens_mask = create_mask_from_lengths(phoneme_tokens_lengths, max_tokens_len) # [B, P, 1]

    raw_audios = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]
    
    return {
        "audio": audio_padded,                      # [B, T]
        "audio_mask": audio_mask,                   # [B, T, 1]  
        "audio_lengths": audio_lengths,             # [B]  
        
        "phoneme_tokens": phoneme_tokens_padded,                # [B, P]
        "phoneme_tokens_mask": phoneme_tokens_mask,             # [B, P, 1]
        "phoneme_tokens_lengths": phoneme_tokens_lengths,       # [B]

        "texts": texts,
        "raw_audios": raw_audios,
    }


if __name__ == "__main__":

    # test code to verify dataset loading
    dataset = VCTKDataset()
    print(len(dataset))
    sample = dataset[0]
    print(sample)