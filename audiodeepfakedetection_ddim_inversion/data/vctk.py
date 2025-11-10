import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk, Audio, Sequence, Value

from audiodeepfakedetection_ddim_inversion.paths import VCTK_DIR, VCTK_PROCESSED_DIR
from audiodeepfakedetection_ddim_inversion.data.phoneme_tokenizer import PhonemeTokenizer

def tokenize_text(texts, phoneme_tokenizer):
    # texts is list of strings (batched=True + input_columns=["text"])
    return {"token_ids": [phoneme_tokenizer(text) for text in texts]}

class VCTKDataset(Dataset):
    def __init__(self, phoneme_tokenizer: PhonemeTokenizer = None, sampling_rate=24000):
        self.sampling_rate = sampling_rate
        self.phoneme_tokenizer = phoneme_tokenizer or PhonemeTokenizer()
        self.dataset = self._process_dataset()

    def _process_dataset(self):
        try:
            return load_from_disk(str(VCTK_PROCESSED_DIR))
        except Exception:
            VCTK_DIR.mkdir(parents=True, exist_ok=True)

            dataset = load_dataset("sanchit-gandhi/vctk", split="train", cache_dir=str(VCTK_DIR))
            dataset = dataset.filter(lambda file: "_mic2" in file, input_columns=["file"])
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

            new_features = dataset.features.copy()
            new_features["token_ids"] = Sequence(Value("int32"))
            dataset = dataset.map(
                tokenize_text,
                fn_kwargs={"phoneme_tokenizer": self.phoneme_tokenizer},
                input_columns=["text"],
                batched=True,
                batch_size=256,
                num_proc=4,
                features=new_features,
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
            "token_ids": item["token_ids"],
            "sampling_rate": item["audio"]["sampling_rate"],
            "speaker_id": item["speaker_id"],
            "age": int(item["age"]), 
            "gender": item["gender"],
            "accent": item["accent"],
            "region": item["region"],
        }

def vctk_collate_fn(batch, pad_token_id=0):

    token_seq_tensors = [torch.tensor(item["token_ids"]) for item in batch]   # list of tensor with variable length
    token_seq_lengths = torch.tensor([len(tensor) for tensor in token_seq_tensors])
    padded_token_seq = pad_sequence(
        token_seq_tensors,
        batch_first=True, 
        padding_value=pad_token_id
    )
    max_token_seq_len = padded_token_seq.shape[1]
    token_mask = (torch.arange(max_token_seq_len).unsqueeze(0) < token_seq_lengths.unsqueeze(1)).unsqueeze(1)

    audio_tensors = [torch.tensor(item["audio"]) for item in batch]
    audio_lengths = torch.tensor([len(tensor) for tensor in audio_tensors])
    padded_audio = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
    max_audio_len = padded_audio.shape[1]
    audio_mask = (torch.arange(max_audio_len).unsqueeze(0) < audio_lengths.unsqueeze(1)).unsqueeze(1)

    texts = [item["text"] for item in batch]
    raw_audios = [item["audio"] for item in batch]
    
    return {
        "padded_audio": padded_audio,               # [B, T_max_audio]
        "padded_token_seq": padded_token_seq,       # [B, T_max_token_seq]
        "audio_mask": audio_mask,                   # [B, 1, T_max_audio]    
        "token_mask": token_mask,                   # [B, 1, T_max_token_seq]
        "audio_lengths": audio_lengths,             # [B]
        "token_lengths": token_seq_lengths,         # [B]
        "text": texts,
        "raw_audio": raw_audios,
    }

if __name__ == "__main__":

    # test code to verify dataset loading
    dataset = VCTKDataset()
    print(len(dataset))
    sample = dataset[0]
    print(sample)