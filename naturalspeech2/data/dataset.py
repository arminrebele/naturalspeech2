from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk, Audio, Value
from einops import rearrange

from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer, build_token_vocabulary
from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper
from naturalspeech2.utils.utils import create_mask_from_lengths


# Global cache for worker processes
_PHONEMIZER_INSTANCE = None

def get_worker_phonemizer():
    global _PHONEMIZER_INSTANCE
    if _PHONEMIZER_INSTANCE is None:
        _PHONEMIZER_INSTANCE = PhonemizerWrapper()
    return _PHONEMIZER_INSTANCE


def phonemize_batch(batch):
    phonemizer = get_worker_phonemizer()
    batch["phonemes"] = [phonemizer(text) for text in batch["text"]]
    return batch

def tokenize_batch(batch, phoneme_tokenizer):
    batch["phoneme_tokens"] = [phoneme_tokenizer(p) for p in batch["phonemes"]]
    return batch

def resample_and_save_audio(sample, target_sr, resampled_dir):
    # Get the resampled audio array.
    resampled_array = sample["audio"]["array"]
    
    # Create a unique path for the new file
    original_path = Path(sample["file"])
    new_filename = f"{original_path.stem}_resampled.flac"
    new_path = resampled_dir / new_filename

    # Save the resampled audio to the new path as FLAC
    tensor = torch.from_numpy(resampled_array)
    tensor = rearrange(tensor, "t -> 1 t")          # torchaudio.save expects [channels, samples]
    torchaudio.save(new_path, tensor, target_sr)
    
    # Avoid type conflicts (dict vs string) during processing.
    return {
        "audio_path": str(new_path),
        "audio_length": tensor.shape[1]
    }

class DatasetWrapper(Dataset):
    def __init__(
        self,
        dataset_source: str,
        dataset_name: str,
        text_column: str = "text",
        audio_column: str = "audio",
        filter_column: str = None,
        filter_substring: str = None,
        token_vocabulary_path: str = None,
        sampling_rate=24000,
        num_proc_phonemize=24,
        num_proc_tokenize=4,
    ):
        super().__init__()
        self.dataset_source = dataset_source
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.audio_column = audio_column
        self.filter_column = filter_column
        self.filter_substring = filter_substring
        
        self.sampling_rate = sampling_rate
        self.num_proc_phonemize = num_proc_phonemize
        self.num_proc_tokenize = num_proc_tokenize
        
        self.dataset_dir = DATA_DIR / self.dataset_name
        self.processed_dir = self.dataset_dir / "processed"
        self.resampled_dir = self.dataset_dir / "resampled"
        self.cache_dir = self.dataset_dir / "cache"
        self.token_vocabulary_path = token_vocabulary_path
        if self.token_vocabulary_path is None:
            self.token_vocabulary_path = self.dataset_dir / "token_vocabulary.json"
        
        self.dataset = self._process_dataset()

    def _process_dataset(self):
        try:
            return load_from_disk(str(self.processed_dir))
        except Exception:
            dataset = load_dataset(self.dataset_source, split="train", cache_dir=str(self.cache_dir))
            # Add index before filtering to keep track of original rows
            dataset = dataset.add_column("original_index", range(len(dataset)))
            
            if self.filter_column and self.filter_substring:
                dataset = dataset.filter(lambda x: self.filter_substring in x, input_columns=[self.filter_column])

            dataset = dataset.rename_column(self.text_column, "text")
            dataset = dataset.rename_column(self.audio_column, "audio")

            # 1. Cast to Audio to leverage HF's on-the-fly resampling during the map operation.
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
            
            # Create the directory for resampled audio
            self.resampled_dir.mkdir(parents=True, exist_ok=True)

            # 2. Map the function to resample and save each audio file to a new location.
            dataset = dataset.map(
                resample_and_save_audio,
                remove_columns=["audio"],                       # Remove original audio dict column
                fn_kwargs={"target_sr": self.sampling_rate, "resampled_dir": self.resampled_dir},
                num_proc=self.num_proc_phonemize,               # Use the phonemize proc count as it's a heavy task
                desc="Resampling and saving audio",
            )
            
            # 3. Rename the path column back to 'audio' and ensure it's treated as a string
            dataset = dataset.rename_column("audio_path", "audio")
            dataset = dataset.cast_column("audio", Value("string"))
            
            dataset = dataset.map(
                phonemize_batch,
                batched=True,
                num_proc=self.num_proc_phonemize,
                desc="Phonemizing transcripts",
            )

            build_token_vocabulary(dataset, save_path=str(self.token_vocabulary_path))
            # Initialize tokenizer without backend so it is picklable
            phoneme_tokenizer = PhonemeTokenizer(phonemizer=None, token_vocabulary_path=str(self.token_vocabulary_path), with_backend=False)
    
            dataset = dataset.map(
                tokenize_batch,
                fn_kwargs={"phoneme_tokenizer": phoneme_tokenizer},
                batched=True,
                num_proc=self.num_proc_tokenize,
                desc="Tokenizing transcripts",
            )

            # Keep only the columns needed for training to save space
            dataset = dataset.select_columns(["audio", "audio_length", "phoneme_tokens", "original_index"])

            self.processed_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(self.processed_dir))

            # Clean up all intermediate cache files created during the process
            print("Cleaning up intermediate cache files...")
            dataset.cleanup_cache_files()
            
            return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Load audio on the fly from the path.
        audio, sr = torchaudio.load(item["audio"])
        return {
            "audio": audio[0], # [T] 
            "phoneme_tokens": item["phoneme_tokens"],
            "original_index": item["original_index"],
            "audio_length": item["audio_length"],
        }

class BucketedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, block_size_multiplier=20):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.block_size_multiplier = block_size_multiplier
        
        # Access lengths directly from the Arrow dataset
        self.lengths = dataset.dataset["audio_length"]

    def __iter__(self):
        # 1. Sort by length
        indices = np.argsort(self.lengths)
        
        # 2. Block/Bucket Shuffle
        if self.shuffle and self.block_size_multiplier > 1:
            block_size = self.batch_size * self.block_size_multiplier
            indices = indices.copy() # Copy to avoid side effects if cached
            for i in range(0, len(indices), block_size):
                end = min(i + block_size, len(indices))
                np.random.shuffle(indices[i:end])

        # 3. Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch.tolist())
        
        # 4. Shuffle the batches order
        if self.shuffle:
            np.random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size if self.drop_last else (len(self.dataset) + self.batch_size - 1) // self.batch_size

def custom_collate_fn(batch, pad_token_id=0):
    # Audio is already a tensor from __getitem__
    audio_tensors = [item["audio"] for item in batch]
    audio_lengths = torch.tensor([item["audio_length"] for item in batch])
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
    
    return {
        "audio": audio_padded,                      # [B, T]
        "audio_mask": audio_mask,                   # [B, T, 1]  
        "audio_lengths": audio_lengths,             # [B]  
        
        "phoneme_tokens": phoneme_tokens_padded,                # [B, P]
        "phoneme_tokens_mask": phoneme_tokens_mask,             # [B, P, 1]
        "phoneme_tokens_lengths": phoneme_tokens_lengths,       # [B]
    }


if __name__ == "__main__":

    # test code to verify dataset loading
    dataset = DatasetWrapper(dataset_source="sanchit-gandhi/vctk", dataset_name="VCTK", filter_column="file", filter_substring="_mic2")
    print(len(dataset))

    sample = dataset[0]
    print(sample)
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio shape: {sample['audio'].shape}")

    # Test Sampler
    sampler = BucketedBatchSampler(dataset, batch_size=4, shuffle=True)
    print(f"Number of batches: {len(sampler)}")
    for batch_indices in sampler:
        print(f"Batch indices: {batch_indices}")
        break