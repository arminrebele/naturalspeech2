import json
import shutil
import logging
import gc
import io
from pathlib import Path
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset, Sampler
import torchaudio
from datasets import load_dataset, load_from_disk, Audio, Value, Dataset as HFDataset
from einops import rearrange
from typing import Any, Optional, Iterator

from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer, build_token_vocabulary
from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper
from naturalspeech2.utils.utils import create_mask_from_lengths

logger = logging.getLogger(__name__)

# Global cache for worker processes
_PHONEMIZER_INSTANCE = None

def get_worker_phonemizer() -> PhonemizerWrapper:
    global _PHONEMIZER_INSTANCE
    if _PHONEMIZER_INSTANCE is None:
        _PHONEMIZER_INSTANCE = PhonemizerWrapper()
    return _PHONEMIZER_INSTANCE


def phonemize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    phonemizer = get_worker_phonemizer()
    batch["phonemes"] = [phonemizer(text) for text in batch["text"]]
    return batch

def tokenize_batch(batch: dict[str, list[Any]], phoneme_tokenizer: PhonemeTokenizer) -> dict[str, list[Any]]:
    batch["phoneme_tokens"] = [phoneme_tokenizer(p) for p in batch["phonemes"]]
    batch["phoneme_tokens_length"] = [len(p) for p in batch["phoneme_tokens"]]
    return batch

def get_audio_metadata_batched(batch: dict[str, list[Any]], target_sr: int) -> dict[str, list[int]]:
    # This function calculates the audio length as if it were resampled.
    # By using torchaudio.info, we only read the file headers, avoiding full decoding.
    lengths = []
    for audio_dict in batch["audio"]:
        # audio_dict["bytes"] contains the raw embedded file bytes from the Parquet/Arrow file
        info = torchaudio.info(io.BytesIO(audio_dict["bytes"]))
        resampled_length = int(info.num_frames * (target_sr / info.sample_rate))
        lengths.append(resampled_length)
        
    return {"audio_length": lengths}


def resample_and_save_audio(sample: dict[str, Any], target_sr: int, resampled_dir: Path) -> dict[str, Any]:
    # Get the resampled audio array.
    resampled_array = sample["audio"]["array"]
    
    # Create a unique path for the new file using the dataset index
    new_filename = f"{sample['original_index']}_resampled.flac"
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
        split: str = "train",
        text_column: str = "text",
        audio_column: str = "audio",
        filter_column: Optional[str] = None,
        filter_substring: Optional[str] = None,
        token_vocabulary_path: Optional[str] = None,
        sampling_rate: int = 24000,
        resample_on_the_fly: bool = False,
        num_proc_phonemize: int = 24,
        num_proc_tokenize: int = 4,
    ):
        super().__init__()
        self.dataset_source = dataset_source
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.audio_column = audio_column
        self.filter_column = filter_column
        self.filter_substring = filter_substring
        
        self.sampling_rate = sampling_rate
        self.resample_on_the_fly = resample_on_the_fly
        self.num_proc_phonemize = num_proc_phonemize
        self.num_proc_tokenize = num_proc_tokenize
        
        self.dataset_dir = DATA_DIR / self.dataset_name
        self.processed_dir = self.dataset_dir / "processed"
        self.resampled_dir = self.dataset_dir / "resampled"
        self.cache_dir = self.dataset_dir / "cache"
        self.token_vocabulary_path = token_vocabulary_path
        if self.token_vocabulary_path is None:
            self.token_vocabulary_path = DATA_DIR / f"{self.dataset_name}_token_vocabulary.json"
        
        # Pre-assign the appropriate get_audio function to avoid if/else overhead in __getitem__
        self._get_audio = self._get_audio_on_the_fly if self.resample_on_the_fly else self._get_audio_pre_resampled

        self.dataset = self._process_dataset()

    def _process_dataset(self) -> HFDataset:
        try:
            dataset = load_from_disk(str(self.processed_dir))
            logger.info(f"Successfully loaded processed dataset from {self.processed_dir}")
            return dataset
        except (
            FileNotFoundError, 
            json.JSONDecodeError, 
            pa.ArrowInvalid, 
            pa.ArrowIOError
        ) as e:
            logger.info(f"Local dataset unavailable or corrupted ({type(e).__name__}). Triggering preprocessing...")

            logger.info(f"Loading dataset '{self.dataset_name}' with split '{self.split}'...")
            dataset = load_dataset(self.dataset_source, split=self.split, cache_dir=str(self.cache_dir))
            # Add index before filtering to keep track of original rows
            dataset = dataset.add_column("original_index", range(len(dataset)))
            dataset.cleanup_cache_files()
            
            if self.filter_column and self.filter_substring:
                dataset = dataset.filter(lambda x: self.filter_substring in x, input_columns=[self.filter_column])
                dataset.cleanup_cache_files()

            dataset = dataset.rename_column(self.text_column, "text")
            dataset = dataset.rename_column(self.audio_column, "audio")

            if self.resample_on_the_fly:
                logger.info("`resample_on_the_fly` is True. Calculating resampled audio lengths without saving.")
                # Tell HF not to decode the array, but keep the raw bytes available
                dataset = dataset.cast_column("audio", Audio(decode=False))
                dataset = dataset.map(
                    get_audio_metadata_batched,
                    batched=True,
                    fn_kwargs={"target_sr": self.sampling_rate},
                    num_proc=self.num_proc_tokenize,
                    desc="Calculating audio lengths",
                )
                dataset.cleanup_cache_files()
            else:
                logger.info("`resample_on_the_fly` is False. Pre-resampling and saving audio files.")
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
                
                # Rename the path column back to 'audio'
                dataset = dataset.rename_column("audio_path", "audio")
                
                # Ensure the audio column is treated as a string path from here on
                dataset = dataset.cast_column("audio", Value("string"))
                dataset.cleanup_cache_files()
            
            dataset = dataset.map(
                phonemize_batch,
                batched=True,
                num_proc=self.num_proc_phonemize,
                desc="Phonemizing transcripts",
            )
            dataset.cleanup_cache_files()

            if not Path(self.token_vocabulary_path).is_file():
                logger.info(f"Token vocabulary not found. Building new token vocabulary at {self.token_vocabulary_path}...")
                build_token_vocabulary(dataset, save_path=str(self.token_vocabulary_path))
            else:
                logger.info(f"Using existing token vocabulary from {self.token_vocabulary_path}.")

            # Initialize tokenizer without backend so it is picklable
            phoneme_tokenizer = PhonemeTokenizer(phonemizer=None, token_vocabulary_path=str(self.token_vocabulary_path), with_backend=False)
    
            dataset = dataset.map(
                tokenize_batch,
                fn_kwargs={"phoneme_tokenizer": phoneme_tokenizer},
                batched=True,
                num_proc=self.num_proc_tokenize,
                desc="Tokenizing transcripts",
            )
            dataset.cleanup_cache_files()

            # Keep only the columns needed for training to save space
            dataset = dataset.select_columns(["audio", "audio_length", "phoneme_tokens", "phoneme_tokens_length", "original_index"])
            dataset.cleanup_cache_files()

            self.processed_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(self.processed_dir))

            logger.info(f"Completely deleting project cache directory: {self.cache_dir}")
            del dataset
            gc.collect()  # Force garbage collection to release file handles
            try:
                shutil.rmtree(self.cache_dir)
            except Exception as e:
                logger.warning(f"Failed to completely delete cache directory: {e}")
            
            logger.info(f"Loading standalone dataset from {self.processed_dir} into memory...")
            return load_from_disk(str(self.processed_dir))

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_audio_on_the_fly(self, audio_dict: dict[str, Any]) -> torch.Tensor:
        # Decode the embedded bytes on the fly
        audio, original_sr = torchaudio.load(io.BytesIO(audio_dict["bytes"]))
        if original_sr != self.sampling_rate:
            # Use torchaudio's functional resample for on-the-fly processing
            audio = torchaudio.functional.resample(audio, orig_freq=original_sr, new_freq=self.sampling_rate)
        return audio[0]

    def _get_audio_pre_resampled(self, audio_path: str) -> torch.Tensor:
        audio, _ = torchaudio.load(audio_path)
        return audio[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        
        audio = self._get_audio(item["audio"])

        return {
            "audio": audio, # [T] 
            "phoneme_tokens": item["phoneme_tokens"],
            "phoneme_tokens_length": item["phoneme_tokens_length"],
            "original_index": item["original_index"],
            "audio_length": audio.shape[-1],
        }

class DynamicBucketedBatchSampler(Sampler):
    """
    A Sampler that yields batches of dynamic sizes to maximize VRAM utilization.
    It groups sequences by length, looks up the corresponding bucket, and chunks
    the dataset using the allowed batch_size for that specific bucket.
    """
    def __init__(
        self, 
        dataset: DatasetWrapper, 
        bucket_mapping: list[dict[str, int]], 
        drop_last: bool = True, 
        shuffle: bool = True,
        seed: int = 42
    ):
        self.dataset = dataset
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_batch_idx = 0
        
        # Ensure buckets are strictly sorted by audio_length from smallest to largest
        self.bucket_mapping = sorted(bucket_mapping, key=lambda x: x['audio_length'])
        
        # Strictly group all sequence indices into their assigned buckets at initialization
        self.bucket_to_indices = {i: [] for i in range(len(self.bucket_mapping))}
        
        # Extract the lengths directly as a NumPy array
        lengths_arr = np.array(dataset.dataset["audio_length"])
        bucket_boundaries = np.array([b['audio_length'] for b in self.bucket_mapping])
        
        # Assignment of all sequences to buckets
        bucket_indices = np.searchsorted(bucket_boundaries, lengths_arr)
        
        # Check for sequences that exceed the maximum bucket length
        invalid_mask = bucket_indices == len(bucket_boundaries)
        if np.any(invalid_mask):
            invalid_idx = np.where(invalid_mask)[0][0]
            max_len = bucket_boundaries[-1]
            raise ValueError(
                f"Sequence at index {invalid_idx} with length {lengths_arr[invalid_idx]} exceeds "
                f"the maximum defined bucket length ({max_len})."
            )
            
        for b_idx in range(len(self.bucket_mapping)):
            self.bucket_to_indices[b_idx] = np.where(bucket_indices == b_idx)[0].tolist()

        # Pre-calculate the exact number of batches for tqdm / DataLoader len()
        self._num_batches = self._compute_len()

    def _compute_len(self) -> int:
        num_batches = 0
        for b_idx, indices in self.bucket_to_indices.items():
            bs = self.bucket_mapping[b_idx]['batch_size']
            if self.drop_last:
                num_batches += len(indices) // bs
            else:
                num_batches += (len(indices) + bs - 1) // bs
        return num_batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = []
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Build batches directly from the isolated buckets
        for b_idx, indices in self.bucket_to_indices.items():
            bs = self.bucket_mapping[b_idx]['batch_size']
            
            # Shuffling within the bucket handles block randomization perfectly
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)
            
            # Strict chunking ensures batch size is absolutely identical
            for i in range(0, len(bucket_indices), bs):
                batch = bucket_indices[i : i + bs]
                
                if len(batch) == bs:
                    batches.append(batch)
                elif not self.drop_last:
                    # Warning: If drop_last=False, this final incomplete batch WILL cause a graph recompile!
                    batches.append(batch)
        
        # Shuffle the global batch order so the model doesn't see sizes sequentially
        if self.shuffle:
            rng.shuffle(batches)
            
        # Instantly fast-forward by slicing the list of batch indices
        batches_to_yield = batches[self.start_batch_idx:]
        for batch in batches_to_yield:
            yield batch

    def __len__(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        
    def set_start_batch_idx(self, batch_idx: int):
        self.start_batch_idx = batch_idx

class BucketedCollateFn:
    """
    A callable Collate Function that receives the bucket mapping. 
    It snaps the padding to exactly the predefined bucket dimensions, drastically
    reducing graph recompilations in torch.compile().
    """
    def __init__(self, bucket_mapping: list[dict[str, int]], pad_token_id: int = 0):
        self.bucket_mapping = sorted(bucket_mapping, key=lambda x: x['audio_length'])
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_tensors = [item["audio"] for item in batch]
        audio_lengths = torch.tensor([item["audio_length"] for item in batch])
        
        phoneme_tokens_tensors = [torch.tensor(item["phoneme_tokens"]) for item in batch]
        phoneme_tokens_lengths = torch.tensor([item["phoneme_tokens_length"] for item in batch])

        batch_max_audio = audio_lengths.max().item()
        batch_max_phoneme = phoneme_tokens_lengths.max().item()

        assigned = False
        for bucket in self.bucket_mapping:
            if batch_max_audio <= bucket['audio_length']:
                target_audio_len = bucket['audio_length']
                target_phoneme_len = bucket['phoneme_length']
                assigned = True
                break
                
        if not assigned:
            max_len = self.bucket_mapping[-1]['audio_length']
            raise ValueError(
                f"Batch contains a sequence of length {batch_max_audio}, "
                f"which exceeds the maximum bucket size ({max_len})."
            )
            
        if batch_max_phoneme > target_phoneme_len:
            raise ValueError(
                f"Batch contains a phoneme sequence of length {batch_max_phoneme}, "
                f"which exceeds the bucket's paired phoneme length ({target_phoneme_len})."
            )

        B = len(batch)

        audio_padded = torch.zeros((B, target_audio_len), dtype=audio_tensors[0].dtype)
        for i, t in enumerate(audio_tensors):
            audio_padded[i, :t.shape[0]] = t

        audio_mask = create_mask_from_lengths(audio_lengths, target_audio_len) # [B, T, 1]

        phoneme_padded = torch.full((B, target_phoneme_len), self.pad_token_id, dtype=phoneme_tokens_tensors[0].dtype)
        for i, t in enumerate(phoneme_tokens_tensors):
            phoneme_padded[i, :t.shape[0]] = t

        phoneme_tokens_mask = create_mask_from_lengths(phoneme_tokens_lengths, target_phoneme_len) # [B, P, 1]
        
        return {
            "audio": audio_padded,                      # [B, static_T]
            "audio_mask": audio_mask,                   # [B, static_T, 1]  
            "audio_lengths": audio_lengths,             # [B]  
            
            "phoneme_tokens": phoneme_padded,           # [B, static_P]
            "phoneme_tokens_mask": phoneme_tokens_mask, # [B, static_P, 1]
            "phoneme_tokens_lengths": phoneme_tokens_lengths, # [B]
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Set logging to INFO to see the processing steps in the terminal
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Initializing DatasetWrapper for VCTK...")
    dataset = DatasetWrapper(
        dataset_source="sanchit-gandhi/vctk", 
        dataset_name="VCTK", 
        split="train",
        text_column="text",
        audio_column="audio",
        filter_column="file", 
        filter_substring="_mic2",
        sampling_rate=24000,
        resample_on_the_fly=False,  # Try True or False!
        num_proc_phonemize=24,
        num_proc_tokenize=4,
    )
    
    logger.info(f"Dataset initialized successfully with length: {len(dataset)}")

    # Test individual sample retrieval
    sample = dataset[0]
    logger.info(f"Sample 0 keys: {sample.keys()}")
    logger.info(f"Sample 0 Audio shape: {sample['audio'].shape}")
    logger.info(f"Sample 0 Phoneme tokens length: {len(sample['phoneme_tokens'])}")

    # Test Sampler and DataLoader
    logger.info("Testing DynamicBucketedBatchSampler and DataLoader...")
    bucket_mapping = [
        {"audio_length": 120000, "phoneme_length": 65, "batch_size": 4},
        {"audio_length": 240000, "phoneme_length": 110, "batch_size": 2},
        {"audio_length": 480000, "phoneme_length": 250, "batch_size": 1}
    ]
    sampler = DynamicBucketedBatchSampler(dataset, bucket_mapping=bucket_mapping, shuffle=True)
    collate_fn = BucketedCollateFn(bucket_mapping=bucket_mapping)
    loader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    for batch in loader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Batched audio shape: {batch['audio'].shape}")
        logger.info(f"Batched audio_mask shape: {batch['audio_mask'].shape}")
        logger.info(f"Batched phoneme_tokens shape: {batch['phoneme_tokens'].shape}")
        break
        
    logger.info("Dataset and DataLoader tests completed successfully! ✅")