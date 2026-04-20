import json
import re
import shutil
import logging
import gc
from pathlib import Path

import numpy as np
import pyarrow as pa
import torchaudio
import torch
from torch.utils.data import Dataset, Sampler
from datasets import load_dataset, load_from_disk, Audio, Value, Dataset as HFDataset
from typing import Any, Optional, Iterator

from naturalspeech2.paths import DATA_DIR
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer, build_token_vocabulary
from naturalspeech2.data.phonemizer_wrapper import PhonemizerWrapper
from naturalspeech2.data.pitch_extractor import PitchExtractor
from naturalspeech2.utils.utils import create_mask_from_lengths, setup_file_logger
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
import soundfile as sf

logger = logging.getLogger(__name__)

# Global cache for worker processes
_PHONEMIZER_INSTANCE = None
_PITCH_EXTRACTOR_INSTANCE = None

def get_worker_phonemizer() -> PhonemizerWrapper:
    global _PHONEMIZER_INSTANCE
    if _PHONEMIZER_INSTANCE is None:
        _PHONEMIZER_INSTANCE = PhonemizerWrapper()
    return _PHONEMIZER_INSTANCE

def get_worker_pitch_extractor(sampling_rate: int) -> PitchExtractor:
    global _PITCH_EXTRACTOR_INSTANCE
    if _PITCH_EXTRACTOR_INSTANCE is None:
        _PITCH_EXTRACTOR_INSTANCE = PitchExtractor(sampling_rate=sampling_rate)
    return _PITCH_EXTRACTOR_INSTANCE


def phonemize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    phonemizer = get_worker_phonemizer()
    batch["phonemes"] = [phonemizer(text) for text in batch["text"]]
    return batch

def tokenize_batch(batch: dict[str, list[Any]], phoneme_tokenizer: PhonemeTokenizer) -> dict[str, list[Any]]:
    batch["phoneme_tokens"] = [phoneme_tokenizer(p) for p in batch["phonemes"]]
    batch["phoneme_tokens_length"] = [len(p) for p in batch["phoneme_tokens"]]
    return batch

def extract_f0_and_metadata_batched(batch: dict[str, list[Any]], target_sr: int) -> dict[str, list[Any]]:
    lengths = []
    f0s = []
    pitch_extractor = get_worker_pitch_extractor(target_sr)
    
    for audio_dict in batch["audio"]:
        # HF has automatically decoded and resampled this to target_sr
        audio_array = audio_dict["array"]
        lengths.append(audio_array.shape[0])
        f0s.append(pitch_extractor(audio_array))
        
    return {"audio_length": lengths, "f0": f0s}


def resample_and_save_audio(sample: dict[str, Any], target_sr: int, resampled_dir: Path) -> dict[str, Any]:
    pitch_extractor = get_worker_pitch_extractor(target_sr)
    # HF already resampled this to target_sr and gave us a numpy array!
    resampled_array = sample["audio"]["array"]
    
    # Create a unique path for the new file using the dataset index
    new_filename = f"{sample['original_index']}_resampled.flac"
    new_path = resampled_dir / new_filename

    # Save to disk. soundfile expects [frames, channels], which matches HF's 1D output for mono
    sf.write(new_path, resampled_array, target_sr)
    
    f0 = pitch_extractor(resampled_array)
    
    # Avoid type conflicts (dict vs string) during processing.
    return {
        "audio_path": str(new_path),
        "audio_length": resampled_array.shape[0],
        "f0": f0
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
        # Remove any special characters or numbers to get the base split name (e.g., "train[:50%]" -> "train")
        clean_split = re.sub(r'[^a-zA-Z]', '', self.split)
        
        self.processed_dir = self.dataset_dir / clean_split / "processed"
        self.resampled_dir = self.dataset_dir / clean_split / "resampled"
        self.cache_dir = self.dataset_dir / clean_split / "cache"
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

            if self.text_column != "text":
                dataset = dataset.rename_column(self.text_column, "text")
            if self.audio_column != "audio":
                dataset = dataset.rename_column(self.audio_column, "audio")

            # Globally cast the audio column. This commands HF to decode/resample automatically
            # whenever the column is accessed (in map or dataloader), using safe backends.
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

            if self.resample_on_the_fly:
                logger.info("`resample_on_the_fly` is True. Extracting F0 and lengths, leaving original bytes intact.")
                dataset = dataset.map(
                    extract_f0_and_metadata_batched,
                    batched=True,
                    fn_kwargs={"target_sr": self.sampling_rate},
                    num_proc=self.num_proc_phonemize,
                    desc="Extracting F0 and audio lengths",
                )
                dataset.cleanup_cache_files()
            else:
                logger.info("`resample_on_the_fly` is False. Pre-resampling and saving audio files.")
                
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
            dataset = dataset.select_columns(["audio", "audio_length", "f0", "phoneme_tokens", "phoneme_tokens_length", "original_index"])
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
        # Because we globally cast the column, HF provides the decoded/resampled array seamlessly
        audio_array = audio_dict["array"]
        return torch.tensor(audio_array, dtype=torch.float32)

    def _get_audio_pre_resampled(self, audio_path: str) -> torch.Tensor:
        audio, _ = torchaudio.load(audio_path)
        return audio[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        
        audio = self._get_audio(item["audio"])

        return {
            "audio": audio, # [T] 
            "pitch": torch.tensor(item["f0"], dtype=torch.float32), # [F]
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
        
        pitch_tensors = [item["pitch"] for item in batch]
        
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
        target_pitch_len = (target_audio_len + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH

        audio_padded = torch.zeros((B, target_audio_len), dtype=audio_tensors[0].dtype)
        for i, t in enumerate(audio_tensors):
            audio_padded[i, :t.shape[0]] = t

        audio_mask = create_mask_from_lengths(audio_lengths, target_audio_len) # [B, T, 1]
        
        pitch_padded = torch.zeros((B, target_pitch_len), dtype=pitch_tensors[0].dtype)
        for i, t in enumerate(pitch_tensors):
            pitch_padded[i, :t.shape[0]] = t

        phoneme_padded = torch.full((B, target_phoneme_len), self.pad_token_id, dtype=phoneme_tokens_tensors[0].dtype)
        for i, t in enumerate(phoneme_tokens_tensors):
            phoneme_padded[i, :t.shape[0]] = t

        phoneme_tokens_mask = create_mask_from_lengths(phoneme_tokens_lengths, target_phoneme_len) # [B, P, 1]
        
        return {
            "audio": audio_padded,                      # [B, static_T]
            "audio_mask": audio_mask,                   # [B, static_T, 1]  
            "audio_lengths": audio_lengths,             # [B]  
            
            "pitch": pitch_padded,                      # [B, static_F]
            
            "phoneme_tokens": phoneme_padded,           # [B, static_P]
            "phoneme_tokens_mask": phoneme_tokens_mask, # [B, static_P, 1]
            "phoneme_tokens_lengths": phoneme_tokens_lengths, # [B]
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from naturalspeech2.paths import CONFIG_DIR, PACKAGE_ROOT

    # Set up logging to both terminal and file
    log_dir = PACKAGE_ROOT / "benchmarks"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dataset_test.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    
    setup_file_logger(logger, log_file, mode="w", format_str=format_str)
    
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(sh)

    def test_pipeline(dataset_cfg_name: str, split_override: str = None):
        dataset_cfg = OmegaConf.load(CONFIG_DIR / "dataset" / f"{dataset_cfg_name}.yaml")
        dataloader_cfg = OmegaConf.load(CONFIG_DIR / "dataloader" / "default.yaml")

        # Standardized schema across all configs
        split = split_override if split_override else dataset_cfg.train_split
        is_train_split = (split == dataset_cfg.train_split)

        logger.info(f"--- Testing Pipeline with Dataset: {dataset_cfg.name} (Split: {split}) ---")

        dataset = DatasetWrapper(
            dataset_source=dataset_cfg.source,
            dataset_name=dataset_cfg.name,
            split=split,
            text_column=dataset_cfg.text_column,
            audio_column=dataset_cfg.audio_column,
            filter_column=dataset_cfg.filter_column,
            filter_substring=dataset_cfg.filter_substring,
            token_vocabulary_path=dataset_cfg.token_vocabulary_path,
            sampling_rate=dataloader_cfg.sampling_rate,
            resample_on_the_fly=dataloader_cfg.resample_on_the_fly,
            num_proc_phonemize=dataloader_cfg.num_proc_phonemize,
            num_proc_tokenize=dataloader_cfg.num_proc_tokenize,
        )
        logger.info(f"Dataset initialized successfully with length: {len(dataset)}")

        sample = dataset[0]
        logger.info(f"Sample 0 Audio shape: {sample['audio'].shape}")
        logger.info(f"Sample 0 Phoneme tokens length: {len(sample['phoneme_tokens'])}")

        logger.info("Testing DynamicBucketedBatchSampler and DataLoader...")
        bucket_mapping = OmegaConf.to_container(dataloader_cfg.bucket_mapping, resolve=True)

        sampler = DynamicBucketedBatchSampler(
            dataset, 
            bucket_mapping=bucket_mapping, 
            drop_last=dataloader_cfg.drop_last,
            shuffle=dataloader_cfg.shuffle if is_train_split else False
        )
        collate_fn = BucketedCollateFn(bucket_mapping=bucket_mapping)
        loader = DataLoader(
            dataset, 
            batch_sampler=sampler, 
            collate_fn=collate_fn,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=True
        )
        
        for batch in loader:
            logger.info(f"Batch keys: {list(batch.keys())}")
            logger.info(f"Batched audio shape: {batch['audio'].shape}")
            logger.info(f"Batched audio_mask shape: {batch['audio_mask'].shape}")
            logger.info(f"Batched pitch shape: {batch['pitch'].shape}")
            logger.info(f"Batched phoneme_tokens shape: {batch['phoneme_tokens'].shape}")
            logger.info(f"Batched phoneme_tokens_mask shape: {batch['phoneme_tokens_mask'].shape}")
            break
            
        logger.info(f"{dataset_cfg.name} pipeline test completed successfully! ✅\n")

    # 1. Test VCTK dataset
    test_pipeline("vctk")

    # 2. Test LibriSpeech dataset (validation/dev split)
    librispeech_cfg = OmegaConf.load(CONFIG_DIR / "dataset" / "librispeech.yaml")
    test_pipeline("librispeech", split_override=librispeech_cfg.val_split)

    logger.info("All dataset and dataloader tests completed successfully! 🎉")