import json
import re
import shutil
import logging
import gc
import io
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
from naturalspeech2.data.phonemizer import PhonemizerWrapper
from naturalspeech2.data.pitch_extractor import PitchExtractor
from naturalspeech2.utils.utils import create_mask_from_lengths
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
import soundfile as sf
import torchaudio.functional as taF

logger = logging.getLogger(__name__)

# Global cache for worker processes
_PHONEMIZER_INSTANCE = None
_PITCH_EXTRACTOR_INSTANCE = None


def _decode_audio_to_target_sr(audio_dict: dict, target_sr: int) -> np.ndarray:
    """HF Audio cell ({bytes,path}) → mono float32 @ target_sr.

    Bypasses datasets 4.x torchcodec auto-decoder (leaks ffmpeg streams under
    fork-based map workers — EAGAIN after ~20k calls). soundfile+torchaudio
    resample is fork-safe, ~4 ms/clip.
    """
    if audio_dict.get("bytes") is not None:
        data, sr = sf.read(io.BytesIO(audio_dict["bytes"]), dtype="float32")
    else:
        data, sr = sf.read(audio_dict["path"], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = taF.resample(torch.from_numpy(data), sr, target_sr).numpy()
    return data

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
        audio_array = _decode_audio_to_target_sr(audio_dict, target_sr)
        lengths.append(audio_array.shape[0])
        f0s.append(pitch_extractor(audio_array))

    return {"audio_length": lengths, "f0": f0s}


def resample_and_save_audio(sample: dict[str, Any], target_sr: int, resampled_dir: Path) -> dict[str, Any]:
    pitch_extractor = get_worker_pitch_extractor(target_sr)
    resampled_array = _decode_audio_to_target_sr(sample["audio"], target_sr)

    # Unique path from dataset index
    new_filename = f"{sample['original_index']}_resampled.flac"
    new_path = resampled_dir / new_filename

    # soundfile expects [frames, channels]; 1D mono matches
    sf.write(new_path, resampled_array, target_sr)

    f0 = pitch_extractor(resampled_array)

    # Avoid dict-vs-string type conflicts downstream.
    return {
        "audio_path": str(new_path),
        "audio_length": resampled_array.shape[0],
        "f0": f0,
    }

def filter_by_substring(texts: list[str], substring: str) -> list[bool]:
    return [substring in text for text in texts]

def filter_audio_lengths(lengths: list[int], min_length: Optional[int], max_length: Optional[int]) -> list[bool]:
    return [
        (min_length is None or l >= min_length) and
        (max_length is None or l <= max_length)
        for l in lengths
    ]

def filter_phoneme_lengths(lengths: list[int], min_length: Optional[int], max_length: Optional[int]) -> list[bool]:
    return [
        (min_length is None or l >= min_length) and
        (max_length is None or l <= max_length)
        for l in lengths
    ]

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
        min_audio_length: Optional[int] = 24000,
        max_audio_length: Optional[int] = None,
        min_phoneme_length: Optional[int] = 1,
        max_phoneme_length: Optional[int] = None,
        sampling_rate: int = 24000,
        resample_on_the_fly: bool = False,
        num_proc_pitch: int = 4,
        num_proc_phonemize: int = 24,
        num_proc_tokenize: int = 4,
        max_train_clips: Optional[int] = None,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.dataset_source = dataset_source
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.audio_column = audio_column
        self.filter_column = filter_column
        self.filter_substring = filter_substring
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.min_phoneme_length = min_phoneme_length
        self.max_phoneme_length = max_phoneme_length

        self.sampling_rate = sampling_rate
        self.resample_on_the_fly = resample_on_the_fly
        self.num_proc_pitch = num_proc_pitch
        self.num_proc_phonemize = num_proc_phonemize
        self.num_proc_tokenize = num_proc_tokenize
        self.max_train_clips = max_train_clips
        self.subset_seed = subset_seed
        
        self.dataset_dir = DATA_DIR / self.dataset_name
        # Split name → valid dir name
        clean_split = self.split.replace("%", "pct")
        clean_split = re.sub(r'[^a-zA-Z0-9]', '_', clean_split)
        clean_split = re.sub(r'_+', '_', clean_split).strip('_')

        # OTF vs pre-resampled cache to distinct dirs
        suffix = "otf" if self.resample_on_the_fly else "pre"
        # Subset caches to its own dir (processed_otf_n<N>) so sizes don't collide
        subset_tag = f"_n{self.max_train_clips}" if self.max_train_clips is not None else ""
        self.processed_dir = self.dataset_dir / clean_split / f"processed_{suffix}{subset_tag}"
        self.resampled_dir = self.dataset_dir / clean_split / "resampled"
        self.cache_dir = self.dataset_dir / clean_split / "cache"
        # Vocab is dataset-level (phoneme inventory is a dataset property, not a split's).
        # First split builds it; others reuse. Manual rebuild if a later split adds phonemes.
        self.token_vocabulary_path = token_vocabulary_path
        if self.token_vocabulary_path is None:
            self.token_vocabulary_path = self.dataset_dir / "token_vocabulary.json"
        
        self.preprocessing_config = {
            "dataset_source": self.dataset_source,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "text_column": self.text_column,
            "audio_column": self.audio_column,
            "filter_column": self.filter_column,
            "filter_substring": self.filter_substring,
            "token_vocabulary_path": str(self.token_vocabulary_path),
            "min_audio_length": self.min_audio_length,
            "max_audio_length": self.max_audio_length,
            "min_phoneme_length": self.min_phoneme_length,
            "max_phoneme_length": self.max_phoneme_length,
            "sampling_rate": self.sampling_rate,
            "resample_on_the_fly": self.resample_on_the_fly,
            "max_train_clips": self.max_train_clips,
            "subset_seed": self.subset_seed if self.max_train_clips is not None else None,
        }

        # Pre-bind get_audio to avoid per-item if/else
        self._get_audio = self._get_audio_on_the_fly if self.resample_on_the_fly else self._get_audio_pre_resampled

        self.dataset = self._process_dataset()

    def _process_dataset(self) -> HFDataset:
        config_path = self.processed_dir / "preprocessing_config.json"

        try:
            dataset = load_from_disk(str(self.processed_dir))
            logger.info(f"Successfully loaded processed dataset from {self.processed_dir}")
            
            if config_path.is_file():
                with open(config_path, "r") as f:
                    cached_config = json.load(f)
                    
                mismatches = []
                for k, v in self.preprocessing_config.items():
                    if k not in cached_config or cached_config[k] != v:
                        mismatches.append(f"{k}: requested={v}, cached={cached_config.get(k)}")
                        
                for k, v in cached_config.items():
                    if k not in self.preprocessing_config:
                        mismatches.append(f"{k}: requested=<removed>, cached={v}")
                
                if mismatches:
                    logger.warning(
                        "\n" + "="*60 + "\n"
                        "⚠️ PREPROCESSING CONFIGURATION MISMATCH ⚠️\n"
                        "The cached dataset was loaded, but it was created with different parameters:\n"
                        + "\n".join(f"  - {m}" for m in mismatches) + "\n\n"
                        f"If you want to apply your new settings, manually delete the cache directory:\n"
                        f"  {self.processed_dir}\n"
                        + "="*60
                    )
            else:
                logger.info("No preprocessing_config.json found in the cache directory. Cannot verify parameters.")

            return dataset
        except (
            FileNotFoundError, 
            json.JSONDecodeError, 
            pa.ArrowInvalid, 
            pa.ArrowIOError
        ) as e:
            logger.info(f"Local dataset unavailable or corrupted ({type(e).__name__}). Triggering preprocessing...")

            logger.info(f"Loading dataset '{self.dataset_name}' with split '{self.split}'...")
            # data_files= restricts the parquet builder to this split's files: HF's
            # `split=` filters post-generation, so without it the builder materializes
            # Arrow for ALL declared splits. Layout follows push_to_hub: data/<split>-*.parquet.
            # verification_mode="no_checks": skip the all-splits-recorded check (else building
            # 1 of the README's 3 splits raises ExpectedMoreSplitsError).
            dataset = load_dataset(
                self.dataset_source,
                data_files={self.split: f"data/{self.split}-*.parquet"},
                split=self.split,
                cache_dir=str(self.cache_dir),
                verification_mode="no_checks",
            )
            # Index before filtering → track original rows
            dataset = dataset.add_column("original_index", range(len(dataset)))
            dataset.cleanup_cache_files()

            # Cap to N clips BEFORE the expensive F0/phonemize/tokenize maps (~1 h vs
            # ~5 days on MLS-train). Seeded shuffle → reproducible, speaker-diverse. Train only.
            if self.max_train_clips is not None and len(dataset) > self.max_train_clips:
                dataset = dataset.shuffle(seed=self.subset_seed).select(range(self.max_train_clips))
                logger.info(
                    f"Subset '{self.split}' to {self.max_train_clips} clips "
                    f"(seeded shuffle, seed={self.subset_seed}) before preprocessing."
                )
            
            if self.filter_column and self.filter_substring:
                dataset = dataset.filter(
                    filter_by_substring,
                    input_columns=[self.filter_column],
                    batched=True,
                    fn_kwargs={"substring": self.filter_substring},
                    num_proc=self.num_proc_tokenize,
                    desc="Filtering by substring",
                )
                dataset.cleanup_cache_files()

            if self.text_column != "text":
                dataset = dataset.rename_column(self.text_column, "text")
            if self.audio_column != "audio":
                dataset = dataset.rename_column(self.audio_column, "audio")

            # Raw bytes (decode=False): datasets 4.x torchcodec leaks ffmpeg streams
            # under fork. Decode manually via _decode_audio_to_target_sr.
            dataset = dataset.cast_column("audio", Audio(decode=False))

            if self.resample_on_the_fly:
                logger.info("`resample_on_the_fly` is True. Extracting F0 and lengths, leaving original bytes intact.")
                dataset = dataset.map(
                    extract_f0_and_metadata_batched,
                    batched=True,
                    fn_kwargs={"target_sr": self.sampling_rate},
                    num_proc=self.num_proc_pitch,
                    desc="Extracting F0 and audio lengths",
                )
                dataset.cleanup_cache_files()
            else:
                logger.info("`resample_on_the_fly` is False. Pre-resampling and saving audio files. Also extracting F0 and lengths.")

                self.resampled_dir.mkdir(parents=True, exist_ok=True)

                # Resample + save each audio file to a new location
                dataset = dataset.map(
                    resample_and_save_audio,
                    remove_columns=["audio"],                       # drop original audio dict column
                    fn_kwargs={"target_sr": self.sampling_rate, "resampled_dir": self.resampled_dir},
                    num_proc=self.num_proc_pitch,                   # Same cost shape as F0 extract (decode + pyworld)
                    desc="Resampling, extracting F0 and lengths, and saving audio",
                )
                
                dataset = dataset.rename_column("audio_path", "audio")

                # Audio column = string path henceforth
                dataset = dataset.cast_column("audio", Value("string"))
                dataset.cleanup_cache_files()
            
            if self.min_audio_length or self.max_audio_length:
                logger.info(f"Filtering audio lengths: min={self.min_audio_length}, max={self.max_audio_length}")
                
                dataset = dataset.filter(
                    filter_audio_lengths,
                    input_columns=["audio_length"],
                    batched=True,
                    fn_kwargs={"min_length": self.min_audio_length, "max_length": self.max_audio_length},
                    num_proc=self.num_proc_tokenize,
                    desc="Filtering by audio length",
                )
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

            if self.min_phoneme_length or self.max_phoneme_length:
                logger.info(f"Filtering phoneme lengths: min={self.min_phoneme_length}, max={self.max_phoneme_length}")
                
                dataset = dataset.filter(
                    filter_phoneme_lengths,
                    input_columns=["phoneme_tokens_length"],
                    batched=True,
                    fn_kwargs={"min_length": self.min_phoneme_length, "max_length": self.max_phoneme_length},
                    num_proc=self.num_proc_tokenize,
                    desc="Filtering by phoneme length",
                )
                dataset.cleanup_cache_files()

            # Keep only training columns to save space. `text` survives for the eval block
            # (Table 2 + overfit_audio_comparison wandb tables: GT transcript vs generated).
            dataset = dataset.select_columns(["audio", "audio_length", "f0", "phoneme_tokens", "phoneme_tokens_length", "original_index", "text"])
            dataset.cleanup_cache_files()

            logger.info(f"Saving processed dataset to {self.processed_dir} ...")
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(self.processed_dir))

            # Save the preprocessing configuration for future cache verification
            with open(config_path, "w") as f:
                json.dump(self.preprocessing_config, f, indent=4)

            logger.info(f"Completely deleting project cache directory: {self.cache_dir}")
            del dataset
            gc.collect()  # release file handles
            try:
                shutil.rmtree(self.cache_dir)
            except Exception as e:
                logger.warning(f"Failed to completely delete cache directory: {e}")
            
            logger.info(f"Loading standalone dataset from {self.processed_dir} into memory...")
            return load_from_disk(str(self.processed_dir))

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_audio_on_the_fly(self, audio_dict: dict[str, Any]) -> torch.Tensor:
        # Column stored decode=False (raw {bytes,path}); decode+resample manually (see above).
        audio_array = _decode_audio_to_target_sr(audio_dict, self.sampling_rate)
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
            "text": item["text"],
        }

class DynamicBucketedBatchSampler(Sampler):
    """Yields dynamic-size batches to maximize VRAM use: group sequences by length,
    look up bucket, chunk by that bucket's batch_size."""
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
        
        # Sort buckets by audio_length ascending
        self.bucket_mapping = sorted(bucket_mapping, key=lambda x: x['audio_length'])

        # Group sequence indices into buckets
        self.bucket_to_indices = {i: [] for i in range(len(self.bucket_mapping))}

        lengths_arr = np.array(dataset.dataset["audio_length"])
        bucket_boundaries = np.array([b['audio_length'] for b in self.bucket_mapping])

        # Assign sequences to buckets
        bucket_indices = np.searchsorted(bucket_boundaries, lengths_arr)

        # Guard: sequences exceeding the max bucket length
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

        # Exact batch count for tqdm / DataLoader len()
        self._num_batches = self._compute_len()

        # Audio throughput stats for logging
        self.expected_batch_audio_samples = 0.0
        self.variance_batch_audio_samples = 0.0
        
        bucket_stats = []
        for b_idx, indices in self.bucket_to_indices.items():
            bs = self.bucket_mapping[b_idx]['batch_size']
            
            if self.drop_last:
                m_i = len(indices) // bs
            else:
                m_i = (len(indices) + bs - 1) // bs
                
            if m_i == 0:
                continue
                
            actual_mean_bs = bs if self.drop_last else len(indices) / m_i
            p_i = m_i / self._num_batches
            bucket_lengths = lengths_arr[indices]
            
            bucket_stats.append({
                'p_i': p_i,
                'E_X_i': actual_mean_bs * np.mean(bucket_lengths),
                'Var_X_i': actual_mean_bs * np.var(bucket_lengths)
            })
            
        # Law of Total Expectation & Law of Total Variance
        mu_X = sum(stat['p_i'] * stat['E_X_i'] for stat in bucket_stats)
        expected_var = sum(stat['p_i'] * stat['Var_X_i'] for stat in bucket_stats)
        var_expected = sum(stat['p_i'] * (stat['E_X_i'] - mu_X)**2 for stat in bucket_stats)
        
        self.expected_batch_audio_samples = mu_X
        self.variance_batch_audio_samples = expected_var + var_expected

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
        
        # Build batches from isolated buckets
        for b_idx, indices in self.bucket_to_indices.items():
            bs = self.bucket_mapping[b_idx]['batch_size']

            # Shuffle within bucket = block randomization
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)

            # Strict chunking → identical batch size
            for i in range(0, len(bucket_indices), bs):
                batch = bucket_indices[i : i + bs]

                if len(batch) == bs:
                    batches.append(batch)
                elif not self.drop_last:
                    # drop_last=False: this final partial batch WILL trigger a graph recompile
                    batches.append(batch)

        # Shuffle global batch order (avoid sequential sizes)
        if self.shuffle:
            rng.shuffle(batches)

        # Fast-forward by slicing
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
    """Collate fn that snaps padding to exact bucket dimensions → minimizes
    torch.compile recompilations."""
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

        pitch_padded = torch.zeros((B, target_pitch_len), dtype=pitch_tensors[0].dtype)
        for i, t in enumerate(pitch_tensors):
            pitch_padded[i, :t.shape[0]] = t

        phoneme_padded = torch.full((B, target_phoneme_len), self.pad_token_id, dtype=phoneme_tokens_tensors[0].dtype)
        for i, t in enumerate(phoneme_tokens_tensors):
            phoneme_padded[i, :t.shape[0]] = t

        phoneme_tokens_mask = create_mask_from_lengths(phoneme_tokens_lengths, target_phoneme_len) # [B, P, 1]
        
        # GT transcripts pass through as list[str] (eval block's Table 2 /
        # overfit_audio_comparison). Not tensor-collatable; training loop ignores it.
        text_list = [item["text"] for item in batch]

        return {
            "audio": audio_padded,                      # [B, static_T]
            "audio_lengths": audio_lengths,             # [B]

            "pitch": pitch_padded,                      # [B, static_F]

            "phoneme_tokens": phoneme_padded,           # [B, static_P]
            "phoneme_tokens_mask": phoneme_tokens_mask, # [B, static_P, 1]
            "phoneme_tokens_lengths": phoneme_tokens_lengths, # [B]

            "text": text_list,                          # list[str] of length B
        }
