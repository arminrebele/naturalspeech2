import json
import re
import os
import math
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
from datasets import load_dataset, load_from_disk, concatenate_datasets, Audio, Value, Dataset as HFDataset
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

def add_phoneme_length(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    # phoneme_length == len(phonemes) == len(phoneme_tokens) (tokenization is 1:1) → vocab-free,
    # so it can drive the phoneme-length filter and bucketing without a tokenizer.
    batch["phoneme_length"] = [len(p) for p in batch["phonemes"]]
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
        delete_raw_cache_after_preprocess: bool = False,
        chunk_size: Optional[int] = None,
        build_vocabulary: bool = False,
        build_if_missing: bool = True,
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
        self.delete_raw_cache_after_preprocess = delete_raw_cache_after_preprocess
        self.chunk_size = chunk_size
        self.build_vocabulary = build_vocabulary
        # Read-only guard: when False, raise instead of preprocessing if the cache is absent — lets a
        # pure consumer (the dataloader benchmark) assert the split is already preprocessed rather than
        # silently kicking off a build.
        self.build_if_missing = build_if_missing
        if self.chunk_size is not None and not self.resample_on_the_fly:
            raise ValueError(
                "Chunked storage (chunk_size) requires resample_on_the_fly=True; the pre-resampled "
                "flac path is not supported with chunking."
            )

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
        # Chunked Stage-A store (train split only; vocab-free shards of a seeded permutation).
        self.chunks_dir = self.dataset_dir / clean_split / f"chunks_{suffix}"
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

        # Vocab is decoupled from the cache: the train split builds it from `phonemes`; every split
        # tokenizes lazily in __getitem__ (phonemes → IDs). Swapping the vocab needs no reprocessing.
        self._ensure_token_vocabulary()
        self.tokenizer = PhonemeTokenizer(
            phonemizer=None, token_vocabulary_path=str(self.token_vocabulary_path), with_backend=False,
        )
        assert "<unk>" in self.tokenizer.token_vocabulary, (
            f"'<unk>' missing from {self.token_vocabulary_path}; OOV phonemes would map to None and crash collate."
        )

    def _process_dataset(self) -> HFDataset:
        # chunk_size is passed (by loaders.py) only for the train split → chunked store.
        # dev/test (and the legacy non-chunked path) go through the single-folder path.
        if self.chunk_size:
            return self._process_train_chunked()
        return self._process_single()

    def _load_dataset_split(self) -> HFDataset:
        logger.info(f"Loading dataset '{self.dataset_name}' split '{self.split}'...")
        # data_files= restricts the parquet builder to this split (HF `split=` filters
        # post-generation → would materialize Arrow for ALL declared splits). verification_mode
        # skips the all-splits-recorded check when building 1 of N declared splits.
        return load_dataset(
            self.dataset_source,
            data_files={self.split: f"data/{self.split}-*.parquet"},
            split=self.split,
            cache_dir=str(self.cache_dir),
            verification_mode="no_checks",
        )

    def _preprocess_untokenized(self, dataset: HFDataset) -> HFDataset:
        """The expensive, vocab-INDEPENDENT preprocessing: rename → (substring filter) →
        decode/resample → F0 + audio-length filter → phonemize → phoneme_length + phoneme-length
        filter → select the cacheable schema. Stops BEFORE turning phonemes into integer token IDs
        (that happens lazily in __getitem__, so a vocab change never invalidates this cache). Input
        must already carry `original_index` and be subset-selected. Shared by the chunked (train)
        and single (dev/test) paths so the on-disk schema can't drift. Output columns:
        [audio, audio_length, f0, phonemes, phoneme_length, original_index, text]."""
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

        # transcript→text / →audio renames are LOAD-BEARING (MLS text_column="transcript").
        if self.text_column != "text":
            dataset = dataset.rename_column(self.text_column, "text")
        if self.audio_column != "audio":
            dataset = dataset.rename_column(self.audio_column, "audio")

        # Raw bytes (decode=False): datasets 4.x torchcodec leaks ffmpeg streams under fork.
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
            dataset = dataset.map(
                resample_and_save_audio,
                remove_columns=["audio"],                       # drop original audio dict column
                fn_kwargs={"target_sr": self.sampling_rate, "resampled_dir": self.resampled_dir},
                num_proc=self.num_proc_pitch,                   # Same cost shape as F0 extract (decode + pyworld)
                desc="Resampling, extracting F0 and lengths, and saving audio",
            )
            dataset = dataset.rename_column("audio_path", "audio")
            dataset = dataset.cast_column("audio", Value("string"))  # audio column = string path henceforth
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

        dataset = dataset.map(
            add_phoneme_length,
            batched=True,
            num_proc=self.num_proc_tokenize,
            desc="Computing phoneme lengths",
        )
        dataset.cleanup_cache_files()

        if self.min_phoneme_length or self.max_phoneme_length:
            logger.info(f"Filtering phoneme lengths: min={self.min_phoneme_length}, max={self.max_phoneme_length}")
            dataset = dataset.filter(
                filter_phoneme_lengths,
                input_columns=["phoneme_length"],
                batched=True,
                fn_kwargs={"min_length": self.min_phoneme_length, "max_length": self.max_phoneme_length},
                num_proc=self.num_proc_tokenize,
                desc="Filtering by phoneme length",
            )
            dataset.cleanup_cache_files()

        # `text` survives for the eval block (GT transcript vs generated). `phonemes` survives for
        # lazy tokenization in __getitem__.
        return dataset.select_columns(
            ["audio", "audio_length", "f0", "phonemes", "phoneme_length", "original_index", "text"]
        )

    def _prune_intermediate_caches(self) -> None:
        # Remove disposable map caches; keeps the reusable raw load_dataset arrow (mls_eng-<split>-*.arrow).
        for f in self.cache_dir.rglob("cache-*.arrow"):
            f.unlink(missing_ok=True)

    def _finalize_raw_cache(self) -> None:
        if self.delete_raw_cache_after_preprocess:
            logger.info(f"Deleting raw cache directory: {self.cache_dir}")
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        else:
            logger.info(f"Keeping raw cache; pruning intermediate map caches in {self.cache_dir}")
            self._prune_intermediate_caches()

    def _process_single(self) -> HFDataset:
        config_path = self.processed_dir / "preprocessing_config.json"
        try:
            dataset = load_from_disk(str(self.processed_dir))
            # Schema guard: a cache predating the tokenize decoupling lacks `phonemes` → rebuild
            # rather than KeyError later in __getitem__.
            if "phonemes" not in dataset.column_names:
                raise FileNotFoundError("cache predates tokenize decoupling; rebuilding")
            logger.info(f"Successfully loaded processed dataset from {self.processed_dir}")

            if config_path.is_file():
                with open(config_path, "r") as f:
                    cached_config = json.load(f)
                mismatches = [
                    f"{k}: requested={v}, cached={cached_config.get(k)}"
                    for k, v in self.preprocessing_config.items()
                    if k not in cached_config or cached_config[k] != v
                ] + [
                    f"{k}: requested=<removed>, cached={v}"
                    for k, v in cached_config.items()
                    if k not in self.preprocessing_config
                ]
                if mismatches:
                    logger.warning(
                        "\n" + "=" * 60 + "\n"
                        "⚠️ PREPROCESSING CONFIGURATION MISMATCH ⚠️\n"
                        "The cached dataset was loaded, but it was created with different parameters:\n"
                        + "\n".join(f"  - {m}" for m in mismatches) + "\n\n"
                        f"If you want to apply your new settings, manually delete:\n  {self.processed_dir}\n"
                        + "=" * 60
                    )
            else:
                logger.info("No preprocessing_config.json found. Cannot verify parameters.")
            return dataset
        except (FileNotFoundError, json.JSONDecodeError, pa.ArrowInvalid, pa.ArrowIOError) as e:
            if not self.build_if_missing:
                raise FileNotFoundError(
                    f"No preprocessed cache at {self.processed_dir} (build_if_missing=False). "
                    f"Preprocess this split with these settings first."
                ) from e
            logger.info(f"Local dataset unavailable or corrupted ({type(e).__name__}). Triggering preprocessing...")

            dataset = self._load_dataset_split()
            dataset = dataset.add_column("original_index", range(len(dataset)))   # global raw index
            dataset.cleanup_cache_files()

            # Legacy non-chunked subset (dev/test pass max_train_clips=None → no-op).
            if self.max_train_clips is not None and len(dataset) > self.max_train_clips:
                dataset = dataset.shuffle(seed=self.subset_seed).select(range(self.max_train_clips))
                logger.info(f"Subset '{self.split}' to {self.max_train_clips} clips (seed={self.subset_seed}).")

            dataset = self._preprocess_untokenized(dataset)

            logger.info(f"Saving processed dataset to {self.processed_dir} ...")
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(self.processed_dir))
            with open(config_path, "w") as f:
                json.dump(self.preprocessing_config, f, indent=4)

            del dataset
            gc.collect()
            self._finalize_raw_cache()

            logger.info(f"Loading standalone dataset from {self.processed_dir} into memory...")
            return load_from_disk(str(self.processed_dir))

    # ---- chunked train store: vocab-free shards of a seeded permutation (pay-as-you-go ladder) ----

    def _untokenized_config(self) -> dict:
        # The preprocessing settings that define a shard's CONTENT (everything in
        # preprocessing_config except the subset cap + shuffle seed, tracked separately). A change
        # here must invalidate the chunked store (meta.json guard).
        skip = {"max_train_clips", "subset_seed"}
        return {k: v for k, v in self.preprocessing_config.items() if k not in skip}

    def _read_chunks_meta(self) -> Optional[dict]:
        meta_path = self.chunks_dir / "meta.json"
        if not meta_path.is_file():
            return None
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("chunk_size") != self.chunk_size:
            raise ValueError(
                f"chunk_size changed ({meta.get('chunk_size')} → {self.chunk_size}); "
                f"delete {self.chunks_dir} and rebuild."
            )
        if meta.get("subset_seed") != self.subset_seed:
            raise ValueError(f"subset_seed changed; delete {self.chunks_dir} and rebuild.")
        if meta.get("untokenized_config") != self._untokenized_config():
            raise ValueError(
                f"Preprocessing settings changed; delete {self.chunks_dir} and rebuild.\n"
                f"  cached={meta.get('untokenized_config')}\n  current={self._untokenized_config()}"
            )
        return meta

    def _write_chunks_meta(self, total_clips: int) -> dict:
        meta = {
            "chunk_size": self.chunk_size,
            "subset_seed": self.subset_seed,
            "total_clips": total_clips,
            "untokenized_config": self._untokenized_config(),
        }
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(self.chunks_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        return meta

    def _load_raw_and_perm(self) -> tuple[HFDataset, np.ndarray]:
        raw = self._load_dataset_split()
        # original_index = raw row position, added ONCE on the full unindexed raw (zero-copy, no audio
        # read); select() in _build_chunk preserves it per shard. Our own canonical shuffle (pinned to
        # numpy, immune to HF shuffle() internals): permuted position i is raw row perm[i].
        raw = raw.add_column("original_index", list(range(len(raw))))
        perm = np.random.default_rng(self.subset_seed).permutation(len(raw))
        return raw, perm

    def _build_chunk(self, raw: HFDataset, perm: np.ndarray, k: int, final_dir: Path) -> None:
        # Sorted so the gather reads ascending row positions (near-sequential across the mmap shards)
        # instead of a random scatter; within-chunk order is irrelevant (the loader shuffles, and
        # original_index is preserved regardless).
        idx = np.sort(perm[k * self.chunk_size : min((k + 1) * self.chunk_size, len(raw))]).tolist()
        # Materialize ONLY the selected rows before preprocessing. raw.select() attaches a chunk-sized
        # indices mask but leaves _data = the full ~10.8M-row mmap table, so the downstream cast/F0/
        # phonemize ops — and the main-process flatten that .map(num_proc>1) runs on an indices view —
        # operate against the whole table's backing store rather than the 50k-row selection. flatten_indices
        # rewrites just the selected rows to a fresh arrow (streamed, writer_batch_size rows at a time),
        # so every downstream op sees only this chunk.
        sub = raw.select(idx).flatten_indices(keep_in_memory=False)   # select carries original_index
        chunk = self._preprocess_untokenized(sub)
        # Atomic build: write into chunk_k.tmp, drop a COMPLETE sentinel, then os.replace. A crash
        # mid-build leaves a .tmp (cleared next run); load never sees a half-written "complete" shard.
        tmp = final_dir.with_name(final_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        chunk.save_to_disk(str(tmp))
        (tmp / "COMPLETE").write_text("ok")
        os.replace(tmp, final_dir)
        del chunk
        sub.cleanup_cache_files()   # drop this chunk's flatten arrow so it doesn't accumulate over the ladder
        gc.collect()
        logger.info(f"Built {final_dir.name} ({len(idx)} clips pre-filter).")

    def _process_train_chunked(self) -> HFDataset:
        # Read-only consumer: no store at all → fail loud instead of building one.
        if not self.build_if_missing and not (self.chunks_dir / "meta.json").is_file():
            raise FileNotFoundError(
                f"No chunked train store at {self.chunks_dir} (build_if_missing=False). "
                f"Preprocess the train split first."
            )
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        C, N = self.chunk_size, self.max_train_clips
        meta = self._read_chunks_meta()
        raw = perm = None
        if meta is None:                              # first build: need total_clips for the ceiling
            raw, perm = self._load_raw_and_perm()
            meta = self._write_chunks_meta(total_clips=len(raw))
        max_k = math.ceil(meta["total_clips"] / C)

        chunks, rows, k, built = [], 0, 0, False
        while (N is None or rows < N) and k < max_k:
            cdir = self.chunks_dir / f"chunk_{k:05d}"
            if not (cdir / "COMPLETE").exists():
                if not self.build_if_missing:
                    raise FileNotFoundError(
                        f"Chunked train store incomplete at {self.chunks_dir} (missing {cdir.name}, "
                        f"build_if_missing=False). Finish preprocessing, or lower dataset.max_train_clips "
                        f"to the already-built amount."
                    )
                if cdir.exists():
                    shutil.rmtree(cdir)               # clear an aborted partial
                if raw is None:
                    raw, perm = self._load_raw_and_perm()
                self._build_chunk(raw, perm, k, cdir)
                built = True
            ds_k = load_from_disk(str(cdir))
            chunks.append(ds_k)
            rows += len(ds_k)
            k += 1

        if built:
            self._finalize_raw_cache()
        dataset = concatenate_datasets(chunks)
        if N is not None and len(dataset) > N:        # exact prefix; len() guard avoids IndexError
            dataset = dataset.select(range(N))
        logger.info(f"Chunked '{self.split}': {k} shard(s) → {len(dataset)} clips (target N={N}).")
        return dataset

    def _ensure_token_vocabulary(self) -> None:
        # Only the train split builds the vocab (so IDs are train-derived); dev/test require it to
        # already exist. Decoupled from the cache → rebuilding = delete the json, no reprocessing.
        if self.build_vocabulary:
            if not Path(self.token_vocabulary_path).is_file():
                logger.info(f"Building token vocabulary at {self.token_vocabulary_path} from split '{self.split}'...")
                # Project to `phonemes` so vocab-building never reads the audio column.
                build_token_vocabulary(self.dataset.select_columns(["phonemes"]), save_path=str(self.token_vocabulary_path))
            else:
                logger.info(f"Using existing token vocabulary at {self.token_vocabulary_path}.")
        elif not Path(self.token_vocabulary_path).is_file():
            raise FileNotFoundError(
                f"Token vocabulary {self.token_vocabulary_path} is missing. Preprocess the train split "
                f"first — only it builds the vocab (keeps IDs train-derived)."
            )

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
        # Lazy, vocab-DEPENDENT tokenization (phonemes → IDs); ~50 dict lookups, free vs audio decode.
        phoneme_tokens = self.tokenizer(item["phonemes"])

        return {
            "audio": audio, # [T]
            "pitch": torch.tensor(item["f0"], dtype=torch.float32), # [F]
            "phoneme_tokens": phoneme_tokens,
            "phoneme_tokens_length": item["phoneme_length"],   # == len(phonemes)
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
        seed: int = 42,
        validate_phoneme_caps: bool = True,
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

        # phoneme_length is a HARD collate cap; fail loud here (mirrors the audio guard above) instead
        # of mid-run in BucketedCollateFn. measure_bucket_phoneme_lengths.py passes False — it reports
        # every bucket itself rather than dying on the first overflow.
        if validate_phoneme_caps:
            self._assert_phoneme_caps_fit()

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

    def _assert_phoneme_caps_fit(self) -> None:
        """Raise if any clip's phoneme count exceeds its audio-bucket's phoneme_length cap."""
        phoneme_lengths = np.asarray(self.dataset.dataset["phoneme_length"])
        for b_idx, indices in self.bucket_to_indices.items():
            cap = self.bucket_mapping[b_idx].get("phoneme_length")
            if cap is None or not indices:
                continue
            bucket_phon = phoneme_lengths[indices]
            mx = int(bucket_phon.max())
            if mx > cap:
                worst = indices[int(np.argmax(bucket_phon))]
                raise ValueError(
                    f"Sequence at index {worst} has {mx} phonemes, exceeding bucket {b_idx}'s "
                    f"phoneme_length cap ({cap}). Raise it to >= {mx} — "
                    f"scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py reports the per-bucket max."
                )

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
