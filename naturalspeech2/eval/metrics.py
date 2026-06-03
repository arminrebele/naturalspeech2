"""Objective TTS eval metrics — consumer-agnostic, off the autograd path.

WER (intelligibility) via HuBERT-Large CTC, no LM (`facebook/hubert-large-ls960-ft`):
no-LM CTC transcribes acoustics faithfully instead of letting a decoder LM paper over
mispronunciations. Lazy module-level singleton on the idle 2nd GPU (off the training
card's budget), FP32, no_grad. Standard word-level Levenshtein(ref,hyp)/len(ref),
shared ref/hyp normalization. SIM-o (speaker similarity) deferred (s3prl/numpy-2.x).
"""
import re
from functools import lru_cache

import numpy as np
import torch
import torchaudio.functional as taF

_ASR_NAME = "facebook/hubert-large-ls960-ft"
_ASR_SR = 16000
_MIN_SAMPLES = _ASR_SR // 4   # <0.25 s → degenerate (early-training garbage); skip → NaN


_METRIC_DEVICE_OVERRIDE: str | None = None


def resolve_metric_device(value: str) -> str:
    """Resolve a config `metric_device`. 'auto' → idle 2nd GPU if present, else CPU — never the
    training card unless named explicitly (eval is infrequent; CPU beats stealing train VRAM).
    Any other value passes through verbatim (e.g. 'cuda:0', 'cuda:1', 'cpu')."""
    if value != "auto":
        return value
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cpu"


def set_metric_device(device: str | None) -> None:
    """Pin the ASR device; call once at startup before the first WER. Clears the cached ASR
    singleton so a re-pin re-homes it. None → revert to the resolved 'auto' default.
    Daemon sets its own card ('cuda:0' under CUDA_VISIBLE_DEVICES=1 == physical GPU1)."""
    global _METRIC_DEVICE_OVERRIDE
    _METRIC_DEVICE_OVERRIDE = device
    _load_asr.cache_clear()


def _metric_device() -> str:
    """Override if pinned (config/daemon), else the resolved 'auto' default (idle 2nd GPU / CPU)."""
    return _METRIC_DEVICE_OVERRIDE if _METRIC_DEVICE_OVERRIDE is not None else resolve_metric_device("auto")


@lru_cache(maxsize=1)
def _load_asr():
    from transformers import HubertForCTC, Wav2Vec2Processor
    device = _metric_device()
    proc = Wav2Vec2Processor.from_pretrained(_ASR_NAME)
    model = HubertForCTC.from_pretrained(_ASR_NAME).to(device).eval()
    return proc, model, device


def _to_numpy(audio) -> np.ndarray:
    return audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)


def _prep_16k(audio, src_sr: int) -> np.ndarray:
    """numpy/torch @ src_sr → 16 kHz float32 numpy (gen audio is numpy; reference may be torch)."""
    wav = torch.from_numpy(np.ascontiguousarray(_to_numpy(audio))).float()
    if src_sr != _ASR_SR:
        wav = taF.resample(wav, src_sr, _ASR_SR)
    return wav.numpy()


def _norm_text(s: str) -> list[str]:
    """Lowercase, strip punct, split → words. Applied to BOTH ref+hyp (HuBERT-CTC emits
    uppercase/no-punct → raw compare inflates WER). Digits kept (MLS/VCTK text pre-verbalized)."""
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def _word_error_rate(ref_words: list[str], hyp_words: list[str]) -> float:
    """Standard WER = word-level Levenshtein(ref, hyp) / len(ref). NaN if ref empty."""
    if not ref_words:
        return float("nan")
    prev = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, start=1):
        cur = [i] + [0] * len(hyp_words)
        for j, hw in enumerate(hyp_words, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(ref_words)


def _degenerate(audio) -> bool:
    arr = _to_numpy(audio)
    return arr.size < _MIN_SAMPLES or not np.isfinite(arr).all()


@torch.no_grad()
def transcribe(audio, src_sr: int = 24000) -> str:
    proc, model, device = _load_asr()
    wav = _prep_16k(audio, src_sr)
    iv = proc(wav, sampling_rate=_ASR_SR, return_tensors="pt").input_values.to(device)
    return proc.batch_decode(model(iv).logits.argmax(-1))[0]


@torch.no_grad()
def compute_wer(gen_audio, target_text: str, src_sr: int = 24000) -> float:
    """WER of ASR(gen_audio) vs target_text. NaN on degenerate audio (drops out of
    np.nanmean instead of crashing/skewing)."""
    if _degenerate(gen_audio):
        return float("nan")
    hyp = transcribe(gen_audio, src_sr)
    return _word_error_rate(_norm_text(target_text), _norm_text(hyp))
